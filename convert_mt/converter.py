import os
import re
from argparse import Namespace
from pathlib import Path
import shutil
from typing import Optional, Any

import numpy as np
import torch
from ctranslate2.converters.marian import _get_model_config, _SUPPORTED_ACTIVATIONS
from ctranslate2.specs.common_spec import Activation


_INVERSE_ACTIVATION = {v: k for k, v in _SUPPORTED_ACTIVATIONS.items()}


NAME_MAP = {
    'encoder.embeddings_0.weight': 'encoder.embeddings.make_embedding.emb_luts.0.weight',
    'encoder.transformer.0.self_attn.linear_0.weight': (
        'encoder.transformer.0.self_attn.linear_query.weight',
        'encoder.transformer.0.self_attn.linear_keys.weight',
        'encoder.transformer.0.self_attn.linear_values.weight',
    ),
    'encoder.transformer.0.self_attn.linear_0.bias': (
        'encoder.transformer.0.self_attn.linear_query.bias',
        'encoder.transformer.0.self_attn.linear_keys.bias',
        'encoder.transformer.0.self_attn.linear_values.bias',
    ),
}


layer_regex = re.compile(r'(layer_)(\d)')
self_attn_regex = re.compile(r'(encoder|decoder).transformer.(\d).self_attn.linear_0.(weight|bias)')
embeddings_regex = re.compile(r'embeddings(_0)?')
decoder_layers_regex = re.compile(r'decoder.(transformer).\d.(self_attn|feed_forward|context_attn).\s*')
self_attn_norm_regex = re.compile(r'transformer.(\d).self_attn.layer_norm')
layer_norm_regex1 = re.compile(r'transformer_layers.\d.self_attn.layer_norm')
layer_norm_regex2 = re.compile(r'transformer_layers.\d.context_attn.layer_norm')
ctx_attn_regex = re.compile(r'context_attn.linear_(\d)')


def process_name(name: str) -> str:
    name = (name.replace("/", ".")
            .replace("self_attention.linear_1", "self_attn.final_linear")
            .replace("self_attention", "self_attn")
            .replace("ffn.linear_0", "feed_forward.w_1")
            .replace("ffn.linear_1", "feed_forward.w_2")
            .replace("ffn", "feed_forward")
            .replace("layer_norm.gamma", "layer_norm.weight")
            .replace("layer_norm.beta", "layer_norm.bias")
            .replace("attention", "context_attn")
            .replace("decoder.projection", "generator"))
    name = layer_regex.sub(r"transformer.\2", name)
    name = embeddings_regex.sub('embeddings.make_embedding.emb_luts.0', name)

    if 'position_encodings' in name:
        return name.replace('position_encodings.encodings', 'embeddings.make_embedding.pe.pe')
    if match := self_attn_regex.match(name):
        transformer = "transformer" if match.group(1) == "encoder" else "transformer_layers"
        name = (
            f'{match.group(1)}.{transformer}.{match.group(2)}.self_attn.linear_query.{match.group(3)}',
            f'{match.group(1)}.{transformer}.{match.group(2)}.self_attn.linear_keys.{match.group(3)}',
            f'{match.group(1)}.{transformer}.{match.group(2)}.self_attn.linear_values.{match.group(3)}',
        )
        return name
    if decoder_layers_regex.match(name):
        name = name.replace("transformer", "transformer_layers")
    if match := ctx_attn_regex.search(name):
        if match.group(1) == "0":
            return ctx_attn_regex.sub("context_attn.linear_query", name)
        elif match.group(1) == "1":
            return (
                ctx_attn_regex.sub("context_attn.linear_keys", name),
                ctx_attn_regex.sub("context_attn.linear_values", name),
            )
        elif match.group(1) == "2":
            return ctx_attn_regex.sub("context_attn.final_linear", name)
        else:
            raise ValueError(f"Wrong parsing for {name}")
    if layer_norm_regex1.search(name):
        return name.replace("self_attn.layer_norm", "layer_norm_1")
    elif layer_norm_regex2.search(name):
        return name.replace("context_attn.layer_norm", "layer_norm_2")
    elif self_attn_norm_regex.search(name):
        return name.replace("self_attn.layer_norm", "layer_norm")
    return name


def convert_tensor(name, variable):
    new_name = process_name(name)
    if isinstance(new_name, str):
        if not isinstance(variable, torch.Tensor):
            if variable.shape == ():
                yield new_name, torch.tensor(variable.array)
            else:
                torcharray = torch.from_numpy(variable.array)
                if 'pe' in new_name:
                    torcharray = torcharray.unsqueeze(1)
                    yield 'num_positional_encoding', torch.from_numpy(np.array(torcharray.size(0)))
                yield new_name, torcharray
    elif isinstance(new_name, tuple):
        variables = np.split(variable.array, len(new_name))
        for k, v in zip(new_name, variables):
            yield k, torch.from_numpy(v)


def convert_vocabularies(vocabs: dict[list[list[str]]], config):
    if config.decoder_start_token in vocabs["target"][0]:
        decoder_start_token = config.decoder_start_token
    else:
        decoder_start_token = config.eos_token
    vocabs_ = {
        "src": {x: i for i, x in enumerate(vocabs["source"][0])},
        "tgt": {x: i for i, x in enumerate(vocabs["target"][0])},
        "data_task": "seq2seq",
        "decoder_start_token": decoder_start_token,
    }
    return vocabs_


def make_opts(config):
    opt = Namespace(**config)
    opt.freeze_word_vecs_enc = False
    opt.freeze_word_vecs_dec = False
    opt.dropout = 0.0
    opt.optim = "adafactor"
    opt.encoder_type = "transformer"
    opt.decoder_type = "transformer"
    opt.enc_layers = opt.enc_depth
    opt.dec_layers = opt.dec_depth
    opt.enc_hid_size = opt.dim_emb
    opt.dec_hid_size = opt.dim_emb
    opt.final_layer_norm = False
    opt.add_qkvbias = True
    opt.src_word_vec_size = opt.dim_emb
    opt.tgt_word_vec_size = opt.dim_emb
    opt.share_embeddings = opt.tied_embeddings_all
    opt.feat_merge = "sum"
    opt.position_encoding = True
    opt.heads = getattr(opt, "encoder.num_heads")
    opt.pos_ffn_activation_fn = _INVERSE_ACTIVATION[Activation(int(getattr(opt, "decoder.activation")))]
    if opt.pos_ffn_activation_fn == "swish":
        opt.pos_ffn_activation_fn = "silu"

    return opt


def prepare_variables(variables):
    del variables["save"]
    new_params = dict(variables)
    for k, v in new_params.items():
        if isinstance(v, str):
            new_params[k] = new_params[v]
    return acquire_variables(new_params)


def acquire_variables(variable_dict):
    new_params = {}
    options = {}
    for k, v in variable_dict.items():
        for new_name, new_value in convert_tensor(k, v):
            if new_value.ndim == 0:
                options[new_name] = new_value
            else:
                new_params[new_name] = new_value
    return new_params, options


def save(self, output_dir: str, config: dict[str, Any]) -> None:
    """Saves this model on disk.

    Arguments:
      output_dir: Output directory where the model is saved.
    """
    variables, options = prepare_variables(self.variables())
    generator = {
        "weight": variables.pop('generator.weight'),
        "bias": variables.pop('generator.bias'),
    }
    config_ = {k.replace("-", "_"): v for k, v in config.items()}
    opt = make_opts(self._config.__dict__ | config_ | options)
    checkpoint = {
        "model": variables,
        "generator": generator,
        "vocab": convert_vocabularies(self._vocabularies, self._config),
        "opt": opt,
    }
    ckpt_path = Path(output_dir) / "opusmt_converted.pt"
    torch.save(checkpoint, ckpt_path)


def convert(
    self,
    output_dir: str,
    vmap: Optional[str] = None,
    quantization: Optional[str] = None,
    force: bool = False,
) -> str:
    """Converts the model to the CTranslate2 format.

    Arguments:
      self: the Converter to which this function will be bound as a method
      output_dir: Output directory where the CTranslate2 model is saved.
      vmap: Optional path to a vocabulary mapping file that will be included
        in the converted model directory.
      quantization: Weight quantization scheme (possible values are: int8, int8_float32,
        int8_float16, int8_bfloat16, int16, float16, bfloat16, float32).
      force: Override the output directory if it already exists.

    Returns:
      Path to the output directory.

    Raises:
      RuntimeError: If the output directory already exists and :obj:`force`
        is not set.
      NotImplementedError: If the converter cannot convert this model to the
        CTranslate2 format.
    """
    if os.path.exists(output_dir) and not force:
        raise RuntimeError(
            "output directory %s already exists, use --force to override"
            % output_dir
        )

    model_spec = self._load()
    if model_spec is None:
        raise NotImplementedError(
            "This model is not supported by CTranslate2 or this converter"
        )
    if vmap is not None:
        model_spec.register_vocabulary_mapping(vmap)

    model_spec.validate()
    model_spec.optimize(quantization=quantization)

    # Create model directory.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Keep it down here, or the "save" function would be added to its "variables" and fail during the call to optimize
    patch_converter(model_spec, "save")
    model = np.load(self._model_path)
    config = _get_model_config(model)
    model_spec.save(output_dir, config)
    return output_dir


def patch_converter(obj, fun_name="convert"):
    fun = globals()[fun_name]
    setattr(obj, fun_name, lambda *args, **kwargs: fun(obj, *args, **kwargs))
    return obj
