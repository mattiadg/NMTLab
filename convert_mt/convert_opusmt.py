"""
Code mostly copied from CTranslate2 and OpenNMT-py. Both projects have a MIT license.
"""
import argparse

from ctranslate2.converters import OpusMTConverter
from ctranslate2.specs.model_spec import ACCEPTED_MODEL_TYPES

from convert_mt.converter import patch_converter


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_dir", required=True, help="Path to the OPUS-MT model directory."
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output model directory."
    )
    parser.add_argument(
        "--vocab_mapping", default=None, help="Vocabulary mapping file (optional)."
    )
    parser.add_argument(
        "--quantization",
        default=None,
        choices=ACCEPTED_MODEL_TYPES,
        help="Weight quantization type.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force conversion even if the output directory already exists.",
    )
    args = parser.parse_args()
    converter = patch_converter(OpusMTConverter(args.model_dir))
    converter.convert_from_args(args)


if __name__ == "__main__":
    main()
