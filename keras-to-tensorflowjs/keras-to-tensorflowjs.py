#!/usr/bin/env python3

"""Quantize and convert a keras model to tensorflowjs model."""

import argparse
import pathlib
import tempfile

import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs


def main() -> None:
    """Execute the main routine."""
    ##
    # Parse command-line arguments
    ##

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keras_path",
        help="Path to the keras model",
        required=True)

    parser.add_argument(
        "--tfjs_dir",
        help="Path to the tensorflowjs model",
        required=True)

    args = parser.parse_args()

    keras_path = pathlib.Path(args.keras_path)
    tfjs_dir = pathlib.Path(args.tfjs_dir)

    ##
    # Check command-line arguments
    ##

    if not keras_path.exists():
        raise FileNotFoundError("Keras model does not exist: {}".format(
            keras_path))

    if not keras_path.is_file():
        raise RuntimeError("Expected keras model to be a file: {}".format(
            keras_path))

    if tfjs_dir.exists() and not tfjs_dir.is_dir():
        raise NotADirectoryError(
            "Expected tfjs model path to be a directory: {}".format(
                tfjs_dir))

    ##
    # Prepare
    ##

    tfjs_dir.mkdir(exist_ok=True, parents=True)

    ##
    # Quantize and convert
    #
    # (paraphrased from tensorflowjs/converters/converter.py
    ##

    model = tf.keras.models.load_model(str(keras_path))

    with tempfile.TemporaryDirectory(
            prefix='savedmodel', dir=str(tfjs_dir)) as temp_savedmodel_dir:
        tf.keras.experimental.export_saved_model(
            model, str(temp_savedmodel_dir), serving_only=True)

        quantization_dtype = np.uint8  # quantize to 1-byte
        skip_op_check = False  # default
        strip_debug_ops = False  # default

        tfjs.converters.tf_saved_model_conversion_v2.convert_tf_saved_model(
            str(temp_savedmodel_dir),
            str(tfjs_dir),
            signature_def='serving_default',
            saved_model_tags='serve',
            quantization_dtype=quantization_dtype,
            skip_op_check=skip_op_check,
            strip_debug_ops=strip_debug_ops)


if __name__ == "__main__":
    main()
