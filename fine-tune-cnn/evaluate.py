#!/usr/bin/env python3

"""Evaluate the classification model."""

import argparse
import os
import pathlib
from typing import List

import icontract
import numpy as np
import sklearn.metrics
import tabulate
import tensorflow as tf

import file_iterator
import specsmod


@icontract.require(lambda confusion_matrix: (confusion_matrix >= 0.0).all())
@icontract.ensure(
    lambda confusion_matrix, result:
    len(result) == confusion_matrix.shape[0] + 1)
@icontract.ensure(
    lambda confusion_matrix, result: all(
        len(row) == confusion_matrix.shape[1] + 1 for row in result))
def confusion_matrix_report(
        confusion_matrix: np.ndarray,
        target_names: List[str]) -> List[List[str]]:
    """
    Render a confusion matrix to a table.

    :param confusion_matrix: source to be rendered
    :param target_names: class names
    :return: rows of table cells

    >>> tbl = confusion_matrix_report(
    ...     confusion_matrix=np.array(
    ...         [[13,  0,  0],
    ...          [ 0, 10,  6],
    ...          [ 0,  0,  9]]),
    ...     target_names=["red", "blue", "green"])
    >>> assert tbl == [
    ...     ['Estimated    →\\nGround-truth ↓', 'red', 'blue', 'green'],
    ...     ['red',  '13 (100.0%)',  '0 (  0.0%)', '0 (  0.0%)'],
    ...     ['blue',  '0 (  0.0%)', '10 ( 62.5%)', '6 ( 37.5%)'],
    ...     ['green', '0 (  0.0%)',  '0 (  0.0%)', '9 (100.0%)']]
    """
    class_samples = confusion_matrix.sum(axis=1)
    result = [
        ["Estimated    →\nGround-truth ↓"] + target_names
    ]

    for i in range(confusion_matrix.shape[0]):
        row = [target_names[i]]
        for j in range(confusion_matrix.shape[1]):
            row.append("{} ({:5.1f}%)".format(
                confusion_matrix[i, j],
                (confusion_matrix[i, j] / class_samples[i]) * 100.0))

        result.append(row)

    return result


def main() -> None:
    """Execute the main routine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test_dir",
        help="Directory with the test data set",
        default=os.path.expanduser("~/mediti-train/data/test"))

    parser.add_argument(
        "--model_path",
        help="Path to the trained keras model",
        default=os.path.expanduser(
            "~/mediti-train/model/fine-tuned-mobilenet.keras"))

    args = parser.parse_args()

    test_dir = pathlib.Path(args.test_dir)
    model_path = pathlib.Path(args.model_path)

    if not test_dir.exists():
        raise FileNotFoundError(
            "Directory with the test data does not exist: {}".format(test_dir))

    if not test_dir.is_dir():
        raise NotADirectoryError(
            "Test data expected in a directory, but got: {}".format(
                test_dir))

    if not model_path.exists():
        raise FileNotFoundError("Model not found: {}".format(model_path))

    if not model_path.is_file():
        raise RuntimeError("Expected model in a file, but got: {}".format(
            model_path))

    ##
    # Load the model
    ##

    print("Loading the model...")

    model = tf.keras.models.load_model(filepath=str(model_path))
    specs = specsmod.Specs()

    ##
    # Evaluate
    ##

    print("Creating test set...")

    test_generator = file_iterator.FileIterator(
        shuffle=False,
        specs=specs,
        directory=str(test_dir),
        image_data_generator=tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=specs.preprocess_input),
        target_size=(specs.image_size, specs.image_size),
        batch_size=32,
        class_mode='categorical')

    print("There are {} test image(s).".format(test_generator.samples))

    print("Predicting...")

    scores = model.predict_generator(
        test_generator,
        steps=test_generator.samples / test_generator.batch_size)

    estimated = np.argmax(scores, axis=1)

    ##
    # Report
    ##

    indices_classes = sorted(
        (idx, cls) for cls, idx in specs.class_indices.items())

    target_names = [cls for _, cls in indices_classes]

    print('Classification Report')
    print(sklearn.metrics.classification_report(
        test_generator.classes, estimated, target_names=target_names))

    conf_mat = sklearn.metrics.confusion_matrix(
        test_generator.classes, estimated)

    table = confusion_matrix_report(
        confusion_matrix=conf_mat, target_names=target_names)

    print("Confusion Matrix")
    print(tabulate.tabulate(
        tabular_data=table, tablefmt="grid",
        colalign=["left"] + ["right"] * conf_mat.shape[1]))


if __name__ == "__main__":
    main()
