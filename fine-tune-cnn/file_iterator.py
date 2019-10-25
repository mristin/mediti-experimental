"""Iterate over a mediti data set."""

import pathlib
import re
from typing import List, Optional, Tuple

import icontract
import keras_preprocessing.image
import numpy as np
import tensorflow as tf

import specsmod

_FORMAT_RE = re.compile(
    r'^(?P<identifier>[ a-zA-Z_0-9-]+)'
    r'\.'
    r'(?P<timestamp>'
    r'[0-9]{4}-[0-9]{2}-[0-9]{2}T'
    r'[0-9]{2}-[0-9]{2}-[0-9]{2}(.[0-9]+)?)Z'
    r'\.'
    r'(?P<label>[ a-zA-Z0-9_-]+)'
    r'\.'
    r'(jpg|JPG|jpeg|JPEG)$')


def _extract_label(filename: str) -> Optional[str]:
    """
    Parse the file name and return the label if the file name matches.

    :param filename: to be matched
    :return: label, if matched

    >>> _extract_label("not-matching")

    >>> _extract_label(
    ...     "some-identifier.2016-07-03T21-22-23.123456Z.attending.jpg")
    'attending'
    """
    mtch = _FORMAT_RE.match(filename)
    if mtch is None:
        return None

    lbl = mtch.group("label")

    return lbl


@icontract.require(lambda directory: directory.exists() and directory.is_dir())
@icontract.ensure(lambda result: len(result[0]) == len(result[1]))
@icontract.ensure(
    lambda directory, result: all(
        directory in pth.parents for pth in result[0]))
def find_images(
        directory: pathlib.Path,
        specs: specsmod.Specs) -> Tuple[List[pathlib.Path], List[str]]:
    """
    Find all the sample images beneath the directory.

    :param directory: where to search
    :param specs: specification of the problem
    :return: list of image paths, list of class labels
    """
    pths_lables = []  # type: List[Tuple[pathlib.Path, str]]

    for pth in directory.glob("**/*"):
        lbl = _extract_label(filename=pth.name)
        if not lbl:
            continue

        if lbl not in specs.class_set:
            raise ValueError("Unexpected label: {}, expected one of: {}".format(
                lbl, specs.classes))

        pths_lables.append((pth, lbl))

    pths_lables = sorted(pths_lables)

    return [pth for pth, _ in pths_lables], [lbl for _, lbl in pths_lables]


class FileIterator(
    keras_preprocessing.image.iterator.BatchFromFilesMixin,
    tf.keras.preprocessing.image.Iterator):
    """
    Iterate over images beneath the directory.

    Image names correspond to "{identifier}.{timestamp}.{label}.jpg".
    """

    ALLOWED_CLASS_MODES = {'categorical', 'binary', 'sparse', 'input', None}

    def __init__(self,
                 specs: specsmod.Specs,
                 directory: str,
                 image_data_generator,
                 target_size=(256, 256),
                 color_mode='rgb',
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 subset=None,
                 interpolation='nearest',
                 dtype='float32'):
        """See documentation of BatchFromFilesMixin."""
        super(FileIterator, self).set_processing_attrs(
            image_data_generator,
            target_size,
            color_mode,
            data_format,
            save_to_dir,
            save_prefix,
            save_format,
            subset,
            interpolation)

        self.directory = directory

        if class_mode not in FileIterator.ALLOWED_CLASS_MODES:
            raise ValueError(
                'Invalid class_mode: {}; expected one of: {}'.format(
                    class_mode, FileIterator.ALLOWED_CLASS_MODES))
        self.class_mode = class_mode

        self.dtype = dtype

        # Find image paths and parse their labels
        pths, lbls = find_images(
            directory=pathlib.Path(directory),
            specs=specs)

        lbl_set = set(lbls)

        assert lbl_set == specs.class_set, \
            "Unexpected label set: {}".format(sorted(lbl_set))

        # Figure out the classes
        self.class_indices = specs.class_indices
        self.num_classes = len(specs.classes)

        # Figure out the samples
        self._filepaths = [str(pth) for pth in pths]
        self.filenames = [pth.name for pth in pths]
        self.samples = len(self._filepaths)

        # Set the integer labels
        self.classes = np.zeros((self.samples,), dtype='int32')
        assert len(lbls) == self.samples
        for i, lbl in enumerate(lbls):
            self.classes[i] = self.class_indices[lbl]

        super(FileIterator, self).__init__(
            self.samples,
            batch_size,
            shuffle,
            seed)

    @property
    def filepaths(self):
        return self._filepaths

    @property
    def labels(self):
        return self.classes

    @property  # mixin needs this property to work
    def sample_weight(self):
        # no sample weights will be returned
        return None
