"""Specify the classification."""

import tensorflow as tf

class Specs:
    """Specify the classification problem."""

    def __init__(self) -> None:
        """Initialize with default values."""
        self.classes = ['attending', 'unattending', 'meditating']
        self.class_set = set(self.classes)
        self.class_indices = dict(zip(self.classes, range(len(self.classes))))

        self.image_size = 224
