#!/usr/bin/env python3

"""Fine-tune a model on mediti train/val/test split."""
import argparse
import os
import pathlib

import tensorflow as tf

import file_iterator
import specsmod


def define_model(specs: specsmod.Specs) -> tf.keras.Model:
    """
    Adapt the base model with trainable layers.

    :param specs: specification of the classification
    :return:
    """
    """Adapt the base model with trainable layers."""
    vgg_conv = tf.keras.applications.vgg16.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(specs.image_size, specs.image_size, 3))

    # Freeze the layers except the last 4 layers
    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False

    # Create the model
    model = tf.keras.models.Sequential()

    # Add the vgg convolutional base model
    model.add(vgg_conv)

    # Add new layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(len(specs.classes), activation="softmax"))

    return model


def main() -> None:
    """Execute the main routine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train_dir",
        help="Directory with the training data set",
        default=os.path.expanduser("~/mediti-train/data/train"))

    parser.add_argument(
        "--validation_dir",
        help="Directory with the validation data set",
        default=os.path.expanduser("~/mediti-train/data/val"))

    parser.add_argument(
        "--model_path",
        help="Where to save the trained model",
        default=os.path.expanduser(
            "~/mediti-train/model/fine-tuned-vgg16.keras"))

    args = parser.parse_args()

    train_dir = pathlib.Path(args.train_dir)
    val_dir = pathlib.Path(args.validation_dir)
    model_path = pathlib.Path(args.model_path)

    if not train_dir.exists():
        raise FileNotFoundError(
            "Directory with the training set does not exist: {}".format(
                train_dir))

    if not val_dir.exists():
        raise FileNotFoundError(
            "Directory with the validation set does not exist: {}".format(
                val_dir))

    ##
    # Specify the problem and the model
    ##

    preprocess_input = tf.keras.applications.vgg16.preprocess_input

    specs = specsmod.Specs()

    model = define_model(specs=specs)
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'])

    ##
    # Prepare training data
    ##

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preprocess_input)

    train_generator = file_iterator.FileIterator(
        specs=specs,
        directory=str(train_dir),
        image_data_generator=train_datagen,
        target_size=(specs.image_size, specs.image_size),
        batch_size=32,
        class_mode='categorical')

    print("There are: {} training image(s).".format(train_generator.samples))

    ##
    # Prepare validation data
    ##

    val_generator = file_iterator.FileIterator(
        specs=specs,
        directory=str(val_dir),
        image_data_generator=tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_input),
        target_size=(specs.image_size, specs.image_size),
        batch_size=32,
        class_mode='categorical')

    if val_generator.class_indices != train_generator.class_indices:
        raise RuntimeError(
            "Class indices of the validation set do not match "
            "the class indices of the training set.\n"
            "Got: {}\n, but expected:\n{}".format(
                val_generator.class_indices, train_generator.class_indices))

    print("There are {} validation image(s).".format(val_generator.samples))

    ##
    # Train
    ##

    print("Training...")

    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True)

    save_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(model_path),
        monitor='val_categorical_accuracy',
        verbose=1,
        save_best_only=True)

    model.fit_generator(
        train_generator,
        # Go once through the whole training set at every epoch
        steps_per_epoch=train_generator.samples / train_generator.batch_size,
        epochs=10,
        validation_data=val_generator,
        # Go once through the whole validation set at every epoch
        validation_steps=val_generator.samples / val_generator.batch_size,
        callbacks=[save_best])


if __name__ == "__main__":
    main()
