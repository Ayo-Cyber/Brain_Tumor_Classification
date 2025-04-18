import os
import tensorflow as tf


BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
SEED = 42


def data_generator(train_dir, test_dir):
    train_generator = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        seed=SEED
    )

    val_generator = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        validation_split=0.3,
        subset='validation',
        seed=SEED
    )

    test_generator = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        validation_split=0.3,
        subset='training',
        seed=SEED
    )

    return train_generator, val_generator, test_generator