import os
import tensorflow as tf

# Parameters
BATCH_SIZE = 32
IMG_SIZE = (224, 224)  # Resize images to this size
SEED = 42
TEST_SPLIT = 0.1
VAL_SPLIT = 0.2
DATA_DIR = "data/raw"  # Path where raw data is stored

def load_data(data_dir, img_size, batch_size):
    """
    Load and split the dataset into training, validation, and test sets.
    """
    # Load the full dataset without splitting
    full_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',  # One-hot encoded labels for multi-class
        seed=SEED,
        shuffle=True  # Shuffle the data
    )

    # Calculate dataset sizes
    dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
    test_size = int(TEST_SPLIT * dataset_size)
    val_size = int(VAL_SPLIT * (dataset_size - test_size))

    # Split dataset into train, validation, and test sets
    test_dataset = full_dataset.take(test_size)
    remaining_dataset = full_dataset.skip(test_size)

    val_dataset = remaining_dataset.take(val_size)
    train_dataset = remaining_dataset.skip(val_size)

    return train_dataset, val_dataset, test_dataset


