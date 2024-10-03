import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.model_selection import train_test_split
import os

# Parameters
BATCH_SIZE = 32
IMG_SIZE = (224, 224)  # ResNet and similar models typically use 224x224 images
SEED = 42
TEST_SPLIT = 0.1
VAL_SPLIT = 0.2
DATA_DIR = "data/raw/Brain_Tumor"  # Path where DVC-tracked raw data is stored

def load_data(data_dir, img_size, batch_size):
    """
    Function to load and split data into training, validation, and test sets.
    """
    # Load full dataset without split
    full_dataset = image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        seed=SEED,
        label_mode='categorical',  # Multiclass classification
        shuffle=True
    )

    # Calculate dataset size
    dataset_size = len(full_dataset)
    test_size = int(TEST_SPLIT * dataset_size)
    val_size = int(VAL_SPLIT * (dataset_size - test_size))

    # Split dataset into training, validation, and test sets
    test_dataset = full_dataset.take(test_size)
    remaining_dataset = full_dataset.skip(test_size)

    val_dataset = remaining_dataset.take(val_size)
    train_dataset = remaining_dataset.skip(val_size)

    return train_dataset, val_dataset, test_dataset
