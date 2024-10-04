import os
import tensorflow as tf

BATCH_SIZE = 32
IMG_SIZE = (224, 224)  # Resize images to this size
DATA_DIR = "data/raw"  # Path to your data directory
PROCESSED_DATA_DIR = "data/processed"  # Path for processed data

def load_and_split_data(data_dir, img_size, batch_size):
    """
    Load and split the dataset into train, validation, and test sets.
    """
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',  # One-hot encoded labels for multi-class
        seed=42,
        validation_split=0.2,  # Use 20% for validation
        subset='training'  # This will be the training dataset
    )

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        seed=42,
        validation_split=0.2,
        subset='validation'  # This will be the validation dataset
    )

    # If you have a separate test dataset, load it here
    # For now, we'll keep it as validation just for demonstration
    test_dataset = val_dataset  # Replace this if you have a separate test dataset

    return dataset, val_dataset, test_dataset

def prepare_dataset(dataset):
    """
    Prepare the dataset by normalizing and batching it.
    """
    dataset = dataset.map(lambda x, y: (x / 255.0, y))  # Normalize images to [0, 1]
    return dataset.batch(BATCH_SIZE)

def save_datasets(train_dataset, val_dataset, test_dataset):
    """
    Save the datasets to TFRecord files.
    """
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    
    # Example of saving datasets (you can save in other formats as well)
    train_dataset.save(os.path.join(PROCESSED_DATA_DIR, 'train_dataset'))
    val_dataset.save(os.path.join(PROCESSED_DATA_DIR, 'val_dataset'))
    test_dataset.save(os.path.join(PROCESSED_DATA_DIR, 'test_dataset'))

    print("Datasets saved successfully!")

def main():
    # Load and prepare the data
    train_dataset, val_dataset, test_dataset = load_and_split_data(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    print("Data loaded successfully!")

    # Prepare datasets (normalization and batching)
    train_dataset = prepare_dataset(train_dataset)
    val_dataset = prepare_dataset(val_dataset)
    test_dataset = prepare_dataset(test_dataset)
    print("Datasets prepared successfully!")

    # Print dataset information
    print(f"Train dataset: {train_dataset.cardinality()} batches")
    print(f"Validation dataset: {val_dataset.cardinality()} batches")
    print(f"Test dataset: {test_dataset.cardinality()} batches")

    # Save the datasets to the processed data directory
    save_datasets(train_dataset, val_dataset, test_dataset)

if __name__ == "__main__":
    main()
