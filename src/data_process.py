import os
import tensorflow as tf
from loading import load_data  # Assuming your loading function is in data_loading.py

# Parameters
BATCH_SIZE = 32
IMG_SIZE = (224, 224)  # ResNet and similar models typically use 224x224 images
PROCESSED_DATA_DIR = "data/processed"  # Path to save processed data
DATA_DIR = "data/raw"  # Path where raw data is stored

def serialize_example(image, label):
    """
    Create a tf.train.Example from image and label.
    """
    # Cast image to uint8
    image = tf.cast(image * 255.0, tf.uint8)  # Assuming image is in [0, 1], scale to [0, 255]

    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(image).numpy()])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label.numpy()]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def save_datasets(train_dataset, val_dataset, test_dataset):
    """
    Save the datasets to TFRecord format.
    """
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    # Save training dataset
    with tf.io.TFRecordWriter(os.path.join(PROCESSED_DATA_DIR, 'train_dataset.tfrecord')) as writer:
        for image_batch, label_batch in train_dataset:
            for i in range(len(label_batch)):
                tf_example = serialize_example(image_batch[i], label_batch[i])
                writer.write(tf_example.SerializeToString())

    # Save validation dataset
    with tf.io.TFRecordWriter(os.path.join(PROCESSED_DATA_DIR, 'val_dataset.tfrecord')) as writer:
        for image_batch, label_batch in val_dataset:
            for i in range(len(label_batch)):
                tf_example = serialize_example(image_batch[i], label_batch[i])
                writer.write(tf_example.SerializeToString())

    # Save test dataset
    with tf.io.TFRecordWriter(os.path.join(PROCESSED_DATA_DIR, 'test_dataset.tfrecord')) as writer:
        for image_batch, label_batch in test_dataset:
            for i in range(len(label_batch)):
                tf_example = serialize_example(image_batch[i], label_batch[i])
                writer.write(tf_example.SerializeToString())

    print("Datasets saved successfully in TFRecord format!")


def prepare_dataset(dataset):
    """
    Prepare the dataset by normalizing and batching it.
    """
    dataset = dataset.map(lambda x, y: (x / 255.0, y))  # Normalize images to [0, 1]
    return dataset.batch(BATCH_SIZE)

def main():
    # Load data from the other file
    train_dataset, val_dataset, test_dataset = load_data(DATA_DIR, IMG_SIZE, BATCH_SIZE)
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
