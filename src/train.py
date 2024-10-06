import json
import tensorflow as tf
import yaml
from model import create_model, compile_model, train_model, evaluate_model
from data_process import prepare_dataset, load_data

DATA_DIR = "data/raw"  # Path to your data directory
# Load parameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

BATCH_SIZE = params['data']['batch_size']
IMG_SIZE = tuple(params['data']['img_size']) + (3,)  # Ensure correct shape
NUM_CLASSES =  4 # Set this according to your specific case

# Load and split the data
train_dataset, val_dataset, test_dataset = load_data(DATA_DIR, IMG_SIZE, BATCH_SIZE)

# Prepare datasets
train_dataset = prepare_dataset(train_dataset)
val_dataset = prepare_dataset(val_dataset)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Create, compile, train, and evaluate the model as before...


# Prepare datasets with normalization and prefetching
train_dataset = prepare_dataset(train_dataset)
val_dataset = prepare_dataset(val_dataset)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# 1. Create the model
model = create_model(IMG_SIZE, NUM_CLASSES)

# 2. Compile the model
compile_model(model)

# 3. Train the model
history = train_model(model, train_dataset, val_dataset, epochs=10)

# 4. Evaluate the model on the test set
test_loss, test_accuracy = evaluate_model(model, test_dataset)
