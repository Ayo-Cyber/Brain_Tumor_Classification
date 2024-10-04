from model import create_model, compile_model , train_model, evaluate_model
from data_process import prepare_dataset
from loading import load_data


# Constants
IMG_SIZE = (224, 224, 3)
NUM_CLASSES = 4  # Adjust based on the number of classes
DATA_DIR = "data/raw/Brain_Tumor"  # Path where DVC-tracked raw data is stored
BATCH_SIZE = 32

# Load and split the data into train, validation, and test sets
train_dataset, val_dataset, test_dataset = load_data(DATA_DIR, IMG_SIZE, BATCH_SIZE)

# Prepare datasets with augmentation and prefetching
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
