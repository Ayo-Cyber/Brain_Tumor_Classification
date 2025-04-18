import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import mlflow
import mlflow.tensorflow
from utils import data_generator
from datetime import datetime

BATCH_SIZE = 32  # Batch size for training
IMAGE_SIZE = (224, 224)  # Image size for the model
SEED = 42  # Random seed for reproducibility
MODEL_DIR = "model_artifacts"  # Directory to store models
os.makedirs(MODEL_DIR, exist_ok=True)  # Create the folder if it doesn't exist
BASE_DIR = "/Users/admin/Documents/GitHub/Brain_Tumor_Classification/data/"
TRAIN_DIR = os.path.join(BASE_DIR, "Training")
TEST_DIR = os.path.join(BASE_DIR, "Testing")


def build_model(train_generator, val_generator):
    mlflow.tensorflow.autolog()

    with mlflow.start_run(run_name='Brain Disease Classification'):

        # Load base model
        base_model = ResNet50(include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False

        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        predictions = Dense(4, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(filepath=os.path.join(MODEL_DIR, 'best_model.keras'),
                            save_best_only=True, monitor='val_loss')
        ]

        # Train model
        model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=20,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )

        # Generate timestamp for final model filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_save_path = os.path.join(MODEL_DIR, f"final_model_{timestamp}.keras")

        # Save the model after training
        model.save(final_model_save_path)

        # Log params and model
        mlflow.log_param("base_model", "ResNet50")
        mlflow.set_tag("project", "Brain Disease Classification")
        mlflow.keras.log_model(model, "ResNet50_model")



if __name__ == "__main__":
    print("Hello from models.py")
    train_gen, val_gen, test_gen = data_generator(TRAIN_DIR, TEST_DIR)
    build_model(train_gen, val_gen)
