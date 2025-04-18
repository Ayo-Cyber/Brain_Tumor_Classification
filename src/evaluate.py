import tensorflow as tf
import os
import mlflow
import mlflow.tensorflow
from utils import data_generator  # Assuming you have a function to generate test data
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Constants
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
SEED = 42
MODEL_DIR = "model_artifacts"  # Directory to store models
TEST_DIR = "path_to_test_data"  # Replace with your test data directory

# Function to evaluate the model
def evaluate_model(model_path, test_dir=TEST_DIR):
    mlflow.tensorflow.autolog()

    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Load the test data generator
    test_gen = data_generator(test_dir, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, shuffle=False)

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)

    # Log evaluation metrics to MLflow
    with mlflow.start_run(run_name='Model Evaluation'):
        mlflow.log_param("model_path", model_path)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)

        # Get predictions for classification report and confusion matrix
        y_true = test_gen.classes
        y_pred = model.predict(test_gen, verbose=1)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Classification report and confusion matrix
        class_report = classification_report(y_true, y_pred_classes, target_names=test_gen.class_indices.keys())
        conf_matrix = confusion_matrix(y_true, y_pred_classes)

        # Log classification report and confusion matrix as artifacts in MLflow
        with open("classification_report.txt", "w") as f:
            f.write(class_report)
        with open("confusion_matrix.txt", "w") as f:
            f.write(str(conf_matrix))

        # Log the confusion matrix and classification report as MLflow artifacts
        mlflow.log_artifact("classification_report.txt")
        mlflow.log_artifact("confusion_matrix.txt")

        print("Test Loss: ", test_loss)
        print("Test Accuracy: ", test_acc)
        print("Classification Report: \n", class_report)
        print("Confusion Matrix: \n", conf_matrix)

if __name__ == "__main__":
    # Example usage, replace with actual model path
    model_path = "model_artifacts/final_model_20250418_124356.keras"  # Change this to your model path
    evaluate_model(model_path)
