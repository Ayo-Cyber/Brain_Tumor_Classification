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
BASE_DIR = "/Users/admin/Documents/GitHub/Brain_Tumor_Classification/data/"
TRAIN_DIR = os.path.join(BASE_DIR, "Training")
TEST_DIR = os.path.join(BASE_DIR, "Testing")

# Function to evaluate the model
def evaluate_model(model_path, test_dir=TEST_DIR):
    mlflow.tensorflow.autolog()

    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Load the test data generator
    train_gen, val_gen, test_gen = data_generator(TRAIN_DIR, TEST_DIR)
    test_gen = test_gen

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)

    # Log evaluation metrics to MLflow
    with mlflow.start_run(run_name='Model Evaluation'):
        mlflow.log_param("model_path", model_path)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)

        class_names = test_gen.class_names

        true_labels = []
        test_images = []
        for images, labels in test_gen:
            true_labels.extend(np.argmax(labels.numpy(), axis=1))
            test_images.extend(images.numpy())

        true_labels = np.array(true_labels)
        test_images = np.array(test_images)


        pred_probabilities = model.predict(test_images)
        pred_labels = np.argmax(pred_probabilities, axis=1)

        class_report = classification_report(true_labels, pred_labels, target_names=class_names)

        # # Get predictions for classification report and confusion matrix
        # y_true = test_gen.class_names
        # y_pred = model.predict(test_gen, verbose=1)
        # y_pred_classes = np.argmax(y_pred, axis=1)

        # # Classification report and confusion matrix
        # class_report = classification_report(y_true, y_pred_classes, target_names=test_gen.class_names)
        conf_matrix = confusion_matrix(true_labels, pred_labels)

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
    model_path = "model_artifacts/final_model_20250418_134412.keras"  
    evaluate_model(model_path)
