import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape, num_classes):
    """
    Create a model using transfer learning with a pretrained MobileNetV2 base model.
    
    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.

    Returns:
        tf.keras.Model: The constructed Keras model.
    """
    # Load the pretrained MobileNetV2 without the top classification layer
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,  # Exclude original top layers
                                                   weights='imagenet')  # Use ImageNet weights
    base_model.trainable = False  # Freeze the base model's weights

    # Add custom classification layers
    model = models.Sequential([
        base_model,  # Pretrained base model
        layers.GlobalAveragePooling2D(),  # Reduce dimensionality
        layers.Dense(128, activation='relu'),  # Custom dense layer
        layers.Dropout(0.3),  # Add dropout to prevent overfitting
        layers.Dense(num_classes, activation='softmax')  # Output layer with softmax for classification
    ])

    return model

def compile_model(model):
    """
    Compile the model with the appropriate optimizer, loss, and metrics.
    
    Args:
        model (tf.keras.Model): The Keras model to compile.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),  # Adam optimizer
        loss='categorical_crossentropy',       # For multiclass classification
        metrics=['accuracy']                   # Track accuracy during training
    )

def train_model(model, train_dataset, val_dataset, epochs=10):
    """
    Train the model using training and validation datasets.
    
    Args:
        model (tf.keras.Model): The Keras model to train.
        train_dataset (tf.data.Dataset): The training dataset.
        val_dataset (tf.data.Dataset): The validation dataset.
        epochs (int): Number of training epochs.

    Returns:
        tf.keras.callbacks.History: Training history object.
    """
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,  # Set the number of epochs
    )
    
    return history

def evaluate_model(model, test_dataset):
    """
    Evaluate the model on the test dataset.
    
    Args:
        model (tf.keras.Model): The Keras model to evaluate.
        test_dataset (tf.data.Dataset): The test dataset.

    Returns:
        tuple: Test loss and accuracy.
    """
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_accuracy:.2f}")
    return test_loss, test_accuracy
