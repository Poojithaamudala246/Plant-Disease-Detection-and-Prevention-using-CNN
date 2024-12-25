import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Load Dataset
def load_data(train_dir, test_dir, img_size=(224, 224)):
    """
    Load train and test datasets.
    :param train_dir: Path to training dataset directory.
    :param test_dir: Path to testing dataset directory.
    :param img_size: Tuple for resizing images.
    :return: Training and validation generators.
    """
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=32,
        class_mode="categorical",
        subset="training",
    )

    val_gen = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=32,
        class_mode="categorical",
        subset="validation",
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=32, class_mode="categorical"
    )

    return train_gen, val_gen, test_gen


# 2. Build Model
def build_model(input_shape=(224, 224, 3), num_classes=5):
    """
    Build EfficientNetV2-based CNN model.
    :param input_shape: Shape of input images.
    :param num_classes: Number of output classes.
    :return: Compiled CNN model.
    """
    base_model = EfficientNetV2B0(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base_model.trainable = False  # Freeze the base model

    model = Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        Dropout(0.2),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# 3. Train Model
def train_model(model, train_gen, val_gen, epochs=10):
    """
    Train the CNN model.
    :param model: Compiled CNN model.
    :param train_gen: Training data generator.
    :param val_gen: Validation data generator.
    :param epochs: Number of training epochs.
    :return: Trained model and history.
    """
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
    )
    return model, history


# 4. Evaluate Model
def evaluate_model(model, test_gen):
    """
    Evaluate the trained model on the test dataset.
    :param model: Trained CNN model.
    :param test_gen: Testing data generator.
    :return: Accuracy of the model on the test data.
    """
    loss, accuracy = model.evaluate(test_gen)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy


# 5. Predict Leaf Disease
def predict_image(model, img_path, class_labels, img_size=(224, 224)):
    """
    Predict disease for a single leaf image.
    :param model: Trained CNN model.
    :param img_path: Path to the image.
    :param class_labels: List of class labels.
    :param img_size: Tuple for resizing the image.
    :return: Predicted class label.
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    img = preprocess_input(img)  # Preprocess for EfficientNet
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[class_index]

    print(f"Predicted Disease: {predicted_label}")
    return predicted_label


# 6. Main Function
def main():
    # Define dataset directories
    train_dir = "datasets/train"  # Path to training data
    test_dir = "datasets/test"    # Path to testing data
    img_size = (224, 224)

    # Load data
    train_gen, val_gen, test_gen = load_data(train_dir, test_dir, img_size)

    # Get class labels
    class_labels = list(train_gen.class_indices.keys())

    # Build and train model
    model = build_model(input_shape=(224, 224, 3), num_classes=len(class_labels))
    model, history = train_model(model, train_gen, val_gen, epochs=10)

    # Evaluate model
    evaluate_model(model, test_gen)

    # Save model
    model.save("plant_disease_detector.h5")
    print("Model saved as plant_disease_detector.h5")

    # Predict on a sample image
    sample_image = "datasets/sample_leaf.jpg"  # Replace with path to a sample leaf image
    predict_image(model, sample_image, class_labels, img_size)

if __name__ == "__main__":
    main()
