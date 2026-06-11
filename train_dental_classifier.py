import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models
image_dataset_from_directory = tf.keras.utils.image_dataset_from_directory


def build_model(input_shape=(224, 224, 3), num_classes=2):
    """Create a simple CNN model.

    Args:
        input_shape: Shape of input images.
        num_classes: Number of output classes.
    Returns:
        A compiled tf.keras Model.
    """
    model = models.Sequential([
        layers.Rescaling(1.0 / 255, input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    parser = argparse.ArgumentParser(description='Train a dental image classifier.')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to a directory containing subfolders for each class (e.g., "healthy" and "diseased").')
    parser.add_argument('--output-model', type=str, default='dental_classifier.h5',
                        help='File path to save the trained model.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training.')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Resize images to IMG_SIZE x IMG_SIZE.')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"[ERROR] Data directory {data_dir} does not exist.")
        sys.exit(1)

    # Load dataset – expects folder structure: data_dir/class_a/*, data_dir/class_b/*
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size)
    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size)

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"[INFO] Detected classes: {class_names}")

    model = build_model(input_shape=(args.img_size, args.img_size, 3), num_classes=num_classes)
    model.summary()

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=args.epochs)

    model.save(args.output_model)
    print(f"[INFO] Model saved to {args.output_model}")

if __name__ == '__main__':
    main()
