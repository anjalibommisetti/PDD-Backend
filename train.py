"""
Train a CNN dental image classifier on the dataset/ folder.
Outputs: ml/dental_classifier.h5
Classes (alphabetical, as Keras reads them):
  Calculus, Data caries, Gingivitis, Mouth Ulcer, Tooth Discoloration, hypodontia
"""
import os
import sys
import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models

DATASET_DIR = "dataset"
OUTPUT_MODEL = "ml/dental_classifier.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

def main():
    if not os.path.isdir(DATASET_DIR):
        print(f"[ERROR] Dataset directory '{DATASET_DIR}' not found.")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)

    # ---------- Load dataset ----------
    print("[INFO] Loading training data...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    print("[INFO] Loading validation data...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"[INFO] Classes ({num_classes}): {class_names}")

    # ---------- Performance prefetch ----------
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # ---------- Data augmentation ----------
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
    ])

    # ---------- Build CNN model ----------
    model = models.Sequential([
        # Input
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

        # Data augmentation (only active during training)
        data_augmentation,

        # Normalize pixel values to [0, 1]
        layers.Rescaling(1.0 / 255),

        # Conv block 1
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        # Conv block 2
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        # Conv block 3
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        # Conv block 4
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        # Classifier head
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # ---------- Callbacks ----------
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True
    )

    # ---------- Train ----------
    print(f"\n[INFO] Training for up to {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop],
    )

    # ---------- Save ----------
    model.save(OUTPUT_MODEL)
    print(f"\n[INFO] Model saved to {OUTPUT_MODEL}")
    print(f"[INFO] Final val_accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"[INFO] Class order: {class_names}")


if __name__ == "__main__":
    main()
