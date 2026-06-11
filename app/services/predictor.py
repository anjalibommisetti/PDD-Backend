# app/services/predictor.py
import os
import io
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "ml/dental_model.h5"
IMG_SIZE = (224, 224)

# Global model cache
_model = None

def load_model():
    """Load the trained CNN model dynamically if available."""
    global _model
    if _model is not None:
        return _model
    if os.path.exists(MODEL_PATH):
        try:
            _model = tf.keras.models.load_model(MODEL_PATH)
            print(f"[INFO] Loaded CNN model from {MODEL_PATH}")
            return _model
        except Exception as e:
            print(f"[ERROR] Failed to load CNN model: {e}")
    else:
        print(f"[WARNING] CNN model file not found at {MODEL_PATH}")
    return None

class DeterministicMockModel:
    """Fallback deterministic mock model to return stable, randomized probabilities."""
    def __init__(self, image_bytes: bytes):
        import hashlib
        h = hashlib.sha256(image_bytes).hexdigest()
        seed = int(h[:16], 16) % (2**32)
        self.rng = np.random.default_rng(seed)

    def predict(self) -> np.ndarray:
        probs = self.rng.random(6)
        probs = probs / probs.sum()
        return probs

def predict(image_bytes: bytes) -> np.ndarray:
    """Predict the class probabilities of a dental image using the CNN model.
    Falls back to a deterministic mock on errors or if the model isn't trained yet.
    """
    model = load_model()
    if model is not None:
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = img.resize(IMG_SIZE)
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)  # shape: (1, 224, 224, 3)
            probs = model.predict(arr)[0]      # shape: (6,)
            
            # Reorder CNN probabilities (alphabetical sorted class names from dataset)
            # to match the order of class names in inference_api.py:
            # CNN classes: [Calculus, Data caries, Gingivitis, Mouth Ulcer, Tooth Discoloration, hypodontia]
            # API classes: [Calculus, Early Childhood Caries, Gingivitis, Tooth Discoloration, Ulcers, Hypodontia]
            ordered_probs = np.array([
                probs[0],  # Calculus -> Calculus
                probs[1],  # Data caries -> Early Childhood Caries
                probs[2],  # Gingivitis -> Gingivitis
                probs[4],  # Tooth Discoloration -> Tooth Discoloration
                probs[3],  # Mouth Ulcer -> Ulcers
                probs[5],  # hypodontia -> Hypodontia
            ], dtype=np.float32)
            return ordered_probs
        except Exception as e:
            print(f"[ERROR] Prediction failed using CNN model: {e}. Falling back to mock.")
            
    # Mock fallback
    mock = DeterministicMockModel(image_bytes)
    return mock.predict()
