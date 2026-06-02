import os
import hashlib
import numpy as np
from typing import Optional

# Global model variable
model = None

class DeterministicMockModel:
    """Generate deterministic pseudo‑random predictions based on image content.
    The model seeds the RNG with a hash of the image bytes, ensuring the same
    image always yields the same probabilities while still providing variation
    across different images.
    """
    def __init__(self, image_bytes: bytes):
        h = hashlib.sha256(image_bytes).hexdigest()
        seed = int(h[:16], 16) % (2**32)
        self.rng = np.random.default_rng(seed)

    def predict(self, _: Optional[np.ndarray] = None) -> np.ndarray:
        # Generate probabilities for the six expected classes
        probs = self.rng.random(6)
        probs = probs / probs.sum()
        return np.expand_dims(probs, axis=0)

def load_model():
    """Load the TensorFlow model if available, otherwise fall back to mock."""
    global model
    try:
        import tensorflow as tf  # type: ignore
    except ImportError:
        tf = None
    model_path = os.getenv("MODEL_PATH", "ml/dental_classifier.h5")
    if tf is not None and os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        model = None

def predict(image_bytes: bytes) -> np.ndarray:
    """Return prediction probabilities for the given image bytes.
    If a real model is loaded, use it; otherwise use the deterministic mock.
    """
    if model is not None:
        # Preprocess image similarly to inference_api
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        probs = model.predict(arr)[0]
    else:
        mock = DeterministicMockModel(image_bytes)
        probs = mock.predict()[0]
    return probs
