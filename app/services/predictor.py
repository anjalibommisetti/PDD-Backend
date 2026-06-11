# app/services/predictor.py
import os
import io
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

MODEL_PATH = "ml/dental_classifier.h5"
IMG_SIZE = (224, 224)

# Global model cache
_model = None

def load_model():
    """Load the trained CNN model. Raises if model file is missing."""
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    _model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[INFO] Loaded CNN model from {MODEL_PATH}")
    print(f"[INFO] Model output shape: {_model.output_shape}")
    return _model

def predict(image_bytes: bytes) -> np.ndarray:
    """Predict the class probabilities of a dental image using the CNN model.
    
    Returns:
        numpy array of shape (6,) with softmax probabilities for each class.
        Class order (alphabetical from dataset folders):
          [Calculus, Data caries, Gingivitis, Mouth Ulcer, Tooth Discoloration, hypodontia]
    """
    model = load_model()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Apply EXIF transpose to ensure correct orientation across different devices
    img = ImageOps.exif_transpose(img)
    
    # Use bilinear interpolation to match training data loading
    resample_method = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    img = img.resize(IMG_SIZE, resample_method)
    
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)   # shape: (1, 224, 224, 3)
    # Note: do NOT divide by 255 here — the model has a Rescaling layer built in
    probs = model.predict(arr)[0]       # shape: (num_classes,)
    return probs
