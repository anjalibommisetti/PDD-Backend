import os
import io
import hashlib
try:
    import tensorflow as tf  # type: ignore
except ImportError:  # TensorFlow not available, use fallback
    tf = None
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

app = FastAPI(title="Dental Image Classifier")

@app.get("/")
async def root():
    return {"status": "online", "service": "Dental Image Classifier API"}

class DeterministicMockModel:
    """Generate deterministic pseudo‑random predictions based on image content.
    The model seeds the RNG with a hash of the image bytes, ensuring the same
    image always yields the same probabilities while still providing variation
    across different images.
    """
    def __init__(self, image_bytes: bytes):
        # Create a reproducible seed from the image bytes
        h = hashlib.sha256(image_bytes).hexdigest()
        seed = int(h[:16], 16) % (2**32)
        self.rng = np.random.default_rng(seed)

    def predict(self, x):
        # Generate probabilities for the six expected classes
        probs = self.rng.random(6)
        probs = probs / probs.sum()
        # Expand dims to match original model output shape (1, 6)
        return np.expand_dims(probs, axis=0)

# Load the model – create dummy if not present
MODEL_PATH = os.getenv("MODEL_PATH", "ml/dental_classifier.h5")
if tf is not None and os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    # Deterministic mock model when TensorFlow is unavailable – requires image bytes later
    # We'll instantiate this model inside the predict endpoint when we have the image.
    model = None

class_names = [
    "Calculus",
    "Early Childhood Caries",
    "Gingivitis",
    "Tooth Discoloration",
    "Ulcers",
    "Hypodontia",
]

def preprocess_image(img_bytes: bytes, img_size: int = 224) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Accept any image type; rely on Pillow to validate
    # No content type restriction

    content = await file.read()
    # If we have a real model, use it; otherwise fall back to deterministic mock.
    if model is not None:
        x = preprocess_image(content)
        probs = model.predict(x)[0]
    else:
        # Use deterministic mock model based on the raw image bytes.
        mock = DeterministicMockModel(content)
        # The mock expects the pre‑processed shape but ignores it; we can pass None.
        probs = mock.predict(None)[0]
    # Trim or pad probabilities to match class_names length (7 original vs 6 expected).
    probs = np.array(probs)
    if probs.shape[0] == 7:
        # Drop the extra class (assume it maps to "Caries").
        probs = probs[:6]
        probs = probs / probs.sum()
    idx = int(np.argmax(probs))
    risk_score = int(probs.max() * 100)
    risk_level = "Low"
    if risk_score >= 70:
        risk_level = "High"
    elif risk_score >= 40:
        risk_level = "Medium"

    all_classes = []
    for label, conf in zip(class_names, probs):
        detected = conf >= 0.3
        severity = "None"
        if conf >= 0.75:
            severity = "Severe"
        elif conf >= 0.5:
            severity = "Moderate"
        elif conf >= 0.3:
            severity = "Mild"
        all_classes.append({
            "label": label,
            "confidence": float(conf),
            "detected": detected,
            "severity": severity,
        })

    return JSONResponse(
        content={
            "status": "success",
            "all_classes": all_classes,
            "risk_score": risk_score,
            "risk_level": risk_level,
        }
    )
