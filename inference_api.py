import os
import io
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

class MockModel:
    def predict(self, x):
        probs = np.random.rand(7)
        probs = probs / probs.sum()
        return np.expand_dims(probs, axis=0)

# Load the model – create dummy if not present
MODEL_PATH = os.getenv("MODEL_PATH", "ml/dental_classifier.h5")
if tf is not None and os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = MockModel()

class_names = [
    "Calculus",
    "Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset",
    "Data caries",
    "Gingivitis",
    "Mouth Ulcer",
    "Tooth Discoloration",
    "hypodontia",
]

def preprocess_image(img_bytes: bytes, img_size: int = 224) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Unsupported image type")
    content = await file.read()
    x = preprocess_image(content)
    probs = model.predict(x)[0]
    idx = int(np.argmax(probs))
    return JSONResponse(
        content={
            "predicted_class": class_names[idx],
            "probability": float(probs[idx]),
            "all_probabilities": {c: float(p) for c, p in zip(class_names, probs)},
        }
    )
