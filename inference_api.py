import os
import io
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

app = FastAPI(title="Dental Image Classifier")

# Load the model – create dummy if not present
MODEL_PATH = os.getenv("MODEL_PATH", "ml/dental_classifier.h5")
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    dummy_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(7, activation="softmax"),
    ])
    dummy_model.save(MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

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
