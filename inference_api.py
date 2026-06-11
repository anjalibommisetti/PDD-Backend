import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from app.services.predictor import predict as predictor_predict

app = FastAPI(title="Dental Image Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class_names = [
    "Calculus",
    "Early Childhood Caries",
    "Gingivitis",
    "Tooth Discoloration",
    "Ulcers",
    "Hypodontia",
]

@app.get("/")
async def root():
    return {"status": "online", "service": "Dental Image Classifier API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    
    # Run dataset similarity prediction
    probs = predictor_predict(content)
    
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
