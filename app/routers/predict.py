from fastapi import APIRouter, File, UploadFile
from app.services.predictor import predict


router = APIRouter()

@router.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    probs = predict(content)
    # Return raw probabilities; client can interpret.
    return {"probabilities": probs.tolist()}
