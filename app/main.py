from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routers import predict, dataset, health
from app.services import predictor



@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load_model()
    yield


app = FastAPI(
    title       = "Dental Health AI API",
    description = "REST API for dental condition detection using YOLOv8",
    version     = "1.0.0",
    lifespan    = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"]
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(dataset.router)
