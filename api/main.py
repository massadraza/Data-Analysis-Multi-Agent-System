import json
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .model_loader import TARGET_NAMES, get_model
from .schemas import HealthResponse, PredictRequest, PredictResponse

app = FastAPI(title="Iris ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "metrics.json")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="healthy")


@app.get("/metrics")
def metrics():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=404, detail="Metrics not found")
    with open(METRICS_PATH) as f:
        return json.load(f)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        model = get_model()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = np.array(request.features).reshape(1, -1)
    prediction_idx = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0].tolist()

    return PredictResponse(
        prediction=TARGET_NAMES[prediction_idx],
        probability=probabilities,
    )
