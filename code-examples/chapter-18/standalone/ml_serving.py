"""
Chapter 18: Production MLOps - ML Model Serving Platform

Demonstrates:
- Model serving
- Monitoring and metrics
- A/B testing
- Model versioning

Run: uvicorn ml_serving:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import time
from datetime import datetime

app = FastAPI(title="ML Model Serving - Chapter 18")

# Metrics storage
metrics = {"predictions": 0, "avg_latency": 0.0}
model_versions = {"v1": 0, "v2": 0}

class PredictionRequest(BaseModel):
    features: List[float]
    model_version: str = "v1"

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Serve model predictions.
    
    CONCEPT: Model Serving
    - Load model
    - Preprocess input
    - Make prediction
    - Log metrics
    """
    start = time.time()
    
    # Fake prediction (replace with real model)
    prediction = sum(request.features) / len(request.features)
    
    latency = time.time() - start
    
    # Update metrics
    metrics["predictions"] += 1
    model_versions[request.model_version] = model_versions.get(request.model_version, 0) + 1
    metrics["avg_latency"] = (metrics["avg_latency"] * (metrics["predictions"] - 1) + latency) / metrics["predictions"]
    
    return {
        "prediction": prediction,
        "model_version": request.model_version,
        "latency_ms": latency * 1000
    }

@app.get("/metrics")
async def get_metrics():
    """
    Get serving metrics.
    
    CONCEPT: Monitoring
    - Track prediction count
    - Monitor latency
    - Version distribution
    """
    return {
        "total_predictions": metrics["predictions"],
        "avg_latency_ms": metrics["avg_latency"] * 1000,
        "model_versions": model_versions,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ml_serving:app", host="0.0.0.0", port=8000, reload=True)

