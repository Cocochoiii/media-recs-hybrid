from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import os, json, logging, time
from prometheus_fastapi_instrumentator import Instrumentator

from .recommend import HybridRecommenderService
from .telemetry import init_tracer_provider
from .logging_conf import configure_logging

app = FastAPI(title="Hybrid Recommender API", version="1.0.0")
Instrumentator().instrument(app).expose(app)  # /metrics

configure_logging()
init_tracer_provider()

SERVICE = HybridRecommenderService()

class RecRequest(BaseModel):
    user_id: str
    k: int = 10
    alpha: float = 0.6

@app.get("/health")
def health():
    return {"status": "ok", "ts": time.time()}

@app.get("/recommendations")
def get_recommendations(user_id: str, k: int = 10, alpha: float = 0.6):
    try:
        recs = SERVICE.recommend(user_id=user_id, k=k, alpha=alpha)
        return {"user_id": user_id, "k": k, "alpha": alpha, "items": recs}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
