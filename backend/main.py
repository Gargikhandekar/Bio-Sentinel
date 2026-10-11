"""BIO-SENTINEL FastAPI application."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import datasets, drift, experiments, explainability, health, models, prediction, research

app = FastAPI(
    title="BIO-SENTINEL",
    description="Adaptive Computational Intelligence for Chronic Disease Risk Prognostics (research prototype).",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(datasets.router)
app.include_router(models.router)
app.include_router(prediction.router)
app.include_router(explainability.router)
app.include_router(experiments.router)
app.include_router(drift.router)
app.include_router(research.router)


@app.get("/")
def root():
    return {
        "name": "BIO-SENTINEL",
        "subtitle": "Adaptive Computational Intelligence for Chronic Disease Risk Prognostics",
        "disclaimer": "Research prototype. Not a medical diagnostic device.",
    }
