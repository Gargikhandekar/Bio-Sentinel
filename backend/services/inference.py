"""Load trained artifacts for online inference."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import project_path
from src.data.preprocessing import FEATURE_COLUMNS
from src.ensemble.adaptive_ensemble import AdaptiveContextWeightedEnsemble
from src.models.baselines import build_baselines  # noqa: F401


DISCLAIMER = (
    "RESEARCH PROTOTYPE. This system estimates model-based disease risk from public "
    "research datasets. It is not a medical diagnostic device and must not be used "
    "for clinical decision-making."
)


def _proba(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        return proba[:, 1] if proba.ndim == 2 else np.asarray(proba).ravel()
    return model.predict(x).astype(float)


class InferenceEngine:
    def __init__(self) -> None:
        self.ready = False
        self.error: str | None = None
        self.preprocessor = None
        self.bundle = None
        self.models: dict[str, Any] = {}
        self.competence = None
        self.results: dict[str, Any] | None = None
        self._load()

    def _load(self) -> None:
        art = project_path("models", "trained")
        results = project_path("results", "experiments", "latest.json")
        try:
            if results.exists():
                self.results = json.loads(results.read_text(encoding="utf-8"))
            pre_path = art / "preprocessor.joblib"
            bundle_path = art / "ensemble_bundle.joblib"
            if not pre_path.exists() or not bundle_path.exists():
                self.error = "Awaiting training run"
                return
            self.preprocessor = joblib.load(pre_path)
            self.bundle = joblib.load(bundle_path)
            label = self.bundle["primary_label"]
            for name in self.bundle["pool_names"]:
                path = art / f"{label}_{name}.joblib"
                if path.exists():
                    self.models[name] = joblib.load(path)
            self.competence = joblib.load(art / "competence.joblib")
            self.ready = bool(self.models)
        except Exception as exc:
            self.error = str(exc)
            self.ready = False

    def status(self) -> dict[str, str]:
        operational = "OPERATIONAL" if self.ready else "DEGRADED"
        return {
            "MODEL_SERVICES": operational,
            "DATA_PIPELINE": "OPERATIONAL" if project_path("data", "processed").exists() else "DEGRADED",
            "DRIFT_MONITOR": "OPERATIONAL" if self.results else "AWAITING_EXPERIMENTS",
            "EXPLANATION_ENGINE": operational,
            "CALIBRATION": operational,
            "DATABASE": "OPERATIONAL",
        }

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError(self.error or "Model artifacts are not available.")
        row = {col: features.get(col) for col in FEATURE_COLUMNS}
        frame = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        x = self.preprocessor.transform(frame)[0]
        predictors = {
            name: (lambda xx, model=model: float(_proba(model, np.atleast_2d(xx))[0]))
            for name, model in self.models.items()
        }
        acwe = AdaptiveContextWeightedEnsemble(
            predictors,
            self.competence,
            recency_window=200,
            reliability=self.bundle["reliability"],
        )
        out = acwe.predict_one(x)
        raw = float(out["probability"])
        calibrated = float(self.bundle["calibrator"].transform(raw))
        conformal = self.bundle["conformal"].predict_set(calibrated)
        prediction = int(calibrated >= 0.5)
        explanation = {"disclaimer": "Model contribution does not imply causation.", "local": []}
        if self.results and self.results.get("explainability", {}).get("local"):
            explanation["local"] = self.results["explainability"]["local"]
        return {
            "prediction": prediction,
            "raw_probability": raw,
            "calibrated_probability": calibrated,
            "uncertainty": conformal["uncertainty"],
            "prediction_set": conformal["prediction_set"],
            "model_predictions": out["model_probabilities"],
            "model_weights": out["weights"],
            "local_competence": out["local_competence"],
            "explanation": explanation,
            "disclaimer": DISCLAIMER,
        }


@lru_cache
def get_engine() -> InferenceEngine:
    return InferenceEngine()


def load_results() -> dict[str, Any] | None:
    path = project_path("results", "experiments", "latest.json")
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
