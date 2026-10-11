"""Load trained artifacts for online inference."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, models_dir, project_path, results_dir
from src.data.loader import LABELS
from src.data.preprocessing import FEATURE_COLUMNS
from src.ensemble.adaptive_ensemble import AdaptiveContextWeightedEnsemble
from src.explainability.shap_service import CAUSATION_DISCLAIMER, local_contributions, shap_matrix

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
        self.cfg = load_config()
        self.error: str | None = None
        self.preprocessor = None
        self.feature_names: list[str] = []
        # label -> {"bundle", "models", "competence"}
        self.targets: dict[str, dict[str, Any]] = {}
        self.results: dict[str, Any] | None = None
        self._load()

    @property
    def ready(self) -> bool:
        return bool(self.targets)

    def _load(self) -> None:
        art = models_dir(self.cfg)
        self.results = load_results()
        try:
            pre_path = art / "preprocessor.joblib"
            if not pre_path.exists():
                self.error = "Awaiting training run"
                return
            self.preprocessor = joblib.load(pre_path)
            self.feature_names = list(self.preprocessor.get_feature_names_out())
            for label in LABELS:
                bundle_path = art / f"{label}_ensemble_bundle.joblib"
                if not bundle_path.exists():
                    continue
                bundle = joblib.load(bundle_path)
                models = {
                    name: joblib.load(art / f"{label}_{name}.joblib")
                    for name in bundle["pool_names"]
                    if (art / f"{label}_{name}.joblib").exists()
                }
                if models:
                    self.targets[label] = {
                        "bundle": bundle,
                        "models": models,
                        "competence": joblib.load(art / f"{label}_competence.joblib"),
                    }
            if not self.targets:
                self.error = "Awaiting training run"
        except Exception as exc:
            self.error = str(exc)
            self.targets = {}

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

    def predict(self, features: dict[str, Any], label: str = "diabetes") -> dict[str, Any]:
        if label not in self.targets:
            raise RuntimeError(self.error or f"Model artifacts for '{label}' are not available.")
        target = self.targets[label]
        bundle = target["bundle"]
        row = {col: features.get(col) for col in FEATURE_COLUMNS}
        frame = pd.DataFrame([row], columns=FEATURE_COLUMNS, dtype=float)
        x = self.preprocessor.transform(frame)[0]
        predictors = {
            name: (lambda xx, model=model: float(_proba(model, np.atleast_2d(xx))[0]))
            for name, model in target["models"].items()
        }
        acwe = AdaptiveContextWeightedEnsemble(
            predictors,
            target["competence"],
            recency_window=bundle.get("recency_window", 200),
            reliability=bundle["reliability"],
        )
        out = acwe.predict_one(x)
        raw = float(out["probability"])
        calibrated = float(bundle["calibrator"].transform(raw))
        conformal = bundle["conformal"].predict_set(calibrated)
        return {
            "label": label,
            "prediction": int(calibrated >= 0.5),
            "raw_probability": raw,
            "calibrated_probability": calibrated,
            "uncertainty": conformal["uncertainty"],
            "prediction_set": conformal["prediction_set"],
            "model_predictions": out["model_probabilities"],
            "model_weights": out["weights"],
            "local_competence": out["local_competence"],
            "explanation": self._explain(target, bundle, x),
            "disclaimer": DISCLAIMER,
        }

    def _explain(self, target: dict[str, Any], bundle: dict[str, Any], x: np.ndarray) -> dict[str, Any]:
        """TreeSHAP for this input on the label's LightGBM member (explains that member, not the ensemble)."""
        explanation: dict[str, Any] = {"disclaimer": CAUSATION_DISCLAIMER, "local": []}
        model = target["models"].get(bundle.get("shap_model") or "")
        if model is None:
            return explanation
        values = shap_matrix(model, np.atleast_2d(x))
        if values is not None:
            explanation["model"] = bundle["shap_model"]
            explanation["local"] = local_contributions(values[0], self.feature_names)
        return explanation


@lru_cache
def get_engine() -> InferenceEngine:
    return InferenceEngine()


def load_results() -> dict[str, Any] | None:
    path = results_dir(load_config()) / "experiments" / "latest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
