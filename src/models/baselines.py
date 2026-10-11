"""Baseline models. These are not the research contribution."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


def build_baselines(seed: int = 42) -> dict:
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=400,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=8,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=seed,
        ),
        "linear_svm": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=seed, max_iter=2000, dual=False),
            method="sigmoid",
            cv=3,
        ),
    }
    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=seed,
            verbosity=-1,
        )
    return models
