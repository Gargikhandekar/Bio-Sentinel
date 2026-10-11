"""Split conformal prediction for binary risk scores.

The prediction set is over {0, 1} at level 1-alpha using a validation
calibration split. This is not a Bayesian credible interval.
"""

from __future__ import annotations

import numpy as np


class SplitConformalClassifier:
    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.quantile: float | None = None

    def fit(self, y_cal: np.ndarray, p_cal: np.ndarray) -> "SplitConformalClassifier":
        y_cal = np.asarray(y_cal).astype(int)
        p_cal = np.clip(np.asarray(p_cal).astype(float), 1e-6, 1 - 1e-6)
        scores = np.where(y_cal == 1, 1.0 - p_cal, p_cal)
        n = len(scores)
        q_level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.quantile = float(np.quantile(scores, q_level, method="higher"))
        return self

    def predict_set(self, p: float) -> dict:
        if self.quantile is None:
            raise RuntimeError("Conformal classifier is not fitted.")
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        include_pos = (1.0 - p) <= self.quantile
        include_neg = p <= self.quantile
        pred_set = []
        if include_neg:
            pred_set.append(0)
        if include_pos:
            pred_set.append(1)
        if not pred_set:
            pred_set = [1 if p >= 0.5 else 0]
        width = len(pred_set)
        if width == 2:
            band = "high"
        elif abs(p - 0.5) < 0.1:
            band = "moderate"
        else:
            band = "low"
        return {
            "prediction_set": pred_set,
            "uncertainty": band,
            "conformal_quantile": self.quantile,
            "alpha": self.alpha,
            "set_size": width,
        }
