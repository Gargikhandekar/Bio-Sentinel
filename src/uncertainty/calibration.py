"""Probability calibration fitted only on validation predictions."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    def __init__(self, method: str = "isotonic", clip: float = 1e-3) -> None:
        self.method = method
        # Isotonic regression returns exactly 0 or 1 in its end bins; a few thousand
        # calibration rows cannot justify certainty, so outputs are clipped.
        self.clip = clip
        self._iso: IsotonicRegression | None = None
        self._platt: LogisticRegression | None = None

    def fit(self, y_val: np.ndarray, p_val: np.ndarray) -> "ProbabilityCalibrator":
        p_val = np.clip(np.asarray(p_val).astype(float), 1e-6, 1 - 1e-6)
        y_val = np.asarray(y_val).astype(int)
        if self.method == "platt":
            self._platt = LogisticRegression(max_iter=200)
            self._platt.fit(p_val.reshape(-1, 1), y_val)
        else:
            self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._iso.fit(p_val, y_val)
            self.method = "isotonic"
        return self

    def transform(self, p: np.ndarray | float) -> np.ndarray | float:
        scalar = np.isscalar(p)
        arr = np.clip(np.atleast_1d(p).astype(float), 1e-6, 1 - 1e-6)
        if self._iso is not None:
            out = self._iso.predict(arr)
        elif self._platt is not None:
            out = self._platt.predict_proba(arr.reshape(-1, 1))[:, 1]
        else:
            out = arr
        out = np.clip(out, self.clip, 1 - self.clip)
        return float(out[0]) if scalar else out
