"""Local competence from validation-set nearest neighbors."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class LocalCompetenceEstimator:
    def __init__(self, k: int = 25, metric: str = "euclidean") -> None:
        self.k = k
        self.metric = metric
        self._nn: NearestNeighbors | None = None
        self._y: np.ndarray | None = None
        self._model_correct: dict[str, np.ndarray] = {}

    def fit(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        model_probs: dict[str, np.ndarray],
        threshold: float = 0.5,
    ) -> None:
        self._nn = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
        self._nn.fit(x_val)
        self._y = np.asarray(y_val)
        for name, probs in model_probs.items():
            pred = (np.asarray(probs) >= threshold).astype(int)
            self._model_correct[name] = (pred == self._y).astype(float)

    def competence(self, x: np.ndarray) -> dict[str, float]:
        batch = self.competence_batch(np.atleast_2d(x))
        return {name: float(vals[0]) for name, vals in batch.items()}

    def competence_batch(self, x: np.ndarray) -> dict[str, np.ndarray]:
        if self._nn is None:
            raise RuntimeError("LocalCompetenceEstimator must be fitted.")
        k = min(self.k, len(self._y))
        distances, indices = self._nn.kneighbors(np.atleast_2d(x), n_neighbors=k)
        weights = 1.0 / (distances + 1e-6)
        scores = {}
        for name, correct in self._model_correct.items():
            local = correct[indices]
            scores[name] = np.average(local, axis=1, weights=weights)
        return scores
