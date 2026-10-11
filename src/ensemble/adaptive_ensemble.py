"""Adaptive Context-Weighted Ensemble (ACWE).

This is an integration of existing dynamic ensemble selection, local competence,
and online performance tracking. It is not a newly invented learning algorithm.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from src.ensemble.competence import LocalCompetenceEstimator
from src.ensemble.weighting import combine_probabilities, context_weights, equal_weights, global_performance_weights


class AdaptiveContextWeightedEnsemble:
    def __init__(
        self,
        model_predictors: dict[str, callable],
        competence: LocalCompetenceEstimator,
        recency_window: int = 200,
        reliability: dict[str, float] | None = None,
    ) -> None:
        self.model_predictors = model_predictors
        self.competence = competence
        self.recency_window = recency_window
        self.reliability = reliability or {name: 1.0 for name in model_predictors}
        self._recent: dict[str, deque] = {name: deque(maxlen=recency_window) for name in model_predictors}
        for name in model_predictors:
            self._recent[name].append(0.7)

    def _recent_performance(self) -> dict[str, float]:
        return {name: float(np.mean(buf)) if buf else 0.5 for name, buf in self._recent.items()}

    def predict_one(self, x: np.ndarray) -> dict:
        probs = {name: float(fn(x)) for name, fn in self.model_predictors.items()}
        local = self.competence.competence(x)
        recent = self._recent_performance()
        weights = context_weights(local, recent, self.reliability)
        ensemble_p = combine_probabilities(weights, probs)
        return {
            "probability": ensemble_p,
            "model_probabilities": probs,
            "weights": weights,
            "local_competence": local,
            "recent_performance": recent,
            "soft_vote": combine_probabilities(equal_weights(list(probs)), probs),
            "global_weighted": combine_probabilities(global_performance_weights(recent), probs),
        }

    def update(self, x: np.ndarray, y: int, threshold: float = 0.5) -> None:
        for name, fn in self.model_predictors.items():
            pred = int(fn(x) >= threshold)
            self._recent[name].append(float(pred == y))
