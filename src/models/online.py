"""Genuine incremental learners used in prequential evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OnlineModel:
    name: str
    model: object

    def predict_proba_one(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def learn_one(self, x: np.ndarray, y: int) -> None:
        raise NotImplementedError


class RiverWrapper(OnlineModel):
    def predict_proba_one(self, x: np.ndarray) -> float:
        features = {f"f{i}": float(v) for i, v in enumerate(x)}
        if hasattr(self.model, "predict_proba_one"):
            probs = self.model.predict_proba_one(features)
            if isinstance(probs, dict):
                return float(probs.get(True, probs.get(1, 0.5)))
        pred = self.model.predict_one(features)
        if pred is None:
            return 0.5
        return 0.72 if pred else 0.28

    def learn_one(self, x: np.ndarray, y: int) -> None:
        features = {f"f{i}": float(v) for i, v in enumerate(x)}
        self.model.learn_one(features, bool(y))


def build_online_models(seed: int = 42) -> list[RiverWrapper]:
    from river import linear_model, tree, optim, forest

    return [
        RiverWrapper("online_logreg", linear_model.LogisticRegression(optimizer=optim.SGD(0.05))),
        RiverWrapper("passive_aggressive", linear_model.PAClassifier(C=0.01, mode=1)),
        RiverWrapper("hoeffding_tree", tree.HoeffdingTreeClassifier(grace_period=50, delta=1e-5)),
        RiverWrapper(
            "adaptive_random_forest",
            forest.ARFClassifier(n_models=5, seed=seed, max_features="sqrt"),
        ),
    ]
