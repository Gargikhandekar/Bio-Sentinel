"""Disease-dependency adjustment using validation co-occurrence.

Uses predicted labels of the *other* diseases as context. True test labels
are never used. This is a co-occurrence prior, not a causal model.
"""

from __future__ import annotations

import numpy as np


class CooccurrencePrior:
    def __init__(self, labels: tuple[str, ...] = ("diabetes", "hypertension", "cvd")) -> None:
        self.labels = labels
        self.tables: dict[str, dict[tuple[int, int], float]] = {}
        self.marginals: dict[str, float] = {}

    def fit(self, y: np.ndarray) -> "CooccurrencePrior":
        y = np.asarray(y).astype(int)
        for i, name in enumerate(self.labels):
            others = [j for j in range(len(self.labels)) if j != i]
            self.marginals[name] = float(y[:, i].mean())
            table: dict[tuple[int, int], list[int]] = {}
            for row in y:
                key = (int(row[others[0]]), int(row[others[1]]))
                table.setdefault(key, []).append(int(row[i]))
            self.tables[name] = {k: float(np.mean(v)) for k, v in table.items()}
        return self

    def blend(self, name: str, p_model: np.ndarray, other_a: np.ndarray, other_b: np.ndarray, lam: float = 0.2) -> np.ndarray:
        p_model = np.asarray(p_model, dtype=float)
        keys = list(zip(other_a.astype(int), other_b.astype(int)))
        prior = np.array([self.tables[name].get(key, self.marginals[name]) for key in keys], dtype=float)
        return (1.0 - lam) * p_model + lam * prior
