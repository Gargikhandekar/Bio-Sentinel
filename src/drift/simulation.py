"""Controlled synthetic distribution-shift utilities.

These shifts are experimental interventions on static survey data.
They do not represent actual clinical concept drift.
"""

from __future__ import annotations

import numpy as np


def numerical_feature_shift(x: np.ndarray, columns: list[int], scale: float = 0.75, noise: float = 0.35) -> np.ndarray:
    shifted = x.copy()
    shifted[:, columns] = shifted[:, columns] * scale + np.random.default_rng(7).normal(0, noise, size=shifted[:, columns].shape)
    return shifted


def class_prior_shift(y: np.ndarray, extra_positive_rate: float, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(y).copy()
    negatives = np.where(y == 0)[0]
    n_flip = int(len(y) * extra_positive_rate)
    n_flip = min(n_flip, len(negatives))
    if n_flip:
        flip = rng.choice(negatives, size=n_flip, replace=False)
        y[flip] = 1
    return y


def relationship_noise(y: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(y).copy()
    mask = rng.random(len(y)) < rate
    y[mask] = 1 - y[mask]
    return y
