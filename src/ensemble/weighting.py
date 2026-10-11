"""Configurable dynamic weighting. Not claimed as a new algorithm."""

from __future__ import annotations

import numpy as np


def context_weights(
    local_competence: dict[str, float],
    recent_performance: dict[str, float],
    reliability: dict[str, float],
    eps: float = 1e-6,
) -> dict[str, float]:
    names = list(local_competence)
    scores = {}
    for name in names:
        scores[name] = max(local_competence.get(name, eps), eps)
        scores[name] *= max(recent_performance.get(name, eps), eps)
        scores[name] *= max(reliability.get(name, eps), eps)
    total = sum(scores.values()) or eps
    return {name: value / total for name, value in scores.items()}


def combine_probabilities(weights: dict[str, float], probs: dict[str, float]) -> float:
    return float(sum(weights[name] * probs[name] for name in weights))


def equal_weights(names: list[str]) -> dict[str, float]:
    if not names:
        return {}
    w = 1.0 / len(names)
    return {name: w for name in names}


def global_performance_weights(recent_performance: dict[str, float], eps: float = 1e-6) -> dict[str, float]:
    scores = {name: max(val, eps) for name, val in recent_performance.items()}
    total = sum(scores.values())
    return {name: val / total for name, val in scores.items()}
