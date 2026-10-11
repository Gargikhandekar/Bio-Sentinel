"""Ablation of ACWE components using already-computed prediction streams."""

from __future__ import annotations

from src.evaluation.metrics import classification_bundle


ABLATION_KEYS = [
    "full",
    "no_dynamic_weighting",
    "no_local_competence",
    "no_online_updating",
    "no_drift_adaptation",
    "no_calibration",
]


def ablation_table(bundles: dict[str, dict], metric: str = "roc_auc") -> list[dict]:
    if "full" not in bundles:
        return []
    full = bundles["full"].get(metric)
    rows = []
    for key in ABLATION_KEYS:
        if key not in bundles:
            continue
        value = bundles[key].get(metric)
        delta = None if full is None or value is None else value - full
        rows.append({"variant": key, metric: value, "delta_vs_full": delta, "brier": bundles[key].get("brier")})
    return rows
