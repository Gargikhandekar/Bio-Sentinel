"""Ablation of ACWE components using already-computed prediction streams.

Every variant except ``no_calibration`` is evaluated after its own isotonic
calibrator, so each delta isolates one component instead of mixing it with
the (large) effect of calibration.
"""

from __future__ import annotations

ABLATION_KEYS = [
    "full",
    "no_dynamic_weighting",
    "no_local_competence",
    "no_calibration",
]


def ablation_table(bundles: dict[str, dict], metric: str = "roc_auc") -> list[dict]:
    if "full" not in bundles:
        return []
    full = bundles["full"]
    rows = []
    for key in ABLATION_KEYS:
        if key not in bundles:
            continue
        b = bundles[key]
        value = b.get(metric)
        delta = None if full.get(metric) is None or value is None else value - full[metric]
        rows.append(
            {
                "variant": key,
                metric: value,
                "delta_vs_full": delta,
                "brier": b.get("brier"),
                "ece": b.get("ece"),
                "delta_brier_vs_full": b.get("brier") - full.get("brier"),
                "delta_ece_vs_full": b.get("ece") - full.get("ece"),
            }
        )
    return rows
