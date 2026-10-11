"""Leakage audit for BRFSS multi-label chronic-disease prediction."""

from __future__ import annotations

from typing import Any

FORBIDDEN_PREDICTORS = {
    "DIABETE4",
    "DIABAGE3",
    "INSULIN1",
    "BPHIGH6",
    "BPMEDS",
    "_RFHYPE6",
    "CVDINFR4",
    "CVDCRHD4",
    "_MICHD",
    "CVDSTRK3",
    "diabetes",
    "hypertension",
    "cvd",
}

ALLOWED_METABOLIC_BUT_AUDITED = {
    "high_cholesterol": (
        "Self-reported high cholesterol is a metabolic covariate, not one of the three "
        "research labels. It is retained as a predictor and must not be treated as a proxy "
        "for CVD."
    )
}


def audit_feature_list(feature_names: list[str]) -> dict[str, Any]:
    leaked = sorted(set(feature_names) & FORBIDDEN_PREDICTORS)
    notes = [
        "Targets are self-reported diagnosed conditions, not incident disease.",
        "BPMEDS and DIABAGE3 are skip-pattern variables that encode the labels and are excluded.",
        "Stroke (CVDSTRK3) is excluded because it is a cardiovascular event adjacent to the CVD label.",
        "A controlled sequential stream is constructed from static public data to investigate robustness under simulated distribution shift. This is not longitudinal clinical follow-up.",
    ]
    for name, note in ALLOWED_METABOLIC_BUT_AUDITED.items():
        if name in feature_names:
            notes.append(note)
    return {
        "leaked_features": leaked,
        "passed": len(leaked) == 0,
        "notes": notes,
    }
