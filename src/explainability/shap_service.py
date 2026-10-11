"""SHAP explanations. Contributions are associative, not causal."""

from __future__ import annotations

from typing import Any

import numpy as np

CAUSATION_DISCLAIMER = "Model contribution does not imply causation."


def shap_for_tree_model(model, x_background: np.ndarray, x_explain: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
    try:
        import shap
    except Exception as exc:
        return {"available": False, "error": str(exc), "disclaimer": CAUSATION_DISCLAIMER}

    explainer = shap.TreeExplainer(model, data=x_background[: min(400, len(x_background))])
    values = explainer.shap_values(x_explain)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(-mean_abs)
    global_rows = [
        {"feature": feature_names[i] if i < len(feature_names) else f"f{i}", "mean_abs_shap": float(mean_abs[i])}
        for i in order[:20]
    ]
    local = []
    if len(x_explain):
        row = values[0]
        pairs = sorted(enumerate(row), key=lambda kv: -abs(kv[1]))[:12]
        for idx, val in pairs:
            local.append(
                {
                    "feature": feature_names[idx] if idx < len(feature_names) else f"f{idx}",
                    "contribution": float(val),
                    "direction": "increased_risk" if val > 0 else "reduced_risk",
                    "magnitude": float(abs(val)),
                }
            )
    return {
        "available": True,
        "disclaimer": CAUSATION_DISCLAIMER,
        "global": global_rows,
        "local": local,
    }


def permutation_importance_table(model, x, y, feature_names: list[str], n_repeats: int = 3) -> list[dict]:
    from sklearn.inspection import permutation_importance

    result = permutation_importance(model, x, y, n_repeats=n_repeats, scoring="roc_auc", random_state=42)
    order = np.argsort(-result.importances_mean)
    rows = []
    for i in order[:20]:
        rows.append(
            {
                "feature": feature_names[i] if i < len(feature_names) else f"f{i}",
                "importance": float(result.importances_mean[i]),
                "std": float(result.importances_std[i]),
            }
        )
    return rows
