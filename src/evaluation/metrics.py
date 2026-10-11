"""Classification metrics used across static and adaptive experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.clip(np.asarray(y_prob).astype(float), 1e-7, 1 - 1e-7)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= left) & (y_prob < right if right < 1 else y_prob <= right)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def classification_bundle(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob).astype(float), 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "ece": expected_calibration_error(y_true, y_prob),
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    metrics["roc_curve"] = {"fpr": fpr.tolist()[:: max(1, len(fpr) // 80)], "tpr": tpr.tolist()[:: max(1, len(tpr) // 80)]}
    metrics["pr_curve"] = {
        "precision": prec.tolist()[:: max(1, len(prec) // 80)],
        "recall": rec.tolist()[:: max(1, len(rec) // 80)],
    }
    return metrics


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict]:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= left) & (y_prob < right if right < 1 else y_prob <= right)
        if not np.any(mask):
            rows.append({"bin": f"{left:.1f}-{right:.1f}", "mean_predicted": None, "mean_observed": None, "count": 0})
            continue
        rows.append(
            {
                "bin": f"{left:.1f}-{right:.1f}",
                "mean_predicted": float(y_prob[mask].mean()),
                "mean_observed": float(y_true[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return rows
