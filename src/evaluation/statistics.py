"""Bootstrap intervals and paired tests. No p-hacking: pre-specified comparisons only."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def bootstrap_ci(values: np.ndarray, n_boot: int = 400, alpha: float = 0.05, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    stats_ = []
    n = len(values)
    for _ in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        stats_.append(sample.mean())
    lo, hi = np.quantile(stats_, [alpha / 2, 1 - alpha / 2])
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if n > 1 else 0.0,
        "ci95": [float(lo), float(hi)],
    }


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred_a = np.asarray(pred_a).astype(int)
    pred_b = np.asarray(pred_b).astype(int)
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    b01 = int((~a_correct & b_correct).sum())
    b10 = int((a_correct & ~b_correct).sum())
    table = np.array([[int((a_correct & b_correct).sum()), b01], [b10, int((~a_correct & ~b_correct).sum())]])
    if b01 + b10 == 0:
        pvalue = 1.0
        statistic = 0.0
    else:
        statistic = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
        pvalue = float(stats.chi2.sf(statistic, 1))
    return {"statistic": float(statistic), "p_value": pvalue, "b01": b01, "b10": b10, "table": table.tolist()}


def delong_roc_auc(y_true: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> dict:
    """Approximate paired AUC comparison via bootstrap rather than claiming a full DeLong implementation if unstable."""
    y_true = np.asarray(y_true).astype(int)
    rng = np.random.default_rng(0)
    diffs = []
    n = len(y_true)
    for _ in range(300):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y_true[idx], p_a[idx]) - roc_auc_score(y_true[idx], p_b[idx]))
    diffs = np.asarray(diffs)
    mean = float(diffs.mean()) if len(diffs) else 0.0
    se = float(diffs.std(ddof=1)) if len(diffs) > 1 else 1.0
    z = mean / se if se else 0.0
    p = float(2 * stats.norm.sf(abs(z)))
    return {"auc_diff_mean": mean, "z": z, "p_value": p, "method": "bootstrap_paired_auc"}
