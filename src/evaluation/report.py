"""Write the per-target results summary and the README figures from a run payload."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.data.loader import LABELS

SUMMARY_MODELS = [
    ("lightgbm", "LightGBM"),
    ("lightgbm_calibrated", "LightGBM + isotonic"),
    ("soft_vote", "Soft vote"),
    ("acwe_raw", "ACWE raw"),
    ("acwe_calibrated", "ACWE + isotonic"),
]
METRICS = ("roc_auc", "pr_auc", "brier", "ece")

# Reference categorical slots 1 and 2 (blue, orange), validated as a CVD-safe pair.
RAW_COLOR = "#eb6834"
CAL_COLOR = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def _round(obj: Any, digits: int = 6) -> Any:
    """Round floats so reruns are byte-identical (multithreaded sums differ in the last ulp)."""
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: _round(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, digits) for v in obj]
    return obj


def summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label in LABELS:
        metrics = payload["metrics"][label]
        conformal = payload["conformal"][label]
        for key, name in SUMMARY_MODELS:
            if key not in metrics:
                continue
            row = {"target": label, "model": name, "prevalence": metrics[key]["prevalence"], "n_test": metrics[key]["n"]}
            row.update({m: metrics[key][m] for m in METRICS})
            if key == "acwe_calibrated":
                row["conformal_coverage"] = conformal["coverage"]
                row["conformal_avg_set_size"] = conformal["avg_set_size"]
            rows.append(row)
    return rows


def write_summary(payload: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = summary_rows(payload)
    conformal = {label: payload["conformal"][label] for label in LABELS}
    summary = {
        "seed": payload.get("seed"),
        "split_sizes": payload["split_sizes"],
        "rows": rows,
        "conformal": conformal,
        "ablation_by_label": payload.get("ablation_by_label"),
        "online": {k: v for k, v in payload["online"].items() if k != "rolling"},
    }
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(_round(summary), indent=2), encoding="utf-8")

    csv_path = out_dir / "summary.csv"
    fields = ["target", "model", *METRICS, "conformal_coverage", "conformal_avg_set_size", "prevalence", "n_test"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _round(row.get(k)) for k in fields})

    md_path = out_dir / "summary.md"
    lines = [
        "| Target | Model | ROC-AUC | PR-AUC | Brier | ECE |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['target']} | {row['model']} | {row['roc_auc']:.3f} | {row['pr_auc']:.3f} | {row['brier']:.3f} | {row['ece']:.3f} |"
        )
    lines += [
        "",
        "| Target | Prevalence | Conformal target | Coverage | Coverage (positives) | Coverage (negatives) | Avg set size |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label in LABELS:
        c = conformal[label]
        prevalence = payload["metrics"][label]["acwe_calibrated"]["prevalence"]
        lines.append(
            f"| {label} | {prevalence:.3f} | {c['target_coverage']:.2f} | {c['coverage']:.3f} | "
            f"{c['coverage_positive']:.3f} | {c['coverage_negative']:.3f} | {c['avg_set_size']:.2f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "md": md_path}


def _clean_feature(name: str) -> str:
    name = name.split("__", 1)[-1]
    for col in ("sex", "race_group", "smoker", "phys_activity", "fruits_daily", "veggies_daily", "difficulty_walking",
                "healthcare_coverage", "binge_drinking", "high_cholesterol", "chol_check"):
        if name.startswith(col + "_"):
            code = name[len(col) + 1:]
            return f"{col}={code[:-2] if code.endswith('.0') else code}"
    return name


def plot_reliability(payload: dict[str, Any], label: str, path: Path, min_count: int = 30) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = payload["calibration"]["by_label"][label]
    ece_raw = payload["metrics"][label]["acwe_raw"]["ece"]
    ece_cal = payload["metrics"][label]["acwe_calibrated"]["ece"]
    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=150)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(0.84, 0.86, "perfect calibration", color=MUTED, fontsize=8, ha="center", va="bottom", rotation=45, rotation_mode="anchor")
    for key, color, name, ece in (("raw", RAW_COLOR, "ACWE raw", ece_raw), ("calibrated", CAL_COLOR, "ACWE + isotonic", ece_cal)):
        # Sparse bins are noise, not miscalibration; they are left out of the plot (not of ECE).
        pts = [(b["mean_predicted"], b["mean_observed"]) for b in bins[key] if b["count"] >= min_count]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=6, mec="white", mew=1.5, label=f"{name} (ECE {ece:.3f})", zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"Mean predicted risk (10 bins; bins with < {min_count} rows omitted)", color=INK, fontsize=9)
    ax.set_ylabel("Observed prevalence", color=INK)
    ax.set_title(f"{label}: reliability on held-out test", color=INK, fontsize=11, loc="left")
    ax.grid(color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_shap(npz_path: Path, label: str, path: Path, max_display: int = 12) -> Path | None:
    if not npz_path.exists():
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    data = np.load(npz_path, allow_pickle=False)
    names = [_clean_feature(str(n)) for n in data["feature_names"]]
    shap.summary_plot(data["values"], data["x"], feature_names=names, max_display=max_display, show=False, plot_size=(6.4, 4.8))
    fig = plt.gcf()
    fig.axes[0].set_title(f"{label}: SHAP, LightGBM member, {len(data['values'])} test rows", fontsize=9, loc="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_figures(payload: dict[str, Any], results: Path) -> list[Path]:
    figures = results / "figures"
    written = []
    for label in LABELS:
        written.append(plot_reliability(payload, label, figures / f"reliability_{label}.png"))
        shap_path = plot_shap(results / "predictions" / f"shap_{label}.npz", label, figures / f"shap_{label}.png")
        if shap_path:
            written.append(shap_path)
    return written
