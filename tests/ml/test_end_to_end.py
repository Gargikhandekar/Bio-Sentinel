"""End-to-end pipeline run on synthetic data (see tests/conftest.py)."""

from __future__ import annotations

from src.config import load_config, models_dir, results_dir
from src.data.loader import LABELS


def test_every_target_gets_ensemble_calibration_and_conformal(trained):
    for label in LABELS:
        metrics = trained["metrics"][label]
        for key in ("lightgbm", "lightgbm_calibrated", "soft_vote", "acwe_raw", "acwe_calibrated"):
            assert 0.5 < metrics[key]["roc_auc"] <= 1.0, (label, key)
            assert 0.0 <= metrics[key]["ece"] <= 1.0
        conformal = trained["conformal"][label]
        assert 0.0 <= conformal["coverage"] <= 1.0
        assert 1.0 <= conformal["avg_set_size"] <= 2.0


def test_holdout_split_sizes_add_up(trained):
    sizes = trained["split_sizes"]
    assert sum(sizes["validation_parts"].values()) == sizes["validation"]
    assert sizes["train"] + sizes["validation"] + sizes["test"] == 3000


def test_calibration_improves_ece_of_the_balanced_ensemble(trained):
    # class_weight='balanced' inflates raw probabilities; isotonic should undo most of it.
    m = trained["metrics"]["diabetes"]
    assert m["acwe_calibrated"]["ece"] < m["acwe_raw"]["ece"]


def test_ablation_rows_are_distinct_variants(trained):
    rows = trained["ablation_by_label"]["cvd"]
    assert [r["variant"] for r in rows] == ["full", "no_dynamic_weighting", "no_local_competence", "no_calibration"]
    assert rows[0]["delta_vs_full"] == 0


def test_artifacts_summary_and_figures_written(trained):
    cfg = load_config()
    art, out = models_dir(cfg), results_dir(cfg)
    for label in LABELS:
        assert (art / f"{label}_ensemble_bundle.joblib").exists()
        assert (art / f"{label}_competence.joblib").exists()
        assert (out / "figures" / f"reliability_{label}.png").exists()
        assert (out / "figures" / f"shap_{label}.png").exists()
    for ext in ("json", "csv", "md"):
        assert (out / "metrics" / f"summary.{ext}").exists()
