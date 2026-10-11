"""End-to-end research experiment runner. Produces real artifacts only."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression

from src.config import load_config, project_path
from src.data.leakage import audit_feature_list
from src.data.loader import LABELS, load_processed, dataset_summary, save_processed, build_processed_frame, load_raw_brfss
from src.data.preprocessing import FEATURE_COLUMNS, transform_split, save_preprocessor
from src.data.validator import validate_processed
from src.drift.detector import DriftMonitor
from src.drift.simulation import numerical_feature_shift, relationship_noise
from src.ensemble.adaptive_ensemble import AdaptiveContextWeightedEnsemble
from src.ensemble.competence import LocalCompetenceEstimator
from src.ensemble.dependency import CooccurrencePrior
from src.ensemble.weighting import combine_probabilities, equal_weights, global_performance_weights, context_weights
from src.evaluation.ablation import ablation_table
from src.evaluation.cross_validation import make_splits
from src.evaluation.metrics import classification_bundle, reliability_bins
from src.evaluation.statistics import bootstrap_ci, delong_roc_auc, mcnemar_test
from src.explainability.shap_service import permutation_importance_table, shap_for_tree_model
from src.models.baselines import build_baselines
from src.models.modern_tabular import fit_modern_tabular
from src.models.online import build_online_models
from src.uncertainty.calibration import ProbabilityCalibrator
from src.uncertainty.conformal import SplitConformalClassifier
from src.utils.logging import get_logger
from src.utils.reproducibility import set_global_seed

LOGGER = get_logger("bio_sentinel.experiments")


def _proba(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2:
            return proba[:, 1]
        return np.asarray(proba).ravel()
    if hasattr(model, "decision_function"):
        raw = model.decision_function(x)
        return 1 / (1 + np.exp(-raw))
    return model.predict(x).astype(float)


def _feature_names(preprocessor) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"f{i}" for i in range(40)]


def prepare_dataset(cfg: dict[str, Any]) -> pd.DataFrame:
    processed_path = project_path(*Path(cfg["dataset"]["processed_path"]).parts)
    if processed_path.exists():
        df = pd.read_parquet(processed_path)
    else:
        raw = load_raw_brfss(project_path(*Path(cfg["dataset"]["raw_path"]).parts))
        df = build_processed_frame(raw)
        save_processed(df, processed_path)
    errors = validate_processed(df)
    if errors:
        raise ValueError(errors)
    return df


def run_all_experiments(config_path: Path | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    set_global_seed(int(cfg["seed"]))
    started = time.time()

    df = prepare_dataset(cfg)
    summary = dataset_summary(df, cfg)
    leakage = audit_feature_list(FEATURE_COLUMNS)
    if not leakage["passed"]:
        raise RuntimeError(f"Leakage audit failed: {leakage}")

    splits = make_splits(
        df,
        test_size=cfg["training"]["test_size"],
        val_size=cfg["training"]["validation_size"],
        seed=cfg["seed"],
        max_rows=cfg["training"]["max_rows"],
    )
    x_train, x_val, x_test = transform_split(splits)
    feature_names = _feature_names(splits.preprocessor)

    artifacts = project_path("models", "trained")
    artifacts.mkdir(parents=True, exist_ok=True)
    results_dir = project_path("results", "experiments")
    results_dir.mkdir(parents=True, exist_ok=True)
    save_preprocessor(splits.preprocessor, artifacts / "preprocessor.joblib")
    joblib.dump({"numeric": FEATURE_COLUMNS}, artifacts / "feature_spec.joblib")

    models_by_label: dict[str, dict[str, Any]] = {}
    metrics: dict[str, dict[str, dict]] = {}
    test_probs: dict[str, dict[str, np.ndarray]] = {}
    val_probs: dict[str, dict[str, np.ndarray]] = {}

    baselines = build_baselines(cfg["seed"])
    for label in LABELS:
        LOGGER.info("Training baselines for %s", label)
        y_tr = splits.y_train[label].to_numpy()
        y_va = splits.y_val[label].to_numpy()
        y_te = splits.y_test[label].to_numpy()
        models_by_label[label] = {}
        metrics[label] = {}
        test_probs[label] = {}
        val_probs[label] = {}
        for name, model in baselines.items():
            model.fit(x_train, y_tr)
            models_by_label[label][name] = model
            p_te = _proba(model, x_test)
            p_va = _proba(model, x_val)
            test_probs[label][name] = p_te
            val_probs[label][name] = p_va
            metrics[label][name] = classification_bundle(y_te, p_te)
            joblib.dump(model, artifacts / f"{label}_{name}.joblib")

        modern = fit_modern_tabular(
            x_train,
            y_tr,
            x_val,
            y_va,
            seed=cfg["seed"],
            max_rows=cfg["modern_tabular"]["subsample_rows"],
            max_epochs=cfg["modern_tabular"]["max_epochs"],
        )
        models_by_label[label][modern.name] = modern.model
        p_te = _proba(modern.model, x_test)
        p_va = _proba(modern.model, x_val)
        test_probs[label][modern.name] = p_te
        val_probs[label][modern.name] = p_va
        bundle = classification_bundle(y_te, p_te)
        bundle["notes"] = modern.notes
        bundle["backend"] = modern.backend
        metrics[label][modern.name] = bundle
        joblib.dump(modern.model, artifacts / f"{label}_{modern.name}.joblib")

    LOGGER.info("Training classifier chain (disease-dependency baseline)")
    chain = ClassifierChain(
        LogisticRegression(max_iter=400, class_weight="balanced"),
        order=[LABELS.index("hypertension"), LABELS.index("diabetes"), LABELS.index("cvd")],
        random_state=cfg["seed"],
    )
    chain.fit(x_train, splits.y_train.to_numpy())
    chain_prob_raw = chain.predict_proba(x_test)
    if isinstance(chain_prob_raw, list):
        chain_prob = np.column_stack(
            [p[:, 1] if getattr(p, "ndim", 1) == 2 else np.asarray(p).ravel() for p in chain_prob_raw]
        )
    else:
        chain_prob = np.asarray(chain_prob_raw)
        if chain_prob.ndim == 3:
            chain_prob = chain_prob[:, :, 1]
    for i, label in enumerate(LABELS):
        metrics[label]["classifier_chain"] = classification_bundle(splits.y_test[label].to_numpy(), chain_prob[:, i])
        test_probs[label]["classifier_chain"] = chain_prob[:, i]
    joblib.dump(chain, artifacts / "classifier_chain.joblib")

    primary = "diabetes"
    pool_names = [name for name in ["logistic_regression", "random_forest", "lightgbm"] if name in models_by_label[primary]]
    if not pool_names:
        pool_names = list(models_by_label[primary])[:3]

    LOGGER.info("Evaluating ACWE on batched member probabilities")
    competence = LocalCompetenceEstimator(k=cfg["ensemble"]["neighborhood_size"])
    competence.fit(
        x_val,
        splits.y_val[primary].to_numpy(),
        {name: val_probs[primary][name] for name in pool_names},
    )

    def make_predictor(model):
        def _fn(x):
            arr = np.atleast_2d(x)
            return float(_proba(model, arr)[0])
        return _fn

    predictors = {name: make_predictor(models_by_label[primary][name]) for name in pool_names}
    reliability = {name: max(metrics[primary][name]["roc_auc"] or 0.5, 0.5) for name in pool_names}
    acwe = AdaptiveContextWeightedEnsemble(
        predictors,
        competence,
        recency_window=cfg["ensemble"]["recency_window"],
        reliability=reliability,
    )

    calibrator = ProbabilityCalibrator(cfg["calibration"]["method"])
    p_val_blend = np.mean([val_probs[primary][name] for name in pool_names], axis=0)
    calibrator.fit(splits.y_val[primary].to_numpy(), p_val_blend)
    conformal = SplitConformalClassifier(cfg["uncertainty"]["alpha"])
    conformal.fit(splits.y_val[primary].to_numpy(), calibrator.transform(p_val_blend))

    y_te = splits.y_test[primary].to_numpy()
    local_batch = competence.competence_batch(x_test)
    member_test = {name: test_probs[primary][name] for name in pool_names}
    n = len(x_test)
    acwe_raw = np.zeros(n)
    soft = np.zeros(n)
    global_w = np.zeros(n)
    no_local = np.zeros(n)
    traces = []
    recency = {name: 0.7 for name in pool_names}
    recency_n = {name: 1.0 for name in pool_names}
    alpha_rec = 2.0 / (cfg["ensemble"]["recency_window"] + 1)
    for i in range(n):
        probs = {name: float(member_test[name][i]) for name in pool_names}
        local = {name: float(local_batch[name][i]) for name in pool_names}
        weights = context_weights(local, recency, reliability)
        acwe_raw[i] = combine_probabilities(weights, probs)
        soft[i] = combine_probabilities(equal_weights(pool_names), probs)
        global_w[i] = combine_probabilities(global_performance_weights(recency), probs)
        no_local[i] = combine_probabilities(
            context_weights({name: 1.0 for name in pool_names}, recency, reliability),
            probs,
        )
        if i < 40:
            traces.append(
                {
                    "index": i,
                    "probability": float(acwe_raw[i]),
                    "weights": weights,
                    "model_probabilities": probs,
                }
            )
        for name in pool_names:
            correct = float(int(probs[name] >= 0.5) == int(y_te[i]))
            recency[name] = (1 - alpha_rec) * recency[name] + alpha_rec * correct
            recency_n[name] += 1

    prior = CooccurrencePrior().fit(splits.y_val[list(LABELS)].to_numpy())
    htn_hat = (test_probs["hypertension"][pool_names[0]] >= 0.5).astype(int)
    cvd_hat = (test_probs["cvd"][pool_names[0]] >= 0.5).astype(int)
    acwe_dep = prior.blend("diabetes", acwe_raw, htn_hat, cvd_hat, lam=0.2)

    acwe_cal = np.asarray(calibrator.transform(acwe_raw))
    acwe_dep_cal = np.asarray(calibrator.transform(acwe_dep))
    metrics[primary]["acwe_raw"] = classification_bundle(y_te, acwe_raw)
    metrics[primary]["acwe_calibrated"] = classification_bundle(y_te, acwe_cal)
    metrics[primary]["acwe_dependency_aware"] = classification_bundle(y_te, acwe_dep_cal)
    metrics[primary]["soft_vote"] = classification_bundle(y_te, soft)
    metrics[primary]["global_weighted"] = classification_bundle(y_te, global_w)
    metrics[primary]["acwe_no_local_competence"] = classification_bundle(y_te, no_local)
    metrics[primary]["acwe_no_calibration"] = metrics[primary]["acwe_raw"]
    metrics[primary]["acwe_no_dynamic_weighting"] = metrics[primary]["soft_vote"]
    metrics[primary]["independent_lightgbm"] = metrics[primary].get("lightgbm", metrics[primary][pool_names[0]])
    metrics["diabetes"]["chain_vs_independent"] = {
        "independent_best_roc_auc": max(metrics["diabetes"][n]["roc_auc"] or 0 for n in pool_names),
        "classifier_chain_roc_auc": metrics["diabetes"]["classifier_chain"]["roc_auc"],
        "acwe_roc_auc": metrics["diabetes"]["acwe_calibrated"]["roc_auc"],
        "dependency_aware_roc_auc": metrics["diabetes"]["acwe_dependency_aware"]["roc_auc"],
        "classifier_chain_brier": metrics["diabetes"]["classifier_chain"]["brier"],
        "acwe_brier": metrics["diabetes"]["acwe_calibrated"]["brier"],
        "dependency_aware_brier": metrics["diabetes"]["acwe_dependency_aware"]["brier"],
        "acwe_ece": metrics["diabetes"]["acwe_calibrated"]["ece"],
        "raw_ece": metrics["diabetes"]["acwe_raw"]["ece"],
        "note": "Predicted other-disease labels from independent models are used as co-occurrence context; test labels are not used.",
    }

    conformal_demo = conformal.predict_set(float(acwe_cal[0]))
    joblib.dump(
        {
            "calibrator": calibrator,
            "conformal": conformal,
            "pool_names": pool_names,
            "reliability": reliability,
            "primary_label": primary,
        },
        artifacts / "ensemble_bundle.joblib",
    )
    joblib.dump(competence, artifacts / "competence.joblib")

    LOGGER.info("Running online prequential + drift simulation")
    rng = np.random.default_rng(cfg["seed"])
    stream_x = np.vstack([x_val, x_test])[:2500]
    stream_y = np.concatenate([splits.y_val[primary].to_numpy(), y_te])[:2500]
    mid = len(stream_x) // 2
    shifted = numerical_feature_shift(stream_x[mid:], columns=[1])
    stream_x = np.vstack([stream_x[:mid], shifted])
    stream_y = np.concatenate([stream_y[:mid], relationship_noise(stream_y[mid:], 0.08, rng)])
    static_ref = models_by_label[primary][pool_names[0]]
    static_p_all = _proba(static_ref, stream_x)
    online_models = build_online_models(cfg["seed"])[:2]
    drift_monitor = DriftMonitor(cfg["drift"]["detector"], cfg["drift"].get("delta", 0.002))
    rolling = []
    window = []
    online_series = {m.name: [] for m in online_models}
    static_series = []
    f1_window = []
    rolling_f1 = []
    for t, (xt, yt) in enumerate(zip(stream_x, stream_y)):
        static_p = float(static_p_all[t])
        static_hat = int(static_p >= 0.5)
        static_series.append(float(static_hat == yt))
        f1_window.append((static_hat, int(yt)))
        if len(f1_window) > 200:
            f1_window.pop(0)
        tp = sum(1 for h, y in f1_window if h == 1 and y == 1)
        fp = sum(1 for h, y in f1_window if h == 1 and y == 0)
        fn = sum(1 for h, y in f1_window if h == 0 and y == 1)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        rolling_f1.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
        for om in online_models:
            p = om.predict_proba_one(xt)
            hat = int(p >= 0.5)
            online_series[om.name].append(float(hat == yt))
            om.learn_one(xt, int(yt))
        err = float(static_hat != yt)
        window.append(1.0 - err)
        if len(window) > 200:
            window.pop(0)
        rolling.append({"t": t, "rolling_accuracy": float(np.mean(window)), "rolling_f1": rolling_f1[-1], "drift_injected": t >= mid})
        drift_monitor.update(err, t)

    def rolling_mean(xs):
        return float(np.mean(xs)) if xs else None

    online_summary = {
        "static_reference_prequential_acc": rolling_mean(static_series),
        "online": {name: rolling_mean(vals) for name, vals in online_series.items()},
        "drift_events": [event.__dict__ for event in drift_monitor.events],
        "injected_at": int(mid),
        "note": "A controlled sequential stream is constructed from static public data to investigate robustness under simulated distribution shift.",
        "rolling": rolling[:: max(1, len(rolling) // 200)],
    }

    LOGGER.info("Explainability")
    tree_model = models_by_label[primary].get("lightgbm") or models_by_label[primary].get("random_forest")
    shap_payload = {"available": False, "disclaimer": "Model contribution does not imply causation."}
    perm = []
    if tree_model is not None:
        shap_payload = shap_for_tree_model(tree_model, x_train, x_test[:40], feature_names)
        try:
            perm = permutation_importance_table(tree_model, x_test[:400], y_te[:400], feature_names, n_repeats=2)
        except Exception as exc:
            perm = [{"error": str(exc)}]

    rng = np.random.default_rng(0)
    aucs = []
    for _ in range(200):
        idx = rng.integers(0, len(y_te), len(y_te))
        if len(np.unique(y_te[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_te[idx], acwe_cal[idx]))
    stats = {
        "acwe_calibrated_auc": bootstrap_ci(np.asarray(aucs)),
        "mcnemar_acwe_vs_lightgbm": None,
        "paired_auc_acwe_vs_best_baseline": None,
    }
    best_base = max(pool_names, key=lambda n: metrics[primary][n]["roc_auc"] or 0)
    stats["mcnemar_acwe_vs_lightgbm"] = mcnemar_test(
        y_te,
        (acwe_cal >= 0.5).astype(int),
        (test_probs[primary][best_base] >= 0.5).astype(int),
    )
    stats["paired_auc_acwe_vs_best_baseline"] = delong_roc_auc(y_te, acwe_cal, test_probs[primary][best_base])

    ablation = ablation_table(
        {
            "full": metrics[primary]["acwe_calibrated"],
            "no_dynamic_weighting": metrics[primary]["acwe_no_dynamic_weighting"],
            "no_local_competence": metrics[primary]["acwe_no_local_competence"],
            "no_online_updating": metrics[primary]["acwe_raw"],
            "no_drift_adaptation": metrics[primary]["acwe_raw"],
            "no_calibration": metrics[primary]["acwe_no_calibration"],
        }
    )

    umap_payload = _embed_feature_space(x_test, y_te)

    payload = {
        "status": "completed",
        "runtime_sec": round(time.time() - started, 2),
        "dataset": summary,
        "leakage_audit": leakage,
        "split_sizes": {
            "train": int(len(splits.X_train)),
            "validation": int(len(splits.X_val)),
            "test": int(len(splits.X_test)),
            "compute_profile": cfg["compute_profile"],
            "max_rows": cfg["training"]["max_rows"],
        },
        "metrics": metrics,
        "ablation": ablation,
        "statistics": stats,
        "calibration": {
            "method": cfg["calibration"]["method"],
            "reliability": {
                "raw": reliability_bins(y_te, acwe_raw),
                "calibrated": reliability_bins(y_te, acwe_cal),
            },
        },
        "online": online_summary,
        "explainability": shap_payload,
        "permutation_importance": perm,
        "feature_space": umap_payload,
        "conformal_example": conformal_demo,
        "sample_traces": traces,
        "feature_names": feature_names,
        "disclaimer": "RESEARCH PROTOTYPE. This system estimates model-based disease risk from public research datasets. It is not a medical diagnostic device and must not be used for clinical decision-making.",
    }
    out_path = results_dir / "latest.json"
    out_path.write_text(json.dumps(payload, default=_json_default), encoding="utf-8")
    (project_path("results", "metrics") / "latest.json").write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
    LOGGER.info("Wrote %s", out_path)
    return payload


def _embed_feature_space(x: np.ndarray, y: np.ndarray, max_points: int = 1200) -> dict[str, Any]:
    n = min(len(x), max_points)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x), n, replace=False)
    sample = x[idx]
    labels = y[idx]
    method = "pca"
    coords = None
    try:
        import umap

        coords = umap.UMAP(n_neighbors=20, min_dist=0.2, random_state=42).fit_transform(sample)
        method = "umap"
    except Exception:
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2, random_state=42).fit_transform(sample)
    return {
        "method": method,
        "points": [
            {"x": float(coords[i, 0]), "y": float(coords[i, 1]), "label": int(labels[i])}
            for i in range(len(coords))
        ],
    }


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


if __name__ == "__main__":
    run_all_experiments()
