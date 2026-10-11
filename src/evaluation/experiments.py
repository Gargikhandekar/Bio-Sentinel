"""End-to-end research experiment runner. Produces real artifacts only.

Data flow. No partition is used for more than one fitting step:

    train (60%)                -> preprocessor, member models
    validation (20%), thirds:
        competence             -> ACWE kNN competence and reliability weights
        calibration            -> isotonic calibrators
        conformal              -> split-conformal quantile
    test (20%)                 -> reported metrics only
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.multioutput import ClassifierChain

from src.config import load_config, models_dir, project_path, results_dir
from src.data.leakage import audit_feature_list
from src.data.loader import LABELS, build_processed_frame, dataset_summary, load_raw_brfss, save_processed
from src.data.preprocessing import FEATURE_COLUMNS, save_preprocessor, transform_split
from src.data.validator import validate_processed
from src.drift.detector import DriftMonitor
from src.drift.simulation import numerical_feature_shift, relationship_noise
from src.ensemble.competence import LocalCompetenceEstimator
from src.ensemble.dependency import CooccurrencePrior
from src.ensemble.weighting import combine_probabilities, context_weights, equal_weights, global_performance_weights
from src.evaluation.ablation import ablation_table
from src.evaluation.cross_validation import make_splits, split_holdout
from src.evaluation.metrics import classification_bundle, reliability_bins
from src.evaluation.statistics import bootstrap_ci, delong_roc_auc, mcnemar_test
from src.explainability.shap_service import permutation_importance_table, shap_for_tree_model, shap_matrix
from src.models.baselines import build_baselines
from src.models.modern_tabular import fit_modern_tabular
from src.models.online import build_online_models
from src.uncertainty.calibration import ProbabilityCalibrator
from src.uncertainty.conformal import SplitConformalClassifier
from src.utils.logging import get_logger
from src.utils.reproducibility import set_global_seed

LOGGER = get_logger("bio_sentinel.experiments")

POOL = ("logistic_regression", "random_forest", "lightgbm")
PRIMARY = "diabetes"
DISCLAIMER = (
    "RESEARCH PROTOTYPE. This system estimates model-based disease risk from public research datasets. "
    "It is not a medical diagnostic device and must not be used for clinical decision-making."
)


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


def acwe_stream(
    member_probs: dict[str, np.ndarray],
    local_competence: dict[str, np.ndarray],
    reliability: dict[str, float],
    y: np.ndarray,
    recency_window: int,
) -> dict[str, Any]:
    """Score a stream prequentially: predict row i, then update recency with its label.

    Recency at row i only uses labels of rows < i (test-then-train), so no row's
    own label influences its prediction. Variants share the stream for ablation.
    """
    names = list(member_probs)
    n = len(y)
    out = {key: np.zeros(n) for key in ("acwe", "soft_vote", "global_weighted", "no_local_competence")}
    recency = {name: 0.7 for name in names}
    alpha_rec = 2.0 / (recency_window + 1)
    flat_local = {name: 1.0 for name in names}
    traces = []
    for i in range(n):
        probs = {name: float(member_probs[name][i]) for name in names}
        local = {name: float(local_competence[name][i]) for name in names}
        weights = context_weights(local, recency, reliability)
        out["acwe"][i] = combine_probabilities(weights, probs)
        out["soft_vote"][i] = combine_probabilities(equal_weights(names), probs)
        out["global_weighted"][i] = combine_probabilities(global_performance_weights(recency), probs)
        out["no_local_competence"][i] = combine_probabilities(context_weights(flat_local, recency, reliability), probs)
        if i < 40:
            traces.append({"index": i, "probability": float(out["acwe"][i]), "weights": weights, "model_probabilities": probs})
        for name in names:
            correct = float(int(probs[name] >= 0.5) == int(y[i]))
            recency[name] = (1 - alpha_rec) * recency[name] + alpha_rec * correct
    out["traces"] = traces
    return out


def _train_members(label, cfg, splits, x_train, x_val, x_test, parts, artifacts):
    y_tr = splits.y_train[label].to_numpy()
    y_va = splits.y_val[label].to_numpy()
    y_te = splits.y_test[label].to_numpy()
    models, val_p, test_p, metrics = {}, {}, {}, {}
    # Fresh instances per label: re-fitting shared objects would leave every
    # label's entry pointing at the model fitted on the last label (cvd).
    for name, model in build_baselines(cfg["seed"]).items():
        model.fit(x_train, y_tr)
        models[name] = model
        val_p[name] = _proba(model, x_val)
        test_p[name] = _proba(model, x_test)
        metrics[name] = classification_bundle(y_te, test_p[name])
        joblib.dump(model, artifacts / f"{label}_{name}.joblib")

    comp = parts["competence"]
    modern = fit_modern_tabular(
        x_train,
        y_tr,
        x_val[comp],  # early stopping (TabNet only) touches the competence third, nothing downstream
        y_va[comp],
        seed=cfg["seed"],
        max_rows=cfg["modern_tabular"]["subsample_rows"],
        max_epochs=cfg["modern_tabular"]["max_epochs"],
    )
    models[modern.name] = modern.model
    val_p[modern.name] = _proba(modern.model, x_val)
    test_p[modern.name] = _proba(modern.model, x_test)
    bundle = classification_bundle(y_te, test_p[modern.name])
    bundle["notes"] = modern.notes
    bundle["backend"] = modern.backend
    metrics[modern.name] = bundle
    joblib.dump(modern.model, artifacts / f"{label}_{modern.name}.joblib")
    return models, val_p, test_p, metrics


def _ensemble_for_label(label, cfg, splits, x_val, x_test, parts, members, other_hat, prior, artifacts):
    """ACWE, calibration, conformal, ablation and statistics for one target."""
    models, val_p, test_p, metrics = members
    y_va = splits.y_val[label].to_numpy()
    y_te = splits.y_test[label].to_numpy()
    comp, cal, conf = parts["competence"], parts["calibration"], parts["conformal"]
    pool = [name for name in POOL if name in models]
    window = cfg["ensemble"]["recency_window"]

    competence = LocalCompetenceEstimator(k=cfg["ensemble"]["neighborhood_size"])
    competence.fit(x_val[comp], y_va[comp], {name: val_p[name][comp] for name in pool})
    reliability = {name: max(float(roc_auc_score(y_va[comp], val_p[name][comp])), 0.5) for name in pool}

    def stream(x, idx, probs_by_model, y):
        probs = {name: probs_by_model[name] if idx is None else probs_by_model[name][idx] for name in pool}
        return acwe_stream(probs, competence.competence_batch(x), reliability, y, window)

    s_cal = stream(x_val[cal], cal, val_p, y_va[cal])
    s_conf = stream(x_val[conf], conf, val_p, y_va[conf])
    s_te = stream(x_test, None, test_p, y_te)

    others = [o for o in LABELS if o != label]
    dep_cal = prior.blend(label, s_cal["acwe"], other_hat["val"][others[0]][cal], other_hat["val"][others[1]][cal], lam=0.2)
    dep_te = prior.blend(label, s_te["acwe"], other_hat["test"][others[0]], other_hat["test"][others[1]], lam=0.2)

    # Each variant gets its own calibrator fitted on the calibration third only.
    on_cal = {
        "acwe": s_cal["acwe"],
        "soft_vote": s_cal["soft_vote"],
        "global_weighted": s_cal["global_weighted"],
        "acwe_no_local_competence": s_cal["no_local_competence"],
        "acwe_dependency_aware": dep_cal,
    }
    on_test = {
        "acwe": s_te["acwe"],
        "soft_vote": s_te["soft_vote"],
        "global_weighted": s_te["global_weighted"],
        "acwe_no_local_competence": s_te["no_local_competence"],
        "acwe_dependency_aware": dep_te,
    }
    if "lightgbm" in models:
        on_cal["lightgbm"] = val_p["lightgbm"][cal]
        on_test["lightgbm"] = test_p["lightgbm"]
    method = cfg["calibration"]["method"]
    calibrators = {name: ProbabilityCalibrator(method).fit(y_va[cal], p) for name, p in on_cal.items()}
    calibrated = {name: np.asarray(calibrators[name].transform(on_test[name])) for name in on_test}

    metrics["acwe_raw"] = classification_bundle(y_te, s_te["acwe"])
    metrics["acwe_calibrated"] = classification_bundle(y_te, calibrated["acwe"])
    metrics["soft_vote"] = classification_bundle(y_te, s_te["soft_vote"])
    metrics["soft_vote_calibrated"] = classification_bundle(y_te, calibrated["soft_vote"])
    metrics["global_weighted"] = classification_bundle(y_te, s_te["global_weighted"])
    metrics["global_weighted_calibrated"] = classification_bundle(y_te, calibrated["global_weighted"])
    metrics["acwe_no_local_competence"] = classification_bundle(y_te, calibrated["acwe_no_local_competence"])
    metrics["acwe_dependency_aware"] = classification_bundle(y_te, calibrated["acwe_dependency_aware"])
    if "lightgbm" in calibrated:
        metrics["lightgbm_calibrated"] = classification_bundle(y_te, calibrated["lightgbm"])

    # Conformal: calibrated ACWE scores on the conformal third, coverage measured on test.
    conformal = SplitConformalClassifier(cfg["uncertainty"]["alpha"])
    conformal.fit(y_va[conf], calibrators["acwe"].transform(s_conf["acwe"]))
    conformal_report = conformal.evaluate(y_te, calibrated["acwe"])
    conformal_report["n_calibration"] = int(len(conf))

    # The comparison baseline is chosen on validation data, never on test.
    best_base = max(pool, key=lambda n: roc_auc_score(y_va[cal], val_p[n][cal]))
    rng = np.random.default_rng(cfg["seed"])
    aucs = []
    for _ in range(200):
        idx = rng.integers(0, len(y_te), len(y_te))
        if len(np.unique(y_te[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_te[idx], calibrated["acwe"][idx]))
    stats = {
        "acwe_calibrated_auc": bootstrap_ci(np.asarray(aucs)),
        "best_baseline_by_validation": best_base,
        "mcnemar_acwe_vs_best_baseline": mcnemar_test(
            y_te, (calibrated["acwe"] >= 0.5).astype(int), (test_p[best_base] >= 0.5).astype(int)
        ),
        "paired_auc_acwe_vs_best_baseline": delong_roc_auc(y_te, calibrated["acwe"], test_p[best_base]),
    }

    ablation = ablation_table(
        {
            "full": metrics["acwe_calibrated"],
            "no_dynamic_weighting": metrics["soft_vote_calibrated"],
            "no_local_competence": metrics["acwe_no_local_competence"],
            "no_calibration": metrics["acwe_raw"],
        }
    )

    joblib.dump(
        {
            "label": label,
            "calibrator": calibrators["acwe"],
            "conformal": conformal,
            "pool_names": pool,
            "reliability": reliability,
            "recency_window": window,
            "shap_model": "lightgbm" if "lightgbm" in models else None,
        },
        artifacts / f"{label}_ensemble_bundle.joblib",
    )
    joblib.dump(competence, artifacts / f"{label}_competence.joblib")

    return {
        "pool": pool,
        "conformal": conformal_report,
        "conformal_model": conformal,
        "statistics": stats,
        "ablation": ablation,
        "reliability": {
            "raw": reliability_bins(y_te, s_te["acwe"]),
            "calibrated": reliability_bins(y_te, calibrated["acwe"]),
        },
        "acwe_calibrated_test": calibrated["acwe"],
        "traces": s_te["traces"],
    }


def _online_drift_experiment(cfg, x_val, x_test, y_val, y_test, static_model) -> dict[str, Any]:
    """Prequential comparison on a constructed stream with an injected shift halfway."""
    rng = np.random.default_rng(cfg["seed"])
    stream_x = np.vstack([x_val, x_test])[:2500]
    stream_y = np.concatenate([y_val, y_test])[:2500]
    mid = len(stream_x) // 2
    shifted = numerical_feature_shift(stream_x[mid:], columns=[1])  # column 1 is scaled BMI
    stream_x = np.vstack([stream_x[:mid], shifted])
    stream_y = np.concatenate([stream_y[:mid], relationship_noise(stream_y[mid:], 0.08, rng)])
    static_p_all = _proba(static_model, stream_x)
    online_models = build_online_models(cfg["seed"])[:2]
    drift_monitor = DriftMonitor(cfg["drift"]["detector"], cfg["drift"].get("delta", 0.002))
    rolling, window, f1_window = [], [], []
    online_series = {m.name: [] for m in online_models}
    static_series, majority_series = [], []
    positives_seen = 0
    for t, (xt, yt) in enumerate(zip(stream_x, stream_y)):
        # Majority-class reference: predicts the class seen most often so far.
        majority_hat = int(positives_seen * 2 > t)
        majority_series.append(float(majority_hat == yt))
        positives_seen += int(yt)
        static_hat = int(static_p_all[t] >= 0.5)
        static_series.append(float(static_hat == yt))
        f1_window.append((static_hat, int(yt)))
        if len(f1_window) > 200:
            f1_window.pop(0)
        tp = sum(1 for h, y in f1_window if h == 1 and y == 1)
        fp = sum(1 for h, y in f1_window if h == 1 and y == 0)
        fn = sum(1 for h, y in f1_window if h == 0 and y == 1)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        for om in online_models:
            online_series[om.name].append(float(int(om.predict_proba_one(xt) >= 0.5) == yt))
            om.learn_one(xt, int(yt))
        err = float(static_hat != yt)
        window.append(1.0 - err)
        if len(window) > 200:
            window.pop(0)
        rolling.append(
            {
                "t": t,
                "rolling_accuracy": float(np.mean(window)),
                "rolling_f1": 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec),
                "drift_injected": t >= mid,
            }
        )
        drift_monitor.update(err, t)

    def mean(xs):
        return float(np.mean(xs)) if xs else None

    return {
        "static_reference_prequential_acc": mean(static_series),
        "majority_class_prequential_acc": mean(majority_series),
        "online": {name: mean(vals) for name, vals in online_series.items()},
        "drift_events": [event.__dict__ for event in drift_monitor.events],
        "injected_at": int(mid),
        "note": (
            "A controlled sequential stream is constructed from static public data to investigate robustness "
            "under simulated distribution shift. Accuracy is threshold-dependent and should be read against "
            "the majority-class reference; the static model uses balanced class weights."
        ),
        "rolling": rolling[:: max(1, len(rolling) // 200)],
    }


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
    parts = split_holdout(splits.y_val, cfg["seed"])
    x_train, x_val, x_test = transform_split(splits)  # preprocessor is fitted on x_train only
    feature_names = _feature_names(splits.preprocessor)

    artifacts = models_dir(cfg)
    artifacts.mkdir(parents=True, exist_ok=True)
    out_dir = results_dir(cfg)
    for sub in ("experiments", "metrics", "predictions"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    save_preprocessor(splits.preprocessor, artifacts / "preprocessor.joblib")

    members = {}
    for label in LABELS:
        LOGGER.info("Training member models for %s", label)
        members[label] = _train_members(label, cfg, splits, x_train, x_val, x_test, parts, artifacts)
    metrics = {label: members[label][3] for label in LABELS}

    LOGGER.info("Training classifier chain (disease-dependency baseline)")
    chain = ClassifierChain(
        LogisticRegression(max_iter=400, class_weight="balanced"),
        order=[LABELS.index("hypertension"), LABELS.index("diabetes"), LABELS.index("cvd")],
        random_state=cfg["seed"],
    )
    chain.fit(x_train, splits.y_train.to_numpy())
    chain_prob = np.asarray(chain.predict_proba(x_test))
    for i, label in enumerate(LABELS):
        metrics[label]["classifier_chain"] = classification_bundle(splits.y_test[label].to_numpy(), chain_prob[:, i])
    joblib.dump(chain, artifacts / "classifier_chain.joblib")

    # Co-occurrence prior from training labels; other diseases enter only as *predicted* labels.
    prior = CooccurrencePrior().fit(splits.y_train[list(LABELS)].to_numpy())
    ref = POOL[0]
    other_hat = {
        "val": {label: (members[label][1][ref] >= 0.5).astype(int) for label in LABELS},
        "test": {label: (members[label][2][ref] >= 0.5).astype(int) for label in LABELS},
    }

    ensembles = {}
    for label in LABELS:
        LOGGER.info("ACWE, calibration and conformal for %s", label)
        ensembles[label] = _ensemble_for_label(
            label, cfg, splits, x_val, x_test, parts, members[label], other_hat, prior, artifacts
        )
        m = metrics[label]
        m["chain_vs_independent"] = {
            "independent_best_roc_auc": max(m[n]["roc_auc"] or 0 for n in ensembles[label]["pool"]),
            "classifier_chain_roc_auc": m["classifier_chain"]["roc_auc"],
            "acwe_roc_auc": m["acwe_calibrated"]["roc_auc"],
            "dependency_aware_roc_auc": m["acwe_dependency_aware"]["roc_auc"],
            "classifier_chain_brier": m["classifier_chain"]["brier"],
            "acwe_brier": m["acwe_calibrated"]["brier"],
            "dependency_aware_brier": m["acwe_dependency_aware"]["brier"],
            "acwe_ece": m["acwe_calibrated"]["ece"],
            "raw_ece": m["acwe_raw"]["ece"],
            "note": "Predicted other-disease labels from independent models are used as co-occurrence context; test labels are not used.",
        }

    LOGGER.info("Online prequential + drift simulation (%s)", PRIMARY)
    online_summary = _online_drift_experiment(
        cfg,
        x_val,
        x_test,
        splits.y_val[PRIMARY].to_numpy(),
        splits.y_test[PRIMARY].to_numpy(),
        members[PRIMARY][0][ensembles[PRIMARY]["pool"][0]],
    )

    LOGGER.info("Explainability")
    shap_by_label = {}
    perm = []
    for label in LABELS:
        tree_model = members[label][0].get("lightgbm") or members[label][0].get("random_forest")
        if tree_model is None:
            continue
        shap_by_label[label] = shap_for_tree_model(tree_model, x_train, x_test[:40], feature_names)
        n_explain = min(1000, len(x_test))
        values = shap_matrix(tree_model, x_test[:n_explain])
        if values is not None:
            np.savez_compressed(
                out_dir / "predictions" / f"shap_{label}.npz",
                values=values,
                x=x_test[:n_explain],
                feature_names=np.asarray(feature_names),
            )
        if label == PRIMARY:
            y_te = splits.y_test[label].to_numpy()
            try:
                perm = permutation_importance_table(tree_model, x_test[:400], y_te[:400], feature_names, n_repeats=2)
            except Exception as exc:
                perm = [{"error": str(exc)}]

    payload = {
        "status": "completed",
        "runtime_sec": round(time.time() - started, 2),
        "seed": cfg["seed"],
        "dataset": summary,
        "leakage_audit": leakage,
        "split_sizes": {
            "train": int(len(splits.X_train)),
            "validation": int(len(splits.X_val)),
            "validation_parts": {name: int(len(idx)) for name, idx in parts.items()},
            "test": int(len(splits.X_test)),
            "compute_profile": cfg["compute_profile"],
            "max_rows": cfg["training"]["max_rows"],
        },
        "metrics": metrics,
        "conformal": {label: ensembles[label]["conformal"] for label in LABELS},
        "ablation": ensembles[PRIMARY]["ablation"],
        "ablation_by_label": {label: ensembles[label]["ablation"] for label in LABELS},
        "statistics": ensembles[PRIMARY]["statistics"],
        "statistics_by_label": {label: ensembles[label]["statistics"] for label in LABELS},
        "calibration": {
            "method": cfg["calibration"]["method"],
            "reliability": ensembles[PRIMARY]["reliability"],
            "by_label": {label: ensembles[label]["reliability"] for label in LABELS},
        },
        "online": online_summary,
        "explainability": shap_by_label.get(PRIMARY, {"available": False}),
        "explainability_by_label": shap_by_label,
        "permutation_importance": perm,
        "feature_space": _embed_feature_space(x_test, splits.y_test[PRIMARY].to_numpy()),
        "conformal_example": ensembles[PRIMARY]["conformal_model"].predict_set(
            float(ensembles[PRIMARY]["acwe_calibrated_test"][0])
        ),
        "sample_traces": ensembles[PRIMARY]["traces"],
        "feature_names": feature_names,
        "disclaimer": DISCLAIMER,
    }
    out_path = out_dir / "experiments" / "latest.json"
    out_path.write_text(json.dumps(payload, default=_json_default), encoding="utf-8")
    LOGGER.info("Wrote %s", out_path)
    return payload


def _embed_feature_space(x: np.ndarray, y: np.ndarray, max_points: int = 1200) -> dict[str, Any]:
    n = min(len(x), max_points)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x), n, replace=False)
    sample = x[idx]
    labels = y[idx]
    method = "pca"
    try:
        import umap

        coords = umap.UMAP(n_neighbors=20, min_dist=0.2, random_state=42).fit_transform(sample)
        method = "umap"
    except Exception:
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2, random_state=42).fit_transform(sample)
    return {
        "method": method,
        "points": [{"x": float(coords[i, 0]), "y": float(coords[i, 1]), "label": int(labels[i])} for i in range(len(coords))],
    }


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


if __name__ == "__main__":
    run_all_experiments()
