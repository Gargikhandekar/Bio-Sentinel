from src.data.leakage import audit_feature_list
from src.data.loader import recode_cvd, recode_diabetes, recode_hypertension
from src.data.preprocessing import FEATURE_COLUMNS
from src.ensemble.weighting import combine_probabilities, context_weights, equal_weights
from src.uncertainty.conformal import SplitConformalClassifier
import numpy as np


def test_target_recoding_excludes_pregnancy_and_borderline():
    assert recode_diabetes(1) == 1
    assert recode_diabetes(2) == 0
    assert recode_diabetes(4) == 0
    assert np.isnan(recode_diabetes(7))
    assert recode_hypertension(1) == 1
    assert recode_hypertension(2) == 0
    assert recode_cvd(1) == 1
    assert recode_cvd(2) == 0


def test_leakage_audit_rejects_label_columns():
    failed = audit_feature_list(["bmi", "DIABETE4", "BPHIGH6"])
    assert failed["passed"] is False
    clean = audit_feature_list(FEATURE_COLUMNS)
    assert clean["passed"] is True
    assert "BPHIGH6" not in FEATURE_COLUMNS
    assert "CVDSTRK3" not in FEATURE_COLUMNS


def test_dynamic_weights_normalize_and_are_not_uniform_when_competence_differs():
    weights = context_weights(
        {"a": 0.9, "b": 0.1},
        {"a": 0.8, "b": 0.2},
        {"a": 1.0, "b": 1.0},
    )
    assert abs(sum(weights.values()) - 1) < 1e-6
    assert weights["a"] > weights["b"]
    blended = combine_probabilities(weights, {"a": 0.8, "b": 0.2})
    assert 0.2 < blended < 0.8
    equal = equal_weights(["a", "b"])
    assert equal["a"] == equal["b"]


def test_cooccurrence_prior_blends_without_using_future_labels():
    from src.ensemble.dependency import CooccurrencePrior

    y = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0]])
    prior = CooccurrencePrior().fit(y)
    blended = prior.blend("diabetes", np.array([0.2, 0.8]), np.array([1, 0]), np.array([0, 0]), lam=0.5)
    assert blended.shape == (2,)
    assert np.all((blended >= 0) & (blended <= 1))

    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.9, 0.85, 0.15, 0.8, 0.05, 0.95])
    conf = SplitConformalClassifier(alpha=0.2).fit(y, p)
    out = conf.predict_set(0.92)
    assert 1 in out["prediction_set"]
    assert out["uncertainty"] in {"low", "moderate", "high"}


def test_baseline_factory_returns_fresh_instances():
    from src.models.baselines import build_baselines

    a, b = build_baselines(0), build_baselines(0)
    assert all(a[name] is not b[name] for name in a)


def test_holdout_parts_are_disjoint_and_cover_validation():
    import pandas as pd

    from src.evaluation.cross_validation import split_holdout

    rng = np.random.default_rng(0)
    y = pd.DataFrame(rng.integers(0, 2, size=(900, 3)), columns=["diabetes", "hypertension", "cvd"])
    parts = split_holdout(y, seed=1)
    sets = [set(v.tolist()) for v in parts.values()]
    assert sum(len(s) for s in sets) == len(y)
    assert set.union(*sets) == set(range(len(y)))
    for i in range(3):
        for j in range(i + 1, 3):
            assert not sets[i] & sets[j]


def test_preprocessor_statistics_come_from_training_rows_only():
    from src.evaluation.cross_validation import make_splits
    from src.data.preprocessing import transform_split
    from tests.synthetic import make_synthetic_brfss

    splits = make_splits(make_synthetic_brfss(1500), test_size=0.2, val_size=0.2, seed=0)
    transform_split(splits)
    imputer = splits.preprocessor.named_transformers_["num"].named_steps["imputer"]
    assert imputer.statistics_[1] == splits.X_train["bmi"].median()
    assert imputer.statistics_[1] != splits.X_test["bmi"].median()


def test_conformal_reaches_target_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 40000)
    y = (rng.uniform(0, 1, 40000) < p).astype(int)  # p is perfectly calibrated
    conf = SplitConformalClassifier(alpha=0.1).fit(y[:20000], p[:20000])
    report = conf.evaluate(y[20000:], p[20000:])
    assert abs(report["coverage"] - 0.9) < 0.02
    assert 1.0 <= report["avg_set_size"] <= 2.0
    neg, pos = conf.predict_sets(np.array([0.92]))
    assert bool(pos[0]) == (1 in conf.predict_set(0.92)["prediction_set"])


def test_isotonic_calibrator_is_monotone_and_never_certain():
    from src.uncertainty.calibration import ProbabilityCalibrator

    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p**2).astype(int)
    cal = ProbabilityCalibrator("isotonic").fit(y, p)
    grid = np.linspace(0, 1, 101)
    out = cal.transform(grid)
    assert np.all(np.diff(out) >= 0)
    assert out.min() >= 1e-3 and out.max() <= 1 - 1e-3


def test_acwe_stream_never_uses_the_current_label():
    from src.evaluation.experiments import acwe_stream

    rng = np.random.default_rng(0)
    probs = {"a": rng.uniform(size=50), "b": rng.uniform(size=50)}
    local = {"a": rng.uniform(size=50), "b": rng.uniform(size=50)}
    y = rng.integers(0, 2, 50)
    flipped = y.copy()
    flipped[10] = 1 - flipped[10]
    base = acwe_stream(probs, local, {"a": 0.8, "b": 0.6}, y, recency_window=20)["acwe"]
    other = acwe_stream(probs, local, {"a": 0.8, "b": 0.6}, flipped, recency_window=20)["acwe"]
    np.testing.assert_array_equal(base[:11], other[:11])  # row 10's label only affects rows > 10
    assert not np.array_equal(base[11:], other[11:])
