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
