"""API tests against artifacts trained on synthetic data (see tests/conftest.py)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app

LOW_RISK = {"bmi": 21, "age_group": 2, "gen_health": 1}
HIGH_RISK = {"bmi": 42, "age_group": 12, "gen_health": 5}


@pytest.fixture(scope="module")
def client(trained):
    return TestClient(app)


def test_health_endpoint(client):
    body = client.get("/api/health").json()
    assert body["services"]["MODEL_SERVICES"] == "OPERATIONAL"


@pytest.mark.parametrize("label", ["diabetes", "hypertension", "cvd"])
def test_predict_serves_each_target(client, label):
    response = client.post("/api/predict", json={"label": label, "features": HIGH_RISK})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == label
    assert 0.0 < body["calibrated_probability"] < 1.0
    assert set(body["prediction_set"]) <= {0, 1} and body["prediction_set"]
    assert abs(sum(body["model_weights"].values()) - 1) < 1e-6


def test_predict_risk_is_monotone_in_obvious_cases(client):
    low = client.post("/api/predict", json={"features": LOW_RISK}).json()
    high = client.post("/api/predict", json={"features": HIGH_RISK}).json()
    assert high["calibrated_probability"] > low["calibrated_probability"]


def test_local_explanation_depends_on_the_input(client):
    low = client.post("/api/predict", json={"features": LOW_RISK}).json()["explanation"]["local"]
    high = client.post("/api/predict", json={"features": HIGH_RISK}).json()["explanation"]["local"]
    assert low and high and low != high


def test_predict_handles_all_missing_features(client):
    assert client.post("/api/predict", json={"features": {}}).status_code == 200


def test_predict_rejects_unknown_label(client):
    assert client.post("/api/predict", json={"label": "cancer", "features": {}}).status_code == 422


def test_predict_validation_rejects_impossible_bmi(client):
    assert client.post("/api/predict", json={"features": {"bmi": 400}}).status_code == 422


def test_explain_uses_requested_label(client):
    body = client.post("/api/explain", json={"label": "cvd", "features": HIGH_RISK}).json()
    assert body["prediction"]["label"] == "cvd"
    assert body["global"]


@pytest.mark.parametrize(
    "path", ["/api/models", "/api/experiments", "/api/drift", "/api/research/metrics", "/api/research/ablation", "/api/research/calibration"]
)
def test_read_endpoints_return_results(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert response.json().get("status") != "Experiment not yet executed"
