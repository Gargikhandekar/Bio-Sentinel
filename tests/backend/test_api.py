from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "services" in body


def test_predict_validation_rejects_impossible_bmi():
    response = client.post("/api/predict", json={"features": {"bmi": 400}})
    assert response.status_code == 422
