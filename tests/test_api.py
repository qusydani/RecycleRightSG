# tests/test_api.py
"""Integration tests for FastAPI endpoints using a mocked predictor."""
import pytest


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_metrics_shape(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "model" in data
    assert "mAP50" in data
    assert data["mAP50"] == pytest.approx(0.888, abs=1e-3)


def test_model_versions(client):
    resp = client.get("/model/versions")
    assert resp.status_code == 200
    versions = resp.json()
    assert isinstance(versions, list)
    active = [v for v in versions if v["active"]]
    assert len(active) == 1, "Exactly one model should be active"


def test_predict_rejects_non_image(client):
    resp = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 400


def test_predict_rejects_corrupt_image(client):
    resp = client.post(
        "/predict",
        files={"file": ("bad.jpg", b"\xff\xd8corrupt", "image/jpeg")},
    )
    assert resp.status_code == 400


def test_predict_valid_image_returns_detections(client, sample_image_bytes):
    resp = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "detections" in body
    assert isinstance(body["detections"], list)


def test_analytics_shape(client):
    resp = client.get("/analytics")
    assert resp.status_code == 200
    data = resp.json()
    for key in ("model", "mAP50", "prediction_requests", "class_distribution"):
        assert key in data, f"'{key}' missing from /analytics response"
