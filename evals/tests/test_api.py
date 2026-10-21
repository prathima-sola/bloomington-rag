from fastapi.testclient import TestClient

from src import api


def test_api_serves_health_and_ui_while_pipeline_loads(monkeypatch):
    monkeypatch.setattr(api, "load_pipeline", lambda: None)
    monkeypatch.setattr(api, "_ready", False)
    monkeypatch.setattr(api, "_loading", True)
    monkeypatch.setattr(api, "_load_error", None)

    with TestClient(api.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "loading"
        assert client.get("/").status_code == 200
        assert client.get("/ready").status_code == 503
