from fastapi.testclient import TestClient

import backend.main as api
from backend.recommender import Recommender


def test_api_contract_and_validation(movies, encoder, monkeypatch):
    monkeypatch.setattr(api, "get_engine", lambda: Recommender(movies, encoder))
    client = TestClient(api.app)
    assert client.get("/api/health").json()["catalog_size"] == 4
    assert "Drama" in client.get("/api/catalog").json()["genres"]
    response = client.post("/api/recommend", json={"description": "I want a warm friendship story."})
    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 3
    assert client.post("/api/recommend", json={"description": " "}).status_code == 422
    invalid = client.post("/api/recommend", json={"description": "I want a warm friendship story.", "excluded_genres": ["Unknown"]})
    assert invalid.status_code == 422


def test_empty_corpus_is_reported(monkeypatch):
    monkeypatch.setattr(api, "get_engine", lambda: Recommender([]))
    client = TestClient(api.app)
    assert client.get("/api/health").json()["catalog_size"] == 0
    response = client.post("/api/recommend", json={"description": "I want an emotional science fiction story."})
    assert response.status_code == 503


def test_invalid_corpus_returns_service_error(monkeypatch):
    def broken():
        raise ValueError("Invalid JSON")
    monkeypatch.setattr(api, "get_engine", broken)
    assert TestClient(api.app).get("/api/health").status_code == 503
