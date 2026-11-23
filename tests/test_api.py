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


def test_responses_are_saved_only_with_consent(movies, encoder, monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(api, "get_engine", lambda: Recommender(movies, encoder))
    monkeypatch.setattr(api, "RESPONSES_PATH", tmp_path / "responses.jsonl")
    client = TestClient(api.app)
    body = {"description": "I want a warm friendship story."}
    assert client.post("/api/recommend", json=body).json()["saved"] is False
    assert not (tmp_path / "responses.jsonl").exists()
    response = client.post("/api/recommend", json={**body, "save_response": True})
    assert response.json()["saved"] is True
    lines = (tmp_path / "responses.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    assert len(lines) == 1
    assert record["answers"]["description"] == body["description"]
    assert "save_response" not in record["answers"]
    assert len(record["recommendations"]) == 3


def test_critic_is_optional_and_grounded(movies, encoder, monkeypatch):
    from backend import critic

    monkeypatch.setattr(api, "get_engine", lambda: Recommender(movies, encoder))
    captured = {}

    def fake_pitch(summary, movie, matches):
        captured.update(summary=summary, title=movie.title, matches=matches)
        return "A warm pick."
    monkeypatch.setattr(critic, "pitch", fake_pitch)
    client = TestClient(api.app)
    body = {"tmdb_id": 1, "profile_summary": "You are looking for: friendship.", "main_contributions": ["Story"]}
    response = client.post("/api/critic", json=body)
    assert response.json()["text"] == "A warm pick."
    assert captured == {"summary": body["profile_summary"], "title": "Test Friendship", "matches": ["Story"]}
    assert client.post("/api/critic", json={**body, "tmdb_id": 999}).status_code == 404

    def offline(*_):
        raise critic.httpx.ConnectError("offline")
    monkeypatch.setattr(critic, "pitch", offline)
    assert client.post("/api/critic", json=body).status_code == 503
