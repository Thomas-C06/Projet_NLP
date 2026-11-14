"""Exercise the real local model without downloading anything during tests."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.main as api
from backend.config import MODEL_PATH
from backend.embeddings import SentenceEncoder
from backend.recommender import Recommender
from backend.schemas import Preferences


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Prepare the local model before this integration check.")
def test_real_model_chunking_cache_and_api(movies, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    encoder = SentenceEncoder(MODEL_PATH, tmp_path)
    vectors = encoder.encode([
        "A friendship across generations.", "A friendship across generations.",
        "A story about friendship and reconciliation. " * 150,
        "",
    ])
    assert vectors.shape == (4, 384)
    assert np.isfinite(vectors).all()
    assert vectors[0] @ vectors[1] == pytest.approx(1, abs=1e-5)
    assert np.linalg.norm(vectors[2]) == pytest.approx(1, abs=1e-5)
    assert np.linalg.norm(vectors[3]) == 0
    first = encoder.encode_corpus(["A story about friendship."])
    assert np.allclose(first, encoder.encode_corpus(["A story about friendship."]))
    assert len(list(tmp_path.glob("*.npy"))) == 1
    engine = Recommender(movies, encoder)
    monkeypatch.setattr(api, "get_engine", lambda: engine)
    client = TestClient(api.app)
    response = client.post("/api/recommend", json={
        "description": "I want a warm film about friendship and people helping each other.",
        "themes": "friendship", "max_runtime": 110,
    })
    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 2
    assert all(item["movie"]["runtime_minutes"] <= 110 for item in response.json()["recommendations"])
    for method in ("tfidf", "sbert_global", "sbert_criteria"):
        result = engine.recommend(Preferences(description="I would like a story about friendship.", method=method))
        assert len(result.recommendations) == 3
