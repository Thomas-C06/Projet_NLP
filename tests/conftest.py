"""Small fictional films for deterministic tests, separate from the real corpus."""

import numpy as np
import pytest

from backend.schemas import Movie


class ControlledEncoder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            vector = np.array([float("friendship" in lower), float("space" in lower), float("fear" in lower)])
            vectors.append(vector / max(np.linalg.norm(vector), 1))
        return np.array(vectors)

    encode_corpus = encode


@pytest.fixture
def encoder():
    return ControlledEncoder()


@pytest.fixture
def movies():
    records = [
        dict(tmdb_id=1, title="Test Friendship", overview="A friendship story.", genres=["Drama"],
             themes=["friendship"], mood="A warm friendship.", narrative_style="Character study.",
             setting="A small town.", runtime_minutes=95),
        dict(tmdb_id=2, title="Test Space", overview="A space journey.", genres=["Science Fiction"],
             themes=["space"], mood="An uncertain space journey.", narrative_style="Adventure.",
             setting="Deep space.", runtime_minutes=140),
        dict(tmdb_id=3, title="Test Fear", overview="A story of fear.", genres=["Horror"],
             themes=["fear"], mood="A feeling of fear.", narrative_style="Suspense.",
             setting="An isolated house.", runtime_minutes=None),
        dict(tmdb_id=4, title="Test Missing", overview="A friendship story.", genres=["Drama"],
             runtime_minutes=100),
    ]
    return [Movie(**record, source="Fictional test record", annotation_version="test-v1") for record in records]
