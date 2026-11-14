"""Download the selected model once and prepare the corpus vectors."""

import json

from backend.catalog import load_catalog
from backend.config import CACHE_PATH, CORPUS_PATH, MODEL_ID, MODEL_PATH
from backend.embeddings import SentenceEncoder
from backend.recommender import Recommender
from backend.schemas import Preferences


def main():
    encoder = SentenceEncoder(MODEL_PATH, CACHE_PATH, allow_download=True)
    encoder.encode(["An emotional story about friendship and discovery."])
    model = encoder.model
    (MODEL_PATH / "preparation.json").write_text(json.dumps({
        "model": MODEL_ID, "device": "cpu", "max_sequence_length": model.max_seq_length,
        "dimension": encoder._dimension(), "chunking": "token chunks, normalised mean",
        "revision": getattr(model[0].auto_model.config, "_commit_hash", None),
    }, indent=2), encoding="utf-8")
    movies = load_catalog(CORPUS_PATH)
    if movies:
        engine = Recommender(movies, encoder)
        query = Preferences(description="I would like an emotional story about friendship and discovery.")
        engine.recommend(query)
        engine.recommend(query.model_copy(update={"method": "sbert_global"}))
    print(f"Local model ready: {MODEL_PATH}")
    print(f"Corpus vectors prepared for {len(movies)} films.")


if __name__ == "__main__":
    main()
