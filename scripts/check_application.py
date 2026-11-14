"""Run real-corpus checks and timings without inventing relevance judgments."""

import json

from backend.catalog import eligible, load_catalog
from backend.config import CACHE_PATH, CORPUS_PATH, MODEL_PATH, ROOT
from backend.embeddings import SentenceEncoder
from backend.recommender import Recommender
from backend.schemas import Preferences
from scripts.evaluate import METHODS, read_queries


def main():
    movies = load_catalog(CORPUS_PATH)
    if len(movies) != 150:
        raise ValueError("The application check expects the agreed 150-film corpus.")
    engine = Recommender(movies, SentenceEncoder(MODEL_PATH, CACHE_PATH))
    queries = read_queries(ROOT / "evaluation/queries.json")
    records = []
    for method in METHODS:
        warm = Preferences(description="I would like a thoughtful story about friendship.", method=method)
        engine.recommend(warm)
        for query in queries:
            profile = Preferences.model_validate(query["preferences"]).model_copy(update={"method": method})
            response = engine.recommend(profile)
            assert len(response.recommendations) <= 3
            assert all(eligible(item.movie, profile) for item in response.recommendations)
            assert all(0 <= item.affinity <= 100 for item in response.recommendations)
            assert all(response.recommendations[index].affinity >= response.recommendations[index + 1].affinity
                       for index in range(len(response.recommendations) - 1))
            records.append({
                "query_id": query["query_id"], "method": method,
                "latency_ms": response.latency_ms, "returned": len(response.recommendations),
                "top_titles": [item.movie.title for item in response.recommendations],
                "strict_constraints_respected": True,
            })
    summary = {
        method: {
            "queries": 30,
            "average_warm_latency_ms": round(sum(row["latency_ms"] for row in records if row["method"] == method) / 30, 2),
            "strict_constraints_respected": all(row["strict_constraints_respected"] for row in records if row["method"] == method),
        } for method in METHODS
    }
    output = ROOT / "evaluation/technical-checks.json"
    output.write_text(json.dumps({
        "catalog_size": len(movies), "corpus_fingerprint": engine.fingerprint,
        "judgment_type": "technical behaviour only; no human relevance scores",
        "summary": summary, "checks": records,
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
