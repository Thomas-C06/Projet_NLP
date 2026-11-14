"""Create a shared judgment pool, then compare the three ranking methods."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from backend.catalog import eligible, load_catalog
from backend.config import CACHE_PATH, CORPUS_PATH, MODEL_PATH, ROOT
from backend.embeddings import SentenceEncoder
from backend.recommender import Recommender
from backend.schemas import Preferences

METHODS = ["tfidf", "sbert_global", "sbert_criteria"]


def query_fingerprint(query: dict) -> str:
    return hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()


def precision_at_3(ids: list[int], judgments: dict[int, float]) -> float:
    return sum(judgments[movie_id] >= 2 for movie_id in ids[:3]) / 3


def ndcg_at_3(ids: list[int], judgments: dict[int, float]) -> float:
    def dcg(values):
        return sum((2 ** value - 1) / math.log2(index + 2) for index, value in enumerate(values[:3]))
    ideal = dcg(sorted(judgments.values(), reverse=True))
    return dcg([judgments[movie_id] for movie_id in ids[:3]]) / ideal if ideal else 0.0


def read_queries(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("Query identifiers must be unique.")
    for row in rows:
        if row["split"] not in ("validation", "test"):
            raise ValueError("Each query must belong to validation or test.")
        Preferences.model_validate(row["preferences"])
    return rows


def make_pool(engine, queries, output: Path):
    if output.exists():
        raise ValueError("The judgment file exists. Choose another output to keep existing annotations.")
    rows = []
    for query in queries:
        profile = Preferences.model_validate(query["preferences"])
        candidates = set()
        for method in METHODS:
            result = engine.recommend(profile.model_copy(update={"method": method}), top_k=10)
            candidates.update(item.movie.tmdb_id for item in result.recommendations)
        titles = {movie.tmdb_id: movie.title for movie in engine.movies}
        for movie_id in sorted(candidates):
            rows.append({
                "query_id": query["query_id"], "split": query["split"], "tmdb_id": movie_id,
                "title": titles[movie_id], "corpus_fingerprint": engine.fingerprint,
                "query_fingerprint": query_fingerprint(query),
                "annotator": "", "relevance": "",
            })
    if not rows:
        raise ValueError("No candidates were retrieved. Check the corpus and constraints.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} judgments to complete: {output}")


def read_judgments(path: Path, corpus_fingerprint: str, queries: list | None = None):
    scores, seen = {}, set()
    expected = {query["query_id"]: query_fingerprint(query) for query in queries or []}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["corpus_fingerprint"] != corpus_fingerprint:
                raise ValueError("The corpus changed after pooling. Rebuild the pool and review the judgments.")
            if queries is not None and row.get("query_fingerprint") != expected.get(row["query_id"]):
                raise ValueError("A query changed after pooling. Rebuild and review the corresponding judgments.")
            if not row["annotator"].strip() or row["relevance"] not in {"0", "1", "2", "3"}:
                raise ValueError("Complete every judgment with an annotator and a relevance from 0 to 3.")
            key = (row["query_id"], int(row["tmdb_id"]))
            reviewer_key = (*key, row["annotator"].strip())
            if reviewer_key in seen:
                raise ValueError("An annotator has duplicate judgments for the same query and film.")
            seen.add(reviewer_key)
            scores.setdefault(key, []).append(int(row["relevance"]))
    return {key: statistics.mean(values) for key, values in scores.items()}


def evaluate(engine, queries, judgments, split: str):
    rows = []
    selected = [query for query in queries if query["split"] == split]
    if not selected:
        raise ValueError("There are no queries in the selected split.")
    for method in METHODS:
        # Warm the model and corpus vectors before measuring search latency.
        warm = Preferences.model_validate(selected[0]["preferences"]).model_copy(update={"method": method})
        engine.recommend(warm)
        for query in selected:
            profile = Preferences.model_validate(query["preferences"]).model_copy(update={"method": method})
            query_scores = {movie_id: value for (query_id, movie_id), value in judgments.items()
                            if query_id == query["query_id"]}
            result = engine.recommend(profile)
            ids = [item.movie.tmdb_id for item in result.recommendations]
            if not query_scores or any(movie_id not in query_scores for movie_id in ids):
                raise ValueError(f"Missing pooled judgments for {query['query_id']}. Do not treat unjudged films as irrelevant.")
            valid = sum(eligible(item.movie, profile) for item in result.recommendations)
            rows.append({
                "query_id": query["query_id"], "method": method,
                "precision_at_3": precision_at_3(ids, query_scores),
                "ndcg_at_3": ndcg_at_3(ids, query_scores),
                "constraint_compliance": valid / len(ids) if ids else None,
                "latency_ms": result.latency_ms, "returned_count": len(ids),
            })
    summary = {}
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        summary[method] = {
            metric: statistics.mean(row[metric] for row in subset if row[metric] is not None)
            if any(row[metric] is not None for row in subset) else None
            for metric in ("precision_at_3", "ndcg_at_3", "constraint_compliance", "latency_ms", "returned_count")
        }
    return {"split": split, "query_count": len(selected), "corpus_fingerprint": engine.fingerprint,
            "relevance_threshold": 2, "methods": summary, "per_query": rows,
            "limitations": "Small query set; nDCG uses the judged candidate pool, not exhaustive corpus judgments."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["pool", "run"])
    parser.add_argument("--queries", type=Path, default=ROOT / "evaluation/queries.json")
    parser.add_argument("--judgments", type=Path, default=ROOT / "evaluation/judgments.csv")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        movies = load_catalog(CORPUS_PATH)
        if not movies:
            raise ValueError("Collect and annotate the film corpus before evaluation.")
        engine = Recommender(movies, SentenceEncoder(MODEL_PATH, CACHE_PATH))
        queries = read_queries(args.queries)
        if args.action == "pool":
            make_pool(engine, queries, args.judgments)
        else:
            judgments = read_judgments(args.judgments, engine.fingerprint, queries)
            result = evaluate(engine, queries, judgments, args.split)
            output = args.output or ROOT / f"evaluation/results/{args.split}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            print(json.dumps(result["methods"], indent=2))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
