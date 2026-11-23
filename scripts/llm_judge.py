"""Produce automatic (silver) relevance judgments with a local LLM through Ollama.

These judgments are NOT human annotations. They give a reproducible first
estimate of ranking quality while independent human judgments are pending.
The judge only sees source metadata (title, year, runtime, genres, overview and
TMDB keywords), never the estimated atmosphere or narrative labels produced by
our own encoder, so that it does not favour the criterion-level method.

Usage:
    python -m scripts.llm_judge                    # judge the current pool
    python -m scripts.evaluate run --split validation --judgments evaluation/judgments-llm.csv
"""

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from backend.catalog import load_catalog
from backend.config import CACHE_PATH, CORPUS_PATH, MODEL_PATH, ROOT
from backend.embeddings import SentenceEncoder
from backend.recommender import Recommender
from backend.schemas import Preferences
from scripts.evaluate import METHODS, query_fingerprint, read_queries

PROMPT = """You are an expert film critic acting as a relevance assessor for a movie recommendation benchmark.

A user described the film they want to watch:
{request}

Candidate film:
Title: {title} ({year})
Runtime: {runtime}
Genres: {genres}
Keywords: {keywords}
Synopsis: {overview}

Using the synopsis and your general knowledge of this film, rate how well it fits the request:
0 = unrelated, or it fails an important expectation
1 = weak connection, important expectations are missing
2 = relevant, most important expectations are covered
3 = highly relevant, a strong contextual match

Hard limits (maximum duration, excluded genres) are enforced elsewhere; judge the positive expectations.
Answer with JSON only: {{"relevance": <0-3>, "reason": "<one short sentence>"}}"""


def describe_request(preferences: dict) -> str:
    labels = {
        "description": "Description", "themes": "Themes", "mood": "Atmosphere",
        "narrative_style": "Storytelling", "setting": "Setting or period",
    }
    lines = [f"- {label}: {preferences[key]}" for key, label in labels.items() if preferences.get(key)]
    if preferences.get("preferred_genres"):
        lines.append("- Preferred genres: " + ", ".join(preferences["preferred_genres"]))
    return "\n".join(lines)


def judge(client: httpx.Client, url: str, model: str, preferences: dict, movie) -> dict:
    prompt = PROMPT.format(
        request=describe_request(preferences), title=movie.title, year=movie.release_year or "unknown",
        runtime=f"{movie.runtime_minutes} min" if movie.runtime_minutes else "unknown",
        genres=", ".join(movie.genres), keywords=", ".join(movie.themes or []) or "none",
        overview=movie.overview or "unavailable",
    )
    response = client.post(f"{url}/api/generate", json={
        "model": model, "prompt": prompt, "stream": False, "think": False, "format": "json",
        "options": {"temperature": 0, "seed": 7, "num_ctx": 4096},
    })
    response.raise_for_status()
    answer = json.loads(response.json()["response"])
    relevance = int(answer["relevance"])
    if relevance not in (0, 1, 2, 3):
        raise ValueError(f"Unexpected relevance {relevance}")
    return {"relevance": relevance, "reason": str(answer.get("reason", ""))[:300]}


def pool(engine, queries, depth: int) -> list[tuple[dict, int]]:
    pairs = []
    for query in queries:
        profile = Preferences.model_validate(query["preferences"])
        candidates = set()
        for method in METHODS:
            result = engine.recommend(profile.model_copy(update={"method": method}), top_k=depth)
            candidates.update(item.movie.tmdb_id for item in result.recommendations)
        pairs.extend((query, movie_id) for movie_id in sorted(candidates))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--ollama", default="http://localhost:11434")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--depth", type=int, default=10, help="Top-k pooled from each method.")
    parser.add_argument("--queries", type=Path, default=ROOT / "evaluation/queries.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "evaluation/llm-judgments-cache.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/judgments-llm.csv")
    args = parser.parse_args()

    movies = load_catalog(CORPUS_PATH)
    by_id = {movie.tmdb_id: movie for movie in movies}
    engine = Recommender(movies, SentenceEncoder(MODEL_PATH, CACHE_PATH))
    queries = read_queries(args.queries)
    # The cache key ties a judgment to the exact query text, film and judge model.
    cache = json.loads(args.cache.read_text(encoding="utf-8")) if args.cache.exists() else {}
    pairs = pool(engine, queries, args.depth)
    todo = [(q, m) for q, m in pairs if f"{args.model}|{query_fingerprint(q)}|{m}" not in cache]
    print(f"{len(pairs)} pooled pairs, {len(todo)} to judge with {args.model}.")
    def run(pair):
        query, movie_id = pair
        return f"{args.model}|{query_fingerprint(query)}|{movie_id}", judge(
            client, args.ollama, args.model, query["preferences"], by_id[movie_id])

    # Ollama serves a few requests in parallel; the cache is saved regularly so a stop loses little.
    with httpx.Client(timeout=300) as client, ThreadPoolExecutor(args.workers) as pool_executor:
        for count, (key, verdict) in enumerate(pool_executor.map(run, todo), 1):
            cache[key] = verdict
            if count % 20 == 0 or count == len(todo):
                args.cache.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")
                print(f"  {count}/{len(todo)} judged", flush=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "query_id", "split", "tmdb_id", "title", "corpus_fingerprint", "query_fingerprint",
            "annotator", "relevance", "reason",
        ])
        writer.writeheader()
        for query, movie_id in pairs:
            verdict = cache[f"{args.model}|{query_fingerprint(query)}|{movie_id}"]
            writer.writerow({
                "query_id": query["query_id"], "split": query["split"], "tmdb_id": movie_id,
                "title": by_id[movie_id].title, "corpus_fingerprint": engine.fingerprint,
                "query_fingerprint": query_fingerprint(query), "annotator": f"llm:{args.model}",
                "relevance": verdict["relevance"], "reason": verdict["reason"],
            })
    print(f"Saved automatic judgments to {args.output}")


if __name__ == "__main__":
    main()
