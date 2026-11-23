"""Compare criterion-level scoring configurations on the validation queries.

Uses the automatic LLM judgments cache (scripts.llm_judge) and judges any newly
retrieved film on the fly, so that no unjudged film is counted as irrelevant.
Only the validation split is used, to keep the test split for the final report.

    python -m scripts.tune_scoring                 # genre scoring (second selection)
    python -m scripts.tune_scoring --grid story    # story scope and overall share (first selection)
"""

import argparse
import json
import statistics
from pathlib import Path

import httpx

from backend.catalog import load_catalog
from backend.config import CACHE_PATH, CORPUS_PATH, MODEL_PATH, ROOT
from backend.embeddings import SentenceEncoder
from backend.recommender import Recommender
from backend.schemas import Preferences
from scripts.evaluate import ndcg_at_3, precision_at_3, query_fingerprint, read_queries
from scripts.llm_judge import judge

# Each configuration lists the Recommender arguments that differ from the original scorer.
ORIGINAL_GENRES = {"genre_match": "semantic", "genre_weight": 0.05}
GRIDS = {
    # First selection: what the story text is compared with, and an extra overall criterion.
    "story": [
        {"story_scope": scope, "overall_weight": overall, **ORIGINAL_GENRES}
        for scope in ("overview", "profile") for overall in (0.0, 0.25, 0.5)
    ],
    # Second selection, with the first choice fixed: semantic genre names against exact
    # genre coverage, and the weight of the genre criterion.
    "genres": [
        {"story_scope": "profile", "overall_weight": 0.0, **ORIGINAL_GENRES},
        *({"story_scope": "profile", "overall_weight": 0.0, "genre_match": "coverage", "genre_weight": weight}
          for weight in (0.05, 0.15, 0.25)),
    ],
}
OUTPUTS = {"story": "evaluation/scoring-selection.json", "genres": "evaluation/scoring-selection-genres.json"}


def label(configuration: dict) -> str:
    return ",".join(f"{key}={value}" for key, value in configuration.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--ollama", default="http://localhost:11434")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--cache", type=Path, default=ROOT / "evaluation/llm-judgments-cache.json")
    parser.add_argument("--grid", default="genres", choices=sorted(GRIDS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    configurations = GRIDS[args.grid]
    output = args.output or ROOT / OUTPUTS[args.grid]

    movies = load_catalog(CORPUS_PATH)
    by_id = {movie.tmdb_id: movie for movie in movies}
    encoder = SentenceEncoder(MODEL_PATH, CACHE_PATH)
    queries = [query for query in read_queries(ROOT / "evaluation/queries.json") if query["split"] == args.split]
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    rankings = {}
    with httpx.Client(timeout=180) as client:
        # First retrieve and judge every configuration, so all of them share the same judged pool.
        for configuration in configurations:
            engine = Recommender(movies, encoder, **configuration)
            for query in queries:
                profile = Preferences.model_validate(query["preferences"])
                ids = [item.movie.tmdb_id for item in engine.recommend(profile).recommendations]
                rankings[(label(configuration), query["query_id"])] = ids
                prefix = f"{args.model}|{query_fingerprint(query)}|"
                for movie_id in ids:
                    if prefix + str(movie_id) not in cache:
                        cache[prefix + str(movie_id)] = judge(client, args.ollama, args.model,
                                                              query["preferences"], by_id[movie_id])
    rows = []
    for configuration in configurations:
        precisions, ndcgs = [], []
        for query in queries:
            prefix = f"{args.model}|{query_fingerprint(query)}|"
            judged = {int(key.rsplit("|", 1)[1]): value["relevance"]
                      for key, value in cache.items() if key.startswith(prefix)}
            ids = rankings[(label(configuration), query["query_id"])]
            precisions.append(precision_at_3(ids, judged))
            ndcgs.append(ndcg_at_3(ids, judged))
        rows.append({**configuration,
                     "precision_at_3": round(statistics.mean(precisions), 4),
                     "ndcg_at_3": round(statistics.mean(ndcgs), 4)})
        print(rows[-1])
    args.cache.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")
    output.write_text(json.dumps({
        "grid": args.grid, "split": args.split, "judge": f"llm:{args.model}", "method": "sbert_criteria",
        "note": "Automatic LLM judgments, not human annotations. nDCG uses the judged pool.",
        "configurations": rows,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
