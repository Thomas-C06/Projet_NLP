"""Collect English TMDB metadata. Unknown manual annotations stay null."""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from backend.config import ROOT
from backend.schemas import Movie


def get_json(client: httpx.Client, path: str, params: dict | None = None):
    for attempt in range(4):
        response = client.get(path, params={"language": "en-US", **(params or {})})
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(min(2 ** attempt, 8))
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("TMDB is temporarily unavailable. Please run the collection again later.")


def collect(limit: int, token: str, output, overwrite: bool = False):
    if output.exists() and not overwrite:
        raise ValueError("The output already exists. Use --overwrite only if its annotations can be replaced.")
    headers = {"Authorization": f"Bearer {token}", "accept": "application/json"}
    with httpx.Client(base_url="https://api.themoviedb.org/3/", headers=headers, timeout=30) as client:
        buckets = []
        # Use genre and decade buckets to avoid selecting only recent popular films.
        for decade in range(1970, 2030, 10):
            for genre in (28, 12, 16, 35, 80, 18, 14, 27, 10749, 878):
                data = get_json(client, "discover/movie", {
                    "with_genres": genre, "primary_release_date.gte": f"{decade}-01-01",
                    "primary_release_date.lte": f"{decade + 9}-12-31",
                    "vote_count.gte": 100, "include_adult": "false", "sort_by": "vote_count.desc",
                })
                buckets.append(data.get("results", []))
        ids = []
        for position in range(20):
            for bucket in buckets:
                if position < len(bucket) and bucket[position]["id"] not in ids:
                    ids.append(bucket[position]["id"])
        movies, raw = [], []
        for movie_id in ids:
            details = get_json(client, f"movie/{movie_id}", {"append_to_response": "keywords"})
            if not details.get("overview") or not details.get("genres"):
                continue
            raw.append(details)
            date = details.get("release_date", "")
            movies.append(Movie(
                tmdb_id=movie_id, title=details["title"], overview=details["overview"],
                genres=[genre["name"] for genre in details["genres"]],
                release_year=int(date[:4]) if date else None,
                runtime_minutes=details.get("runtime") or None,
                poster_path=details.get("poster_path"),
                source=f"https://www.themoviedb.org/movie/{movie_id}",
            ))
            print(f"Collected {len(movies)}/{limit}: {details['title']}")
            if len(movies) == limit:
                break
    if len(movies) != limit:
        raise ValueError(f"Only {len(movies)} usable films were found. The existing corpus was not changed.")
    raw_path = ROOT / "data/raw/tmdb_metadata.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([movie.model_dump() for movie in movies], indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "collected_at": datetime.now(timezone.utc).isoformat(), "language": "en-US",
        "size": len(movies), "source": "TMDB", "annotation_status": "pending manual review",
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=150, choices=range(1, 501), metavar="1-500")
    parser.add_argument("--output", type=Path, default=ROOT / "data/movies.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    token = os.getenv("TMDB_API_TOKEN", "").strip()
    if not token:
        parser.exit(1, "Add TMDB_API_TOKEN to your local .env file before collection.\n")
    try:
        collect(args.limit, token, args.output, args.overwrite)
    except (ValueError, RuntimeError, httpx.HTTPError) as error:
        parser.exit(1, f"Collection failed: {error}\n")


if __name__ == "__main__":
    main()
