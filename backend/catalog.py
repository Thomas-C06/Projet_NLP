"""Read the reference films and keep missing descriptions explicit."""

import hashlib
import json
from pathlib import Path

from pydantic import TypeAdapter

from backend.config import RELEASE_PERIODS
from backend.schemas import Movie


def load_catalog(path: Path) -> list[Movie]:
    if not path.exists():
        return []
    movies = TypeAdapter(list[Movie]).validate_json(path.read_text(encoding="utf-8-sig"))
    ids = [movie.tmdb_id for movie in movies]
    if len(ids) != len(set(ids)):
        raise ValueError("The corpus contains duplicate TMDB identifiers.")
    return movies


def fingerprint(movies: list[Movie]) -> str:
    # Include every field so that an annotation change invalidates cached vectors.
    payload = json.dumps([movie.model_dump() for movie in movies], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def movie_texts(movie: Movie) -> dict[str, str]:
    return {
        "description": movie.overview or "",
        "themes": "; ".join(movie.themes or []),
        "mood": movie.mood or "",
        "narrative_style": movie.narrative_style or "",
        "setting": movie.setting or "",
        "genres": ", ".join(movie.genres),
    }


def eligible(movie: Movie, profile) -> bool:
    if movie.tmdb_id in profile.seen_movie_ids:
        return False
    if {genre.casefold() for genre in movie.genres} & {
        genre.casefold() for genre in profile.excluded_genres
    }:
        return False
    if profile.max_runtime is not None:
        if movie.runtime_minutes is None or movie.runtime_minutes > profile.max_runtime:
            return False
    # An unknown release year cannot prove that a film belongs to the chosen period.
    low, high = RELEASE_PERIODS[getattr(profile, "release_period", "any")]
    if low is not None or high is not None:
        year = movie.release_year
        if year is None or (low is not None and year < low) or (high is not None and year > high):
            return False
    # With a strict theme exclusion, unknown annotations cannot prove compliance.
    if profile.excluded_themes:
        if movie.themes is None:
            return False
        if {theme.casefold() for theme in movie.themes} & {
            theme.casefold() for theme in profile.excluded_themes
        }:
            return False
    return True
