"""Keep the project paths and scoring choices in one place."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def project_path(variable: str, default: str) -> Path:
    path = Path(os.getenv(variable, default))
    return path if path.is_absolute() else ROOT / path


CORPUS_PATH = project_path("MOVIE_CORPUS", "data/movies.json")
MODEL_PATH = project_path("MODEL_PATH", "models/all-MiniLM-L6-v2")
CACHE_PATH = project_path("EMBEDDING_CACHE", "data/cache")
RESPONSES_PATH = project_path("RESPONSES_PATH", "data/responses/responses.jsonl")
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# Optional generative explanations (bonus). The ranking never uses the LLM.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
CRITIC_MODEL = os.getenv("CRITIC_MODEL", "qwen3:8b")
TARGET_SIZE = 150
WEIGHTS = {
    "description": 0.35,
    "themes": 0.25,
    "mood": 0.15,
    "narrative_style": 0.10,
    "setting": 0.10,
    "genres": 0.05,
}
# Release periods offered as a single-choice question. Bounds are inclusive.
RELEASE_PERIODS = {
    "any": (None, None),
    "before-1980": (None, 1979),
    "1980-1999": (1980, 1999),
    "2000-later": (2000, None),
}
PERIOD_LABELS = {
    "before-1980": "released before 1980",
    "1980-1999": "released between 1980 and 1999",
    "2000-later": "released in 2000 or later",
}
# Selected on the validation queries (see docs/report.md). STORY_SCOPE chooses the
# film text compared with the free description: "overview" (synopsis only) or
# "profile" (synopsis plus every annotated attribute). OVERALL_WEIGHT is the share
# of the affinity given to one extra "Overall profile" criterion.
STORY_SCOPE = os.getenv("STORY_SCOPE", "profile")
OVERALL_WEIGHT = float(os.getenv("OVERALL_WEIGHT", "0"))
# How preferred genres are scored. "semantic" compares the genre names with the
# encoder. "coverage" uses the exact TMDB categories: the share of the preferred
# genres that the film has. GENRE_WEIGHT replaces the base weight of the genre
# criterion. scripts.tune_scoring compared them on the validation queries
# (evaluation/scoring-selection-genres.json): coverage did not improve Precision@3
# and lowered nDCG@3, so the original semantic scoring stays the default.
GENRE_MATCH = os.getenv("GENRE_MATCH", "semantic")
GENRE_WEIGHT = float(os.getenv("GENRE_WEIGHT", "0.05"))
LABELS = {
    "description": "Story",
    "themes": "Themes",
    "mood": "Atmosphere",
    "narrative_style": "Storytelling",
    "setting": "Setting",
    "genres": "Genres",
}
