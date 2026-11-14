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
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
TARGET_SIZE = 150
WEIGHTS = {
    "description": 0.35,
    "themes": 0.25,
    "mood": 0.15,
    "narrative_style": 0.10,
    "setting": 0.10,
    "genres": 0.05,
}
LABELS = {
    "description": "Story",
    "themes": "Themes",
    "mood": "Atmosphere",
    "narrative_style": "Storytelling",
    "setting": "Setting",
    "genres": "Genres",
}
