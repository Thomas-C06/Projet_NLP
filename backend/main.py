"""Local REST API. Recommendation requests are not saved on the server."""

import os
import hashlib
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.catalog import load_catalog
from backend.config import CACHE_PATH, CORPUS_PATH, MODEL_PATH, ROOT, TARGET_SIZE, WEIGHTS
from backend.embeddings import SentenceEncoder
from backend.recommender import Recommender
from backend.schemas import Preferences, RecommendationResponse

app = FastAPI(title="Movie Match API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"), "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
)


@lru_cache(maxsize=1)
def get_engine() -> Recommender:
    movies = load_catalog(CORPUS_PATH)
    return Recommender(movies, SentenceEncoder(
        MODEL_PATH, CACHE_PATH, os.getenv("ALLOW_MODEL_DOWNLOAD", "false").lower() == "true"
    ))


@app.get("/api/health")
def health():
    try:
        engine = get_engine()
    except (ValueError, OSError):
        raise HTTPException(503, "The corpus is invalid. Run the corpus validation command.") from None
    return {
        "status": "ok", "catalog_size": len(engine.movies), "target_size": TARGET_SIZE,
        "application": "movie-match",
        "server_pid": os.getpid(),
        "instance_id": hashlib.sha256(str(ROOT).encode()).hexdigest()[:16],
        "model_available": MODEL_PATH.exists(), "weights": WEIGHTS,
        "corpus_ready": len(engine.movies) == TARGET_SIZE,
        "annotated_count": sum(all(movie.model_dump().get(key) for key in
            ("overview", "themes", "mood", "narrative_style", "setting", "annotation_version"))
            for movie in engine.movies),
        "estimated_annotations": sum((movie.annotation_version or "").startswith("source-semantic") for movie in engine.movies),
    }


@app.get("/api/catalog")
def catalog():
    try:
        movies = get_engine().movies
    except (ValueError, OSError):
        raise HTTPException(503, "The corpus is invalid. Check its schema.") from None
    return {
        "movies": [{"tmdb_id": movie.tmdb_id, "title": movie.title} for movie in movies],
        "genres": sorted({genre for movie in movies for genre in movie.genres}),
        "themes": sorted({theme for movie in movies for theme in movie.themes or []}),
    }


@app.post("/api/recommend", response_model=RecommendationResponse)
def recommend(profile: Preferences):
    try:
        engine = get_engine()
        if not engine.movies:
            raise HTTPException(503, "The film corpus is not available yet. Collect and validate the TMDB films first.")
        known_genres = {genre.casefold() for movie in engine.movies for genre in movie.genres}
        requested = profile.preferred_genres + profile.excluded_genres
        if any(genre.casefold() not in known_genres for genre in requested):
            raise ValueError("Please select genres from the available catalog.")
        known_themes = {theme.casefold() for movie in engine.movies for theme in movie.themes or []}
        if any(theme.casefold() not in known_themes for theme in profile.excluded_themes):
            raise ValueError("Please select excluded topics from the annotated catalog.")
        return engine.recommend(profile)
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(422, str(error)) from None
    except (OSError, RuntimeError) as error:
        raise HTTPException(503, str(error)) from None


# Serve the built interface on the same local address as the API.
if (ROOT / "frontend/dist/index.html").exists():
    app.mount("/", StaticFiles(directory=ROOT / "frontend/dist", html=True), name="frontend")
