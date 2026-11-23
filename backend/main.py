"""Local REST API. Requests are saved on the server only when the user opts in."""

import json
import os
import threading
from datetime import datetime, timezone
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import httpx

from backend import critic
from backend.catalog import load_catalog
from backend.config import CACHE_PATH, CORPUS_PATH, MODEL_PATH, RESPONSES_PATH, ROOT, TARGET_SIZE, WEIGHTS
from backend.embeddings import SentenceEncoder
from backend.recommender import Recommender
from backend.schemas import CriticRequest, Preferences, RecommendationResponse

app = FastAPI(title="Movie Match API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"), "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
)


_responses_lock = threading.Lock()


def save_response(profile: Preferences, result: RecommendationResponse) -> None:
    """Append one questionnaire and its Top 3 to a local JSON Lines file."""
    record = {
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "answers": profile.model_dump(exclude={"save_response"}),
        "recommendations": [
            {"tmdb_id": item.movie.tmdb_id, "title": item.movie.title, "affinity": round(item.affinity, 2)}
            for item in result.recommendations
        ],
        "method": result.method,
        "corpus_fingerprint": result.corpus_fingerprint,
    }
    with _responses_lock:
        RESPONSES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RESPONSES_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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
        "model_available": MODEL_PATH.exists(), "weights": WEIGHTS,
        "corpus_ready": len(engine.movies) == TARGET_SIZE,
        "annotated_count": sum(all(movie.model_dump().get(key) for key in
            ("overview", "themes", "mood", "narrative_style", "setting", "annotation_version"))
            for movie in engine.movies),
        "critic_available": critic.available(),
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
        result = engine.recommend(profile)
        if profile.save_response:
            save_response(profile, result)
            result.saved = True
        return result
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(422, str(error)) from None
    except (OSError, RuntimeError) as error:
        raise HTTPException(503, str(error)) from None


@app.post("/api/critic")
def critic_pitch(request: CriticRequest):
    try:
        movies = {movie.tmdb_id: movie for movie in get_engine().movies}
    except (ValueError, OSError):
        raise HTTPException(503, "The corpus is invalid. Check its schema.") from None
    if request.tmdb_id not in movies:
        raise HTTPException(404, "Unknown film.")
    try:
        return {"text": critic.pitch(request.profile_summary, movies[request.tmdb_id], request.main_contributions),
                "model": critic.CRITIC_MODEL}
    except (httpx.HTTPError, KeyError, ValueError):
        raise HTTPException(503, "The local language model is not available. Start Ollama to use this optional feature.") from None


# Serve the built interface on the same local address as the API.
if (ROOT / "frontend/dist/index.html").exists():
    app.mount("/", StaticFiles(directory=ROOT / "frontend/dist", html=True), name="frontend")
