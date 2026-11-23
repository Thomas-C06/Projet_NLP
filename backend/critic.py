"""Optional generative explanation with a local LLM served by Ollama (bonus feature).

The ranking never depends on this module. The model only rephrases facts that the
semantic scorer already computed: the user's summary, the film synopsis and the
criteria that matched. If Ollama is not running, the feature is simply hidden.
"""

import re

import httpx

from backend.config import CRITIC_MODEL, OLLAMA_URL
from backend.schemas import Movie

PROMPT = """You are a friendly film critic. In two or three sentences of plain English, tell the
viewer why this film fits their request. Use only the information below; do not invent
plot details, actors or awards. Do not mention scores or numbers.

Viewer request: {summary}
Film: {title} ({year}), {genres}
Synopsis: {overview}
Strongest semantic matches: {matches}"""


def available() -> bool:
    try:
        tags = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=0.5).json()
    except (httpx.HTTPError, ValueError):
        return False
    names = {model.get("name") for model in tags.get("models", [])}
    return CRITIC_MODEL in names or f"{CRITIC_MODEL}:latest" in names


def pitch(summary: str, movie: Movie, matches: list[str]) -> str:
    prompt = PROMPT.format(
        summary=summary, title=movie.title, year=movie.release_year or "unknown year",
        genres=", ".join(movie.genres), overview=movie.overview or "unavailable",
        matches=", ".join(matches) or "none",
    )
    response = httpx.post(f"{OLLAMA_URL}/api/generate", timeout=120, json={
        "model": CRITIC_MODEL, "prompt": prompt, "stream": False, "think": False,
        "options": {"temperature": 0.4, "num_predict": 160},
    })
    response.raise_for_status()
    # The interface shows plain text, so remove Markdown emphasis the model may add.
    return re.sub(r"[*_]{1,2}([^*_]+)[*_]{1,2}", r"\1", response.json()["response"]).strip()
