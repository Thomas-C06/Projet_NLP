"""Build the local 150-film corpus from the public TMDB metadata snapshot."""

import csv
import hashlib
import io
import json
import re
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import httpx
import numpy as np

from backend.catalog import fingerprint
from backend.config import CACHE_PATH, MODEL_PATH, ROOT, TARGET_SIZE
from backend.embeddings import SentenceEncoder
from backend.schemas import Movie

SOURCE_PAGE = "https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata"
DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/tmdb/tmdb-movie-metadata"
SNAPSHOT_HASH = "0342a9748128f44e6a58e96af19d918478af73a8976b142aa4e7e0aa3c50ef1b"
ANNOTATION_VERSION = "source-semantic-v1"

MOODS = [
    "A warm, hopeful and uplifting atmosphere.",
    "A funny, playful and light-hearted atmosphere.",
    "A tense, suspenseful and dangerous atmosphere.",
    "A dark, frightening and disturbing atmosphere.",
    "A thoughtful, contemplative and philosophical atmosphere.",
    "A moving, emotional and bittersweet atmosphere.",
    "An adventurous, exciting atmosphere with a sense of discovery.",
    "A romantic, intimate and affectionate atmosphere.",
    "A tragic, sorrowful atmosphere shaped by loss and grief.",
    "A magical, imaginative and dreamlike atmosphere.",
]
STYLES = [
    "An adventure narrative built around a journey, obstacles and discovery.",
    "An action narrative driven by pursuit, conflict and physical danger.",
    "An investigation narrative built around a mystery, clues and revelations.",
    "A character-focused narrative about personal relationships and emotional change.",
    "A comic narrative built around social situations, mistakes and misunderstandings.",
    "A romantic narrative focused on attraction, relationships and emotional choices.",
    "A survival narrative about people facing isolation and a hostile environment.",
    "A historical narrative following people caught in political or military conflict.",
    "A coming-of-age narrative about youth, identity and growing independence.",
    "A psychological narrative exploring memory, perception and uncertain identity.",
]
SETTING_PATTERN = re.compile(
    r"\b(?:\d{2}(?:th|st|nd|rd) century|\d{4}s?|new york|los angeles|san francisco|"
    r"london|paris|tokyo|chicago|boston|las vegas|washington|rome|berlin|moscow|"
    r"africa|australia|india|china|japan|england|scotland|ireland|france|germany|"
    r"vietnam|iraq|afghanistan|middle east|ancient|medieval|future|dystopi\w*|"
    r"outer space|space travel|spaceship|spacecraft|planet|island|desert|jungle|"
    r"small town|village|countryside|rural|prison|high school|boarding school|"
    r"university|hospital|underwater|ocean|wild west|world war|civil war)\b", re.I
)
NON_THEME_TAGS = {"duringcreditsstinger", "aftercreditsstinger", "woman director", "3d", "imax", "based on novel", "based on comic", "based on play or musical"}


def read_snapshot(path: Path) -> list[dict]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SNAPSHOT_HASH:
        raise ValueError("The public archive changed. Check its provenance before replacing the frozen snapshot.")
    with ZipFile(io.BytesIO(payload)) as archive:
        text = archive.read("tmdb_5000_movies.csv").decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def select_films(rows: list[dict], count: int = TARGET_SIZE) -> list[dict]:
    usable = []
    for row in rows:
        genres, keywords = json.loads(row["genres"]), json.loads(row["keywords"])
        if len(row["overview"].split()) < 20 or not genres or not keywords:
            continue
        if not row["release_date"] or float(row["runtime"] or 0) <= 0:
            continue
        if int(row["release_date"][:4]) > 2016 or int(row["vote_count"] or 0) < 100:
            continue
        usable.append(row)
    usable.sort(key=lambda row: (-int(row["vote_count"]), int(row["id"])))
    buckets = defaultdict(list)
    for row in usable:
        decade = int(row["release_date"][:4]) // 10 * 10
        for genre in json.loads(row["genres"]):
            buckets[(genre["name"], decade)].append(row)
    # Keep some well documented reference films, then balance genres and decades.
    anchors = {152601, 157336, 244786, 129, 27205, 264660, 77338, 194, 13, 550, 120, 105, 862}
    anchor_rows = [row for row in usable if int(row["id"]) in anchors][:count]
    selected = {int(row["id"]): row for row in anchor_rows}
    if len(selected) == count:
        return list(selected.values())
    for position in range(len(usable)):
        for key in sorted(buckets):
            bucket = buckets[key]
            if position < len(bucket):
                row = bucket[position]
                selected.setdefault(int(row["id"]), row)
                if len(selected) == count:
                    return list(selected.values())
    raise ValueError(f"Not enough usable films for the requested corpus of {count}.")


def setting_from_source(row: dict, keywords: list[str]) -> tuple[str | None, list[str]]:
    cues = [keyword for keyword in keywords if SETTING_PATTERN.search(keyword)]
    if cues:
        return "Setting cues in the source keywords: " + "; ".join(cues[:6]) + ".", cues[:6]
    sentences = re.split(r"(?<=[.!?])\s+", row["overview"])
    for sentence in sentences:
        if SETTING_PATTERN.search(sentence):
            return sentence, [sentence]
    return None, []


def build_catalog(rows: list[dict], encoder: SentenceEncoder):
    selected = select_films(rows)
    source_texts = [row["overview"] + " Keywords: " + "; ".join(
        item["name"] for item in json.loads(row["keywords"])) for row in selected]
    vectors = encoder.encode(source_texts)
    mood_scores = vectors @ encoder.encode(MOODS).T
    style_scores = vectors @ encoder.encode(STYLES).T
    movies, evidence = [], []
    for index, row in enumerate(selected):
        keywords = list(dict.fromkeys(item["name"] for item in json.loads(row["keywords"])))
        themes = [keyword for keyword in keywords if keyword.casefold() not in NON_THEME_TAGS][:12]
        setting, setting_evidence = setting_from_source(row, keywords)
        mood_index, style_index = int(np.argmax(mood_scores[index])), int(np.argmax(style_scores[index]))
        # These labels are estimates from fixed descriptions, not factual manual annotations.
        mood = MOODS[mood_index] if mood_scores[index, mood_index] >= 0.15 else None
        style = STYLES[style_index] if style_scores[index, style_index] >= 0.15 else None
        movie = Movie(
            tmdb_id=int(row["id"]), title=row["title"], overview=row["overview"],
            genres=[item["name"] for item in json.loads(row["genres"])], themes=themes or None,
            mood=mood, narrative_style=style, setting=setting,
            release_year=int(row["release_date"][:4]), runtime_minutes=int(float(row["runtime"])),
            poster_path=None, source=f"https://www.themoviedb.org/movie/{row['id']}",
            annotation_version=ANNOTATION_VERSION,
        )
        movies.append(movie)
        evidence.append({
            "tmdb_id": movie.tmdb_id, "source_keywords": keywords,
            "mood_method": "nearest fixed English description, cosine similarity",
            "mood_similarity": float(mood_scores[index, mood_index]),
            "style_method": "nearest fixed English description, cosine similarity",
            "style_similarity": float(style_scores[index, style_index]),
            "setting_evidence": setting_evidence, "human_reviewed": False,
        })
    return movies, evidence


def main():
    archive = ROOT / "data/raw/tmdb-movie-metadata.zip"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        with httpx.Client(follow_redirects=True, timeout=120) as client:
            response = client.get(DOWNLOAD)
            response.raise_for_status()
        archive.write_bytes(response.content)
    output = ROOT / "data/movies.json"
    if output.exists():
        raise SystemExit("The corpus already exists. It was kept unchanged.")
    movies, evidence = build_catalog(read_snapshot(archive), SentenceEncoder(MODEL_PATH, CACHE_PATH))
    output.write_text(json.dumps([movie.model_dump() for movie in movies], indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "data/annotation-evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    manifest = {
        "size": len(movies), "source": SOURCE_PAGE, "download": DOWNLOAD,
        "archive_sha256": SNAPSHOT_HASH, "corpus_fingerprint": fingerprint(movies),
        "snapshot_period": "TMDB 5000 snapshot, through 2016", "language": "en",
        "annotation_version": ANNOTATION_VERSION,
        "annotation_method": "source keywords, extractive setting cues and estimated fixed semantic labels",
        "human_cross_review": False,
        "missing_values": "null when no source cue or sufficient label similarity is available",
        "license": "TMDB terms, non-commercial educational use with attribution; not CC0",
    }
    (ROOT / "data/movies.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(movies)} real films from the public TMDB snapshot.")
    print(f"Source descriptions for setting: {sum(movie.setting is not None for movie in movies)}")
    print(f"Estimated mood/style labels: {sum(movie.mood is not None and movie.narrative_style is not None for movie in movies)}")


if __name__ == "__main__":
    main()
