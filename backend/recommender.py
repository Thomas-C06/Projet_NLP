"""Compare lexical, global semantic and criterion-level recommendations."""

import time
import re

import numpy as np
from langdetect import DetectorFactory, LangDetectException, detect_langs
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.catalog import eligible, fingerprint, movie_texts
from backend.config import LABELS, OVERALL_WEIGHT, PERIOD_LABELS, STORY_SCOPE, WEIGHTS
from backend.schemas import CriterionScore, Preferences, Recommendation, RecommendationResponse

DetectorFactory.seed = 0


def check_english(profile: Preferences) -> None:
    text = " ".join(profile.criterion_texts().values())
    try:
        languages = detect_langs(text)
    except LangDetectException:
        raise ValueError("Please provide a description in English.") from None
    if languages and languages[0].lang != "en" and languages[0].prob > 0.90:
        raise ValueError("Only English requests are supported. Please write your preferences in English.")


def summarise(profile: Preferences) -> str:
    parts = [f"You are looking for: {profile.description}"]
    for key, text in profile.criterion_texts().items():
        if key != "description" and text:
            parts.append(f"{LABELS[key]}: {text}.")
    if profile.max_runtime is not None:
        parts.append(f"Maximum duration: {profile.max_runtime} minutes.")
    if profile.release_period != "any":
        parts.append(f"Only films {PERIOD_LABELS[profile.release_period]}.")
    if profile.excluded_genres:
        parts.append(f"Excluded genres: {', '.join(profile.excluded_genres)}.")
    if profile.excluded_themes:
        parts.append(f"Excluded annotated topics: {', '.join(profile.excluded_themes)}.")
    if profile.seen_movie_ids:
        parts.append(f"Already watched films excluded: {len(set(profile.seen_movie_ids))}.")
    return " ".join(parts)


class Recommender:
    def __init__(self, movies, encoder=None, story_scope: str = STORY_SCOPE, overall_weight: float = OVERALL_WEIGHT):
        if story_scope not in ("overview", "profile") or not 0 <= overall_weight < 1:
            raise ValueError("Invalid scoring configuration.")
        self.story_scope = story_scope
        self.overall_weight = overall_weight
        self.movies = movies
        self.encoder = encoder
        self.fingerprint = fingerprint(movies)
        self.texts = [movie_texts(movie) for movie in movies]
        self.global_texts = [". ".join(text for text in fields.values() if text) for fields in self.texts]
        # Text compared with each user criterion. The free description can target the whole film profile.
        self.compared = [dict(fields) for fields in self.texts]
        if story_scope == "profile":
            for fields, overall in zip(self.compared, self.global_texts):
                fields["description"] = overall
        self.vectorizer = None
        self.tfidf = None
        self.semantic_fields = None
        self.semantic_global = None

    def check_exclusions(self, profile: Preferences) -> None:
        # Known negative topics belong to explicit filters, rather than positive vectors.
        text = " ".join(profile.criterion_texts().values())
        tags = {genre for movie in self.movies for genre in movie.genres}
        tags.update(theme for movie in self.movies for theme in movie.themes or [])
        for tag in tags:
            pattern = r"\b(?:no|without|avoid|excluding|do not want|don't want)\s+(?:any\s+)?" + re.escape(tag) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                raise ValueError("Please describe positive preferences in the text fields and move exclusions to the genre or topic filters.")
        positive_themes = profile.themes.casefold()
        if any(re.search(r"\b" + re.escape(theme.casefold()) + r"\b", positive_themes)
               for theme in profile.excluded_themes):
            raise ValueError("A topic is both requested and excluded. Please clarify your preferences.")

    def _global_similarities(self, query: str, method: str) -> np.ndarray:
        if method == "tfidf":
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")
                self.tfidf = self.vectorizer.fit_transform(self.global_texts)
            return (self.tfidf @ self.vectorizer.transform([query]).T).toarray().ravel()
        if self.encoder is None:
            raise RuntimeError("The semantic encoder is not configured.")
        if self.semantic_global is None:
            self.semantic_global = self.encoder.encode_corpus(self.global_texts)
        return self.semantic_global @ self.encoder.encode([query])[0]

    def _criterion_similarities(self, active: dict[str, str]) -> dict[str, np.ndarray]:
        if self.encoder is None:
            raise RuntimeError("The semantic encoder is not configured.")
        if self.semantic_fields is None:
            flat = [fields[key] for fields in self.compared for key in WEIGHTS]
            vectors = self.encoder.encode_corpus(flat)
            self.semantic_fields = vectors.reshape(len(self.movies), len(WEIGHTS), -1)
        queries = self.encoder.encode(list(active.values()))
        return {
            key: self.semantic_fields[:, list(WEIGHTS).index(key), :] @ queries[index]
            for index, key in enumerate(active)
        }

    def recommend(self, profile: Preferences, top_k: int = 3, validate_language: bool = True):
        start = time.perf_counter()
        if validate_language:
            check_english(profile)
        self.check_exclusions(profile)
        active = {key: value for key, value in profile.criterion_texts().items() if value}
        candidates = [index for index, movie in enumerate(self.movies) if eligible(movie, profile)]
        warnings = []
        recommendations = []
        if candidates:
            if profile.method == "sbert_criteria":
                similarities = self._criterion_similarities(active)
                global_scores = None
                if self.overall_weight:
                    overall_scores = self._global_similarities(". ".join(active.values()), "sbert_global")
            else:
                query = ". ".join(active.values())
                global_scores = self._global_similarities(query, profile.method)
                similarities = {}
            importance = profile.importance.model_dump()
            for index in candidates:
                available = {key: text for key, text in active.items() if self.compared[index][key]}
                missing = [LABELS[key] for key in active if key not in available]
                if profile.method == "sbert_criteria":
                    denominator = sum(WEIGHTS[key] * importance[key] for key in available)
                    if not denominator:
                        continue
                    details = []
                    for key in available:
                        similarity = float(np.clip(similarities[key][index], 0, 1))
                        weight = (1 - self.overall_weight) * WEIGHTS[key] * importance[key] / denominator
                        details.append(CriterionScore(
                            criterion=key, label=LABELS[key], similarity=similarity,
                            weight=weight, contribution=100 * similarity * weight,
                            evidence=self.compared[index][key],
                        ))
                    if self.overall_weight:
                        similarity = float(np.clip(overall_scores[index], 0, 1))
                        details.append(CriterionScore(
                            criterion="overall", label="Overall profile", similarity=similarity,
                            weight=self.overall_weight, contribution=100 * similarity * self.overall_weight,
                            evidence=self.global_texts[index],
                        ))
                    affinity = sum(item.contribution for item in details)
                else:
                    affinity = 100 * float(np.clip(global_scores[index], 0, 1))
                    details = [CriterionScore(
                        criterion="global", label="Overall text", similarity=affinity / 100,
                        weight=1, contribution=affinity, evidence=self.global_texts[index],
                    )]
                    missing = []
                leading = sorted(details, key=lambda item: -item.contribution)
                positive = [item.label for item in leading if item.contribution > 0][:2]
                if positive:
                    explanation = "Main matching criteria: " + " and ".join(positive) + "."
                else:
                    explanation = "No positive textual match was found for this film."
                if profile.method != "sbert_criteria":
                    explanation += " This comparison method uses one overall text score."
                if missing:
                    explanation += " Unavailable descriptions: " + ", ".join(missing) + "."
                recommendations.append(Recommendation(
                    movie=self.movies[index], affinity=affinity, criteria=details,
                    missing_criteria=missing, main_contributions=positive, explanation=explanation,
                ))
        recommendations.sort(key=lambda item: (-item.affinity, item.movie.tmdb_id))
        selected = recommendations[:top_k]
        if len(selected) < top_k:
            warnings.append(f"Only {len(selected)} films could be recommended. No strict filter was relaxed.")
        if any(item.missing_criteria for item in selected):
            warnings.append("Some film descriptions are missing. Their scores use only the available active criteria.")
        if selected and max(item.affinity for item in selected) == 0:
            warnings.append("These films have no positive match. Try a more informative description.")
        if profile.excluded_themes:
            warnings.append("Topic exclusions use the reviewed theme tags; they cannot guarantee the absence of every scene.")
        return RecommendationResponse(
            profile_summary=summarise(profile), recommendations=selected,
            eligible_count=len(candidates), catalog_size=len(self.movies), method=profile.method,
            warnings=warnings, latency_ms=round((time.perf_counter() - start) * 1000, 2),
            corpus_fingerprint=self.fingerprint,
        )
