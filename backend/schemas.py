"""Validate the corpus, questionnaire and API responses."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Criterion = Literal["description", "themes", "mood", "narrative_style", "setting", "genres"]
Method = Literal["tfidf", "sbert_global", "sbert_criteria"]
ReleasePeriod = Literal["any", "before-1980", "1980-1999", "2000-later"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class Movie(StrictModel):
    tmdb_id: int = Field(gt=0)
    title: str = Field(min_length=1)
    overview: str | None = None
    genres: list[str] = Field(min_length=1)
    themes: list[str] | None = None
    mood: str | None = None
    narrative_style: str | None = None
    setting: str | None = None
    release_year: int | None = Field(default=None, ge=1870, le=2100)
    runtime_minutes: int | None = Field(default=None, gt=0)
    poster_path: str | None = None
    source: str = Field(min_length=1)
    annotation_version: str | None = None

    @field_validator("genres", "themes")
    @classmethod
    def clean_tags(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        cleaned = list(dict.fromkeys(value.strip() for value in values if value.strip()))
        if not cleaned:
            raise ValueError("Use null for unknown themes and at least one genre for each film.")
        return cleaned

    @field_validator("overview", "mood", "narrative_style", "setting", "annotation_version")
    @classmethod
    def explicit_missing(cls, value: str | None) -> str | None:
        return value or None

    @field_validator("poster_path")
    @classmethod
    def valid_poster(cls, value: str | None) -> str | None:
        if value and (not value.startswith("/") or ".." in value or ":" in value):
            raise ValueError("The poster path must be a TMDB relative path.")
        return value


class Importance(StrictModel):
    description: int = Field(default=3, ge=1, le=5)
    themes: int = Field(default=3, ge=1, le=5)
    mood: int = Field(default=3, ge=1, le=5)
    narrative_style: int = Field(default=3, ge=1, le=5)
    setting: int = Field(default=3, ge=1, le=5)
    genres: int = Field(default=3, ge=1, le=5)


class Preferences(StrictModel):
    description: str = Field(min_length=12, max_length=4000)
    themes: str = Field(default="", max_length=1500)
    mood: str = Field(default="", max_length=1000)
    narrative_style: str = Field(default="", max_length=1000)
    setting: str = Field(default="", max_length=1000)
    preferred_genres: list[str] = Field(default_factory=list, max_length=30)
    importance: Importance = Field(default_factory=Importance)
    max_runtime: int | None = Field(default=None, ge=1, le=600)
    excluded_genres: list[str] = Field(default_factory=list, max_length=30)
    excluded_themes: list[str] = Field(default_factory=list, max_length=100)
    seen_movie_ids: list[int] = Field(default_factory=list, max_length=1000)
    release_period: ReleasePeriod = "any"
    method: Method = "sbert_criteria"
    save_response: bool = False

    @field_validator("preferred_genres", "excluded_genres", "excluded_themes")
    @classmethod
    def clean_lists(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @model_validator(mode="after")
    def consistent_preferences(self) -> "Preferences":
        preferred = {value.casefold() for value in self.preferred_genres}
        excluded = {value.casefold() for value in self.excluded_genres}
        if preferred & excluded:
            raise ValueError("A genre cannot be both preferred and excluded. Please clarify your choices.")
        if len(self.description.split()) < 3:
            raise ValueError("Please describe the film in a short English sentence.")
        return self

    def criterion_texts(self) -> dict[str, str]:
        return {
            "description": self.description,
            "themes": self.themes,
            "mood": self.mood,
            "narrative_style": self.narrative_style,
            "setting": self.setting,
            "genres": ", ".join(self.preferred_genres),
        }


class CriticRequest(StrictModel):
    tmdb_id: int = Field(gt=0)
    profile_summary: str = Field(min_length=1, max_length=8000)
    main_contributions: list[str] = Field(default_factory=list, max_length=10)


class CriterionScore(StrictModel):
    criterion: str
    label: str
    similarity: float
    weight: float
    contribution: float
    evidence: str


class Recommendation(StrictModel):
    movie: Movie
    affinity: float
    criteria: list[CriterionScore]
    missing_criteria: list[str]
    main_contributions: list[str]
    explanation: str


class RecommendationResponse(StrictModel):
    profile_summary: str
    recommendations: list[Recommendation]
    eligible_count: int
    catalog_size: int
    method: Method
    warnings: list[str]
    latency_ms: float
    corpus_fingerprint: str
    saved: bool = False
