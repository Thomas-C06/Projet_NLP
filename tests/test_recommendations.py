import json

import numpy as np
import pytest
from pydantic import ValidationError

from backend.catalog import fingerprint, load_catalog
from backend.embeddings import clean_text, normalise
from backend.recommender import Recommender, check_english
from backend.schemas import Importance, Preferences


def test_ranking_and_default_renormalisation(movies, encoder):
    result = Recommender(movies, encoder).recommend(Preferences(
        description="I would like a friendship story.", themes="space", mood="friendship"
    ))
    film = next(item for item in result.recommendations if item.movie.tmdb_id == 1)
    assert film.affinity == pytest.approx(100 * (0.35 + 0.15) / (0.35 + 0.25 + 0.15))
    assert sum(item.weight for item in film.criteria) == pytest.approx(1)
    assert sum(item.contribution for item in film.criteria) == pytest.approx(film.affinity)


def test_missing_annotations_are_visible(movies, encoder):
    result = Recommender(movies, encoder).recommend(Preferences(
        description="I would like a friendship story.", themes="space"
    ), top_k=4)
    film = next(item for item in result.recommendations if item.movie.tmdb_id == 4)
    assert film.affinity == pytest.approx(100)
    assert film.missing_criteria == ["Themes"]
    assert film.criteria[0].weight == 1
    assert "missing" in " ".join(result.warnings).lower()


def test_all_strict_filters_and_unknown_runtime(movies, encoder):
    result = Recommender(movies, encoder).recommend(Preferences(
        description="I would like a friendship story.", excluded_genres=["science fiction"],
        seen_movie_ids=[1], max_runtime=110,
    ))
    assert [item.movie.tmdb_id for item in result.recommendations] == [4]
    assert result.eligible_count == 1
    assert any("No strict filter was relaxed" in text for text in result.warnings)


def test_unknown_themes_do_not_pass_strict_topic_filter(movies, encoder):
    result = Recommender(movies, encoder).recommend(Preferences(
        description="I would like a friendship story.", excluded_themes=["fear"]
    ))
    assert {item.movie.tmdb_id for item in result.recommendations} == {1, 2}


def test_zero_candidates_does_not_load_encoder(movies):
    result = Recommender(movies).recommend(Preferences(
        description="I would like a friendship story.", seen_movie_ids=[1, 2, 3, 4]
    ))
    assert result.recommendations == []
    assert result.eligible_count == 0


def test_priorities_change_the_score(movies, encoder):
    engine = Recommender(movies, encoder)
    profile = Preferences(description="I would like a friendship story.", themes="space")
    normal = engine.recommend(profile, top_k=4)
    changed = engine.recommend(profile.model_copy(update={"importance": Importance(themes=5, description=1)}), top_k=4)
    scores = lambda result: {item.movie.tmdb_id: item.affinity for item in result.recommendations}
    assert scores(changed)[2] > scores(normal)[2]


@pytest.mark.parametrize("method", ["tfidf", "sbert_global", "sbert_criteria"])
def test_methods_share_the_same_filters(movies, encoder, method):
    result = Recommender(movies, encoder).recommend(Preferences(
        description="I would like a friendship story.", max_runtime=110, method=method
    ))
    assert {item.movie.tmdb_id for item in result.recommendations} == {1, 4}
    assert all(0 <= item.affinity <= 100 for item in result.recommendations)


def test_validation_and_unsupported_language():
    with pytest.raises(ValidationError):
        Preferences(description="   ")
    with pytest.raises(ValidationError):
        Preferences(description="I want a moving story.", preferred_genres=["Drama"], excluded_genres=["drama"])
    with pytest.raises(ValueError, match="Only English"):
        check_english(Preferences(description="Je voudrais regarder un film touchant sur une famille qui apprend à se retrouver après une séparation."))
    check_english(Preferences(description="I want a moving story about a family learning to live together again."))


def test_negative_topics_and_conflicting_themes_need_clarification(movies, encoder):
    engine = Recommender(movies, encoder)
    with pytest.raises(ValueError, match="move exclusions"):
        engine.recommend(Preferences(description="I would like a serious film without Horror."))
    with pytest.raises(ValueError, match="both requested and excluded"):
        engine.recommend(Preferences(description="I would like a serious film.", themes="fear", excluded_themes=["fear"]))


def test_duplicate_ids_and_cache_fingerprint(movies, tmp_path):
    path = tmp_path / "movies.json"
    path.write_text(json.dumps([movies[0].model_dump(), movies[0].model_dump()]))
    with pytest.raises(ValueError, match="duplicate"):
        load_catalog(path)
    changed = movies[0].model_copy(update={"mood": "A different atmosphere."})
    assert fingerprint([changed]) != fingerprint([movies[0]])


def test_preprocessing_keeps_negation_and_zero_vectors():
    assert clean_text("  I do not want\nviolence. ") == "I do not want violence."
    assert np.isfinite(normalise(np.zeros((2, 3)))).all()


def test_release_period_is_a_strict_filter(movies, encoder):
    dated = [movie.model_copy(update={"release_year": year}) for movie, year in zip(movies, (1975, 1995, None, 2010))]
    engine = Recommender(dated, encoder)
    result = engine.recommend(Preferences(description="A friendship story in space.", release_period="1980-1999"),
                              validate_language=False)
    assert [item.movie.tmdb_id for item in result.recommendations] == [2]
    assert "between 1980 and 1999" in result.profile_summary


def comedy_films():
    from backend.schemas import Movie
    shared = dict(overview="A friendship story.", themes=["friendship"], mood="A warm friendship.",
                  narrative_style="Character study.", setting="A small town.", runtime_minutes=95,
                  source="Fictional test record", annotation_version="test-v1")
    return [
        Movie(tmdb_id=10, title="Adult Comedy", genres=["Comedy"], **shared),
        Movie(tmdb_id=11, title="Family Comedy", genres=["Comedy", "Family"], **shared),
        Movie(tmdb_id=12, title="Crime Drama", genres=["Crime", "Drama"], **shared),
    ]


def test_genre_coverage_uses_exact_categories():
    from backend.recommender import genre_coverage
    assert genre_coverage(["Comedy", "Family"], ["Comedy"]) == 0.5
    assert genre_coverage(["comedy", "family"], ["Family", "Comedy", "Music"]) == 1
    assert genre_coverage(["Comedy"], ["Drama"]) == 0
    assert genre_coverage([], ["Drama"]) == 0


def test_family_request_prefers_films_with_every_requested_genre(encoder):
    # Same texts: only the genres separate the films, as for The Hangover in a family request.
    result = Recommender(comedy_films(), encoder, genre_match="coverage").recommend(Preferences(
        description="I would like a friendship story.", preferred_genres=["Comedy", "Family"]
    ))
    assert [item.movie.title for item in result.recommendations] == ["Family Comedy", "Adult Comedy", "Crime Drama"]
    genre_scores = {item.movie.title: next(c.similarity for c in item.criteria if c.criterion == "genres")
                    for item in result.recommendations}
    assert genre_scores == {"Family Comedy": 1, "Adult Comedy": 0.5, "Crime Drama": 0}


def test_genre_weight_changes_the_genre_share(encoder):
    profile = Preferences(description="I would like a friendship story.", preferred_genres=["Family"])
    films = comedy_films()
    light = Recommender(films, encoder, genre_match="coverage", genre_weight=0.05).recommend(profile)
    strong = Recommender(films, encoder, genre_match="coverage", genre_weight=0.25).recommend(profile)
    def share(result):
        film = next(item for item in result.recommendations if item.movie.tmdb_id == 11)
        return next(c.weight for c in film.criteria if c.criterion == "genres")
    assert share(light) == pytest.approx(0.05 / 0.40)
    assert share(strong) == pytest.approx(0.25 / 0.60)


def test_invalid_genre_configuration_is_rejected(movies):
    with pytest.raises(ValueError):
        Recommender(movies, genre_match="fuzzy")
    with pytest.raises(ValueError):
        Recommender(movies, genre_weight=2)
