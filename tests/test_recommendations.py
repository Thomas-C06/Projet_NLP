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
