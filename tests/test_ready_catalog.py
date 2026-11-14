import json

import pytest

from backend.catalog import load_catalog
from backend.config import ROOT
from scripts.prepare_catalog import read_snapshot, select_films, setting_from_source


def test_bundled_corpus_is_real_and_has_provenance():
    movies = load_catalog(ROOT / "data/movies.json")
    assert len(movies) == 150
    assert len({movie.tmdb_id for movie in movies}) == 150
    assert all(movie.overview and movie.themes and movie.runtime_minutes for movie in movies)
    assert all(movie.source == f"https://www.themoviedb.org/movie/{movie.tmdb_id}" for movie in movies)
    manifest = json.loads((ROOT / "data/movies.manifest.json").read_text())
    assert manifest["human_cross_review"] is False
    assert manifest["size"] == 150
    evidence = json.loads((ROOT / "data/annotation-evidence.json").read_text())
    assert len(evidence) == 150
    assert all(entry["human_reviewed"] is False for entry in evidence)


def test_source_settings_are_extractive_and_missing_stays_unknown():
    text, evidence = setting_from_source({"overview": "Two people meet and discuss their lives."}, ["friendship"])
    assert text is None and evidence == []
    text, evidence = setting_from_source({"overview": "Two people meet in Paris."}, ["paris"])
    assert "paris" in text and evidence == ["paris"]


def test_changed_snapshot_is_not_silently_imported(tmp_path):
    path = tmp_path / "source.zip"
    path.write_bytes(b"unknown snapshot")
    with pytest.raises(ValueError, match="archive changed"):
        read_snapshot(path)


def test_selection_remains_deterministic_and_unique():
    rows = [{
        "id": str(index), "title": f"Fixture {index}",
        "overview": "A person finds a new friend and together they discover how to support their neighbours during a difficult period of their lives.",
        "genres": '[{"name":"Drama"}]', "keywords": '[{"name":"friendship"}]',
        "release_date": "2000-01-01", "runtime": "100", "vote_count": "300",
    } for index in range(1, 10)]
    first = select_films(rows, count=4)
    assert first == select_films(list(reversed(rows)), count=4)
    assert len({row["id"] for row in first}) == 4
