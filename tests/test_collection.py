import json

import pytest

import scripts.collect_movies as collection


def test_collection_preserves_missing_fields_and_existing_annotations(tmp_path, monkeypatch):
    monkeypatch.setattr(collection, "ROOT", tmp_path)

    def fake_get(client, path, params=None):
        if path == "discover/movie":
            return {"results": [{"id": 11}, {"id": 12}]}
        movie_id = int(path.split("/")[-1])
        return {"id": movie_id, "title": f"Fixture {movie_id}", "overview": "A fictional source description.",
                "genres": [{"name": "Drama"}], "release_date": "2000-01-01", "runtime": 0}

    monkeypatch.setattr(collection, "get_json", fake_get)
    output = tmp_path / "data/movies.json"
    collection.collect(2, "test-token", output)
    movies = json.loads(output.read_text())
    assert len(movies) == 2
    assert all(movie["themes"] is None and movie["runtime_minutes"] is None for movie in movies)
    assert all(movie["annotation_version"] is None for movie in movies)
    assert "test-token" not in output.read_text()
    with pytest.raises(ValueError, match="already exists"):
        collection.collect(2, "test-token", output)
