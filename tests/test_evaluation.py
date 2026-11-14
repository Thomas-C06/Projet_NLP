import csv

import pytest

from scripts.evaluate import evaluate, make_pool, ndcg_at_3, precision_at_3, read_judgments
from backend.recommender import Recommender


def test_ranking_metrics_and_relevance_threshold():
    judgments = {1: 3, 2: 2, 3: 1, 4: 0}
    assert precision_at_3([1, 2, 3], judgments) == pytest.approx(2 / 3)
    assert ndcg_at_3([1, 2, 3], judgments) == pytest.approx(1)
    assert ndcg_at_3([4, 3, 2], judgments) < ndcg_at_3([1, 2, 3], judgments)
    assert ndcg_at_3([], judgments) == 0
    assert precision_at_3([1], judgments) == pytest.approx(1 / 3)
    assert ndcg_at_3([4], {4: 0}) == 0


def test_unfinished_judgments_are_not_counted_as_zero(tmp_path):
    path = tmp_path / "judgments.csv"
    path.write_text("query_id,tmdb_id,corpus_fingerprint,annotator,relevance\nq1,1,abc,,\n")
    with pytest.raises(ValueError, match="Complete every judgment"):
        read_judgments(path, "abc")


def test_judgments_detect_changed_corpus(tmp_path):
    path = tmp_path / "judgments.csv"
    path.write_text("query_id,tmdb_id,corpus_fingerprint,annotator,relevance\nq1,1,old,A,3\n")
    with pytest.raises(ValueError, match="corpus changed"):
        read_judgments(path, "new")


def test_pool_and_evaluation_workflow(movies, encoder, tmp_path):
    engine = Recommender(movies, encoder)
    queries = [{"query_id": "fixture", "split": "validation", "preferences": {
        "description": "I want a story about friendship.", "max_runtime": 110,
    }}]
    path = tmp_path / "judgments.csv"
    make_pool(engine, queries, path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        row.update(annotator="test-reviewer", relevance="3")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    judgments = read_judgments(path, engine.fingerprint, queries)
    result = evaluate(engine, queries, judgments, "validation")
    assert len(result["methods"]) == 3
    assert all(values["constraint_compliance"] == 1 for values in result["methods"].values())
    assert all(values["returned_count"] == 2 for values in result["methods"].values())
    changed = [{**queries[0], "preferences": {"description": "I want a different story about friendship."}}]
    with pytest.raises(ValueError, match="query changed"):
        read_judgments(path, engine.fingerprint, changed)
