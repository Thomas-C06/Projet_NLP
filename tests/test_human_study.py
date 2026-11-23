import csv

import pytest

from backend.recommender import Recommender
from scripts.evaluate import evaluate, read_judgments
from scripts.human_study import agreement, cohen_kappa, make_sheets, merge, read_csv


QUERIES = [
    {"query_id": "t01", "split": "test", "preferences": {"description": "I would like a friendship story."}},
    {"query_id": "v01", "split": "validation", "preferences": {"description": "I want a journey into space."}},
]


def fill(path, values):
    rows = read_csv(path)
    for row in rows:
        row["relevance"] = str(values[row["title"]])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def test_sheets_are_blind_shuffled_and_merge_into_judgments(movies, encoder, tmp_path):
    engine = Recommender(movies, encoder)
    count = make_sheets(engine, QUERIES, "test", 3, ["A", "B"], tmp_path)
    sheet = read_csv(tmp_path / "sheet-A.csv")
    assert len(sheet) == count and all(row["relevance"] == "" for row in sheet)
    # Nothing in a sheet reveals the method, the query identifier or another verdict.
    assert not {"method", "query_id", "annotator"} & set(sheet[0])
    assert {row["item"] for row in sheet} == {row["item"] for row in read_csv(tmp_path / "sheet-B.csv")}
    with pytest.raises(ValueError):
        merge([tmp_path / "sheet-A.csv"], tmp_path / "key.csv", tmp_path / "judgments.csv")
    scores = {"Test Friendship": 3, "Test Space": 0, "Test Fear": 1, "Test Missing": 2}
    fill(tmp_path / "sheet-A.csv", scores)
    fill(tmp_path / "sheet-B.csv", scores)
    merge([tmp_path / "sheet-A.csv", tmp_path / "sheet-B.csv"], tmp_path / "key.csv", tmp_path / "judgments.csv")
    judgments = read_judgments(tmp_path / "judgments.csv", engine.fingerprint, QUERIES)
    assert len(judgments) == count
    result = evaluate(engine, QUERIES, judgments, "test")
    assert result["methods"]["sbert_criteria"]["returned_count"] == 3


def test_existing_sheets_are_not_overwritten(movies, encoder, tmp_path):
    engine = Recommender(movies, encoder)
    make_sheets(engine, QUERIES, "test", 3, ["A"], tmp_path)
    with pytest.raises(ValueError):
        make_sheets(engine, QUERIES, "test", 3, ["A"], tmp_path)


def test_kappa_values():
    assert cohen_kappa([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], weighted=True) == pytest.approx(1)
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0], [0, 1]) == pytest.approx(0)
    # Linear weights penalise a 0/3 disagreement more than a 2/3 one.
    near = cohen_kappa([0, 1, 2, 3], [0, 1, 3, 3], [0, 1, 2, 3], weighted=True)
    far = cohen_kappa([0, 1, 2, 3], [3, 1, 2, 3], [0, 1, 2, 3], weighted=True)
    assert near > far


def test_agreement_between_humans_and_llm():
    def rows(annotator, values):
        return [{"query_id": "t01", "tmdb_id": str(i), "annotator": annotator, "relevance": str(v)}
                for i, v in enumerate(values)]
    human = rows("human:A", [3, 2, 0, 1]) + rows("human:B", [3, 2, 1, 1])
    llm = rows("llm:test", [3, 0, 0, 1])
    report = agreement(human, llm)
    assert report["between_humans"]["human:A vs human:B"]["exact_agreement"] == 0.75
    assert report["humans_vs_llm"]["human:A"]["pairs"] == 4
    assert report["largest_disagreements_with_llm"] == [
        {"query_id": "t01", "tmdb_id": 1, "human_mean": 2, "llm": 0}]
