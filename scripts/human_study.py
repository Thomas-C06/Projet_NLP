"""Prepare, collect and analyse the human relevance study.

1. sheets: pool the Top k of the three methods and write one blind, shuffled
   annotation sheet per annotator, plus a private key that maps each item back
   to its query and film. The sheets never show the method or the LLM verdict.
2. merge: check the completed sheets and convert them to the judgment format
   read by scripts.evaluate (evaluation/judgments-human.csv).
3. agreement: measure agreement between the human annotators, and between the
   humans and the LLM judge, on the pairs judged by both.

    python -m scripts.human_study sheets --split test --depth 3
    # Each annotator fills the relevance column of their own sheet (0 to 3).
    python -m scripts.human_study merge evaluation/human/sheet-A.csv evaluation/human/sheet-B.csv
    python -m scripts.evaluate run --split test --judgments evaluation/judgments-human.csv --output evaluation/results-human-test.json
    python -m scripts.human_study agreement
"""

import argparse
import csv
import io
import json
import random
import statistics
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HUMAN_DIR = ROOT / "evaluation/human"
SHEET_FIELDS = ["item", "request", "title", "year", "runtime", "genres", "keywords", "synopsis",
                "relevance", "comment"]
KEY_FIELDS = ["item", "query_id", "split", "tmdb_id", "title", "corpus_fingerprint", "query_fingerprint"]
JUDGMENT_FIELDS = ["query_id", "split", "tmdb_id", "title", "corpus_fingerprint", "query_fingerprint",
                   "annotator", "relevance", "comment"]
# Semicolons open correctly in a French Excel; the BOM keeps accents readable.
DELIMITER = ";"


def describe(preferences: dict) -> str:
    labels = {"description": "Description", "themes": "Themes", "mood": "Atmosphere",
              "narrative_style": "Storytelling", "setting": "Setting or period"}
    parts = [f"{label}: {preferences[key]}" for key, label in labels.items() if preferences.get(key)]
    if preferences.get("preferred_genres"):
        parts.append("Preferred genres: " + ", ".join(preferences["preferred_genres"]))
    return " | ".join(parts)


def write_csv(path: Path, fields: list[str], rows: list[dict], delimiter: str = ","):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    header = text.split("\n", 1)[0]
    # Excel may save a sheet back with either separator.
    delimiter = DELIMITER if header.count(DELIMITER) > header.count(",") else ","
    return list(csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter))


def make_sheets(engine, queries, split: str, depth: int, annotators: list[str], output: Path, seed: int = 7):
    from scripts.evaluate import query_fingerprint
    from scripts.llm_judge import pool

    selected = [query for query in queries if query["split"] == split]
    if not selected:
        raise ValueError("There are no queries in the selected split.")
    by_id = {movie.tmdb_id: movie for movie in engine.movies}
    pairs = pool(engine, selected, depth)
    key, items = [], []
    for number, (query, movie_id) in enumerate(pairs, 1):
        movie = by_id[movie_id]
        item = f"{number:04d}"
        key.append({"item": item, "query_id": query["query_id"], "split": query["split"], "tmdb_id": movie_id,
                    "title": movie.title, "corpus_fingerprint": engine.fingerprint,
                    "query_fingerprint": query_fingerprint(query)})
        items.append({"item": item, "request": describe(query["preferences"]), "title": movie.title,
                      "year": movie.release_year or "", "runtime": movie.runtime_minutes or "",
                      "genres": ", ".join(movie.genres), "keywords": ", ".join(movie.themes or []),
                      "synopsis": movie.overview or "", "relevance": "", "comment": ""})
    if any((output / f"sheet-{name}.csv").exists() for name in annotators):
        raise ValueError("Annotation sheets already exist. Choose another folder to keep completed work.")
    write_csv(output / "key.csv", KEY_FIELDS, key)
    for index, name in enumerate(annotators):
        # A different order per annotator limits fatigue and order effects shared by both.
        order = items[:]
        random.Random(seed + index).shuffle(order)
        write_csv(output / f"sheet-{name}.csv", SHEET_FIELDS, order, DELIMITER)
    return len(items)


def merge(sheets: list[Path], key_path: Path, output: Path) -> int:
    key = {row["item"]: row for row in read_csv(key_path)}
    rows = []
    for sheet in sheets:
        annotator = "human:" + sheet.stem.removeprefix("sheet-")
        values = {row["item"]: row for row in read_csv(sheet)}
        if set(values) != set(key):
            raise ValueError(f"{sheet.name} does not contain exactly the items of the key.")
        for item, row in values.items():
            relevance = row["relevance"].strip()
            if relevance not in {"0", "1", "2", "3"}:
                raise ValueError(f"{sheet.name}, item {item}: relevance must be 0, 1, 2 or 3.")
            rows.append({**{field: key[item][field] for field in KEY_FIELDS if field != "item"},
                         "annotator": annotator, "relevance": relevance,
                         "comment": row.get("comment", "").strip()})
    rows.sort(key=lambda row: (row["query_id"], int(row["tmdb_id"]), row["annotator"]))
    write_csv(output, JUDGMENT_FIELDS, rows)
    return len(rows)


def cohen_kappa(first: list[int], second: list[int], categories: list[int], weighted: bool = False) -> float | None:
    """Cohen's kappa; the weighted version uses linear disagreement weights."""
    n = len(first)
    if n == 0:
        return None
    span = max(categories) - min(categories) or 1
    def disagreement(a, b):
        return abs(a - b) / span if weighted else float(a != b)
    observed = sum(disagreement(a, b) for a, b in zip(first, second)) / n
    p1 = {c: first.count(c) / n for c in categories}
    p2 = {c: second.count(c) / n for c in categories}
    expected = sum(p1[a] * p2[b] * disagreement(a, b) for a in categories for b in categories)
    return None if expected == 0 else 1 - observed / expected


def compare(first: list[int], second: list[int]) -> dict:
    binary_first = [int(value >= 2) for value in first]
    binary_second = [int(value >= 2) for value in second]
    rounded = lambda value: None if value is None else round(value, 3)
    return {
        "pairs": len(first),
        "exact_agreement": rounded(sum(a == b for a, b in zip(first, second)) / len(first)) if first else None,
        "weighted_kappa_0_3": rounded(cohen_kappa(first, second, [0, 1, 2, 3], weighted=True)),
        "relevant_agreement": rounded(sum(a == b for a, b in zip(binary_first, binary_second)) / len(first))
        if first else None,
        "relevant_kappa": rounded(cohen_kappa(binary_first, binary_second, [0, 1])),
        "mean_absolute_difference": rounded(statistics.mean(abs(a - b) for a, b in zip(first, second)))
        if first else None,
    }


def by_annotator(rows: list[dict]) -> dict[str, dict[tuple[str, int], int]]:
    result = {}
    for row in rows:
        result.setdefault(row["annotator"], {})[(row["query_id"], int(row["tmdb_id"]))] = int(row["relevance"])
    return result


def agreement(human_rows: list[dict], llm_rows: list[dict]) -> dict:
    humans = by_annotator(human_rows)
    report = {"annotators": sorted(humans), "between_humans": {}, "humans_vs_llm": {}}
    for a, b in combinations(sorted(humans), 2):
        shared = sorted(set(humans[a]) & set(humans[b]))
        report["between_humans"][f"{a} vs {b}"] = compare([humans[a][k] for k in shared],
                                                           [humans[b][k] for k in shared])
    llm = {}
    for judge, values in by_annotator(llm_rows).items():
        llm.update(values)
        report["llm_judge"] = judge
    for name, values in humans.items():
        shared = sorted(set(values) & set(llm))
        report["humans_vs_llm"][name] = compare([values[k] for k in shared], [llm[k] for k in shared])
    # The rounded human mean is the reference used for disagreements with the LLM.
    shared = sorted(set.intersection(*(set(values) for values in humans.values())) & set(llm)) if humans else []
    means = [statistics.mean(humans[name][k] for name in humans) for k in shared]
    report["human_mean_vs_llm"] = compare([int(value + 0.5) for value in means], [llm[k] for k in shared])
    report["largest_disagreements_with_llm"] = sorted(
        ({"query_id": k[0], "tmdb_id": k[1], "human_mean": round(mean, 2), "llm": llm[k]}
         for k, mean in zip(shared, means) if abs(mean - llm[k]) >= 1.5),
        key=lambda row: -abs(row["human_mean"] - row["llm"]))
    report["note"] = ("Kappa compares agreement with chance agreement: below 0.2 slight, 0.2-0.4 fair, "
                      "0.4-0.6 moderate, 0.6-0.8 substantial, above 0.8 almost perfect (Landis and Koch).")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="action", required=True)
    sheets = commands.add_parser("sheets", help="Write blind annotation sheets.")
    sheets.add_argument("--split", choices=["validation", "test"], default="test")
    sheets.add_argument("--depth", type=int, default=3, help="Top-k pooled from each method.")
    sheets.add_argument("--annotators", nargs="+", default=["A", "B"])
    sheets.add_argument("--output", type=Path, default=HUMAN_DIR)
    joined = commands.add_parser("merge", help="Convert completed sheets to judgments.")
    joined.add_argument("sheets", nargs="+", type=Path)
    joined.add_argument("--key", type=Path, default=HUMAN_DIR / "key.csv")
    joined.add_argument("--output", type=Path, default=ROOT / "evaluation/judgments-human.csv")
    agree = commands.add_parser("agreement", help="Measure human-human and human-LLM agreement.")
    agree.add_argument("--human", type=Path, default=ROOT / "evaluation/judgments-human.csv")
    agree.add_argument("--llm", type=Path, default=ROOT / "evaluation/judgments-llm.csv")
    agree.add_argument("--output", type=Path, default=ROOT / "evaluation/agreement-human-llm.json")
    args = parser.parse_args()
    try:
        if args.action == "sheets":
            from backend.catalog import load_catalog
            from backend.config import CACHE_PATH, CORPUS_PATH, MODEL_PATH
            from backend.embeddings import SentenceEncoder
            from backend.recommender import Recommender
            from scripts.evaluate import read_queries
            engine = Recommender(load_catalog(CORPUS_PATH), SentenceEncoder(MODEL_PATH, CACHE_PATH))
            count = make_sheets(engine, read_queries(ROOT / "evaluation/queries.json"), args.split,
                                args.depth, args.annotators, args.output)
            print(f"{count} items per annotator written to {args.output}. Keep key.csv away from annotators.")
        elif args.action == "merge":
            print(f"Saved {merge(args.sheets, args.key, args.output)} human judgments to {args.output}")
        else:
            result = agreement(read_csv(args.human), read_csv(args.llm))
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps({key: result[key] for key in ("between_humans", "human_mean_vs_llm")}, indent=2))
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Human study failed: {error}\n")


if __name__ == "__main__":
    main()
