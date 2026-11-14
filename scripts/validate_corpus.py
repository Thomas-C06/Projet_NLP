"""Check the schema, corpus size and manual annotation coverage."""

import argparse
import json
from pathlib import Path

from backend.catalog import fingerprint, load_catalog
from backend.config import CORPUS_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=CORPUS_PATH)
    parser.add_argument("--expected", type=int, default=150)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    try:
        movies = load_catalog(args.path)
        fields = ["overview", "themes", "mood", "narrative_style", "setting", "annotation_version"]
        missing = {field: sum(not getattr(movie, field) for movie in movies) for field in fields}
        print(json.dumps({"size": len(movies), "missing": missing, "fingerprint": fingerprint(movies)}, indent=2))
        if len(movies) != args.expected:
            parser.exit(1, f"Expected {args.expected} films, found {len(movies)}.\n")
        if args.require_complete and any(missing.values()):
            parser.exit(1, "Some annotations are missing. Complete and review them before final evaluation.\n")
    except (ValueError, OSError) as error:
        parser.exit(1, f"Invalid corpus: {error}\n")


if __name__ == "__main__":
    main()
