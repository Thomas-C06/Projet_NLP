"""Start the complete local application after warming its search model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn
from backend.main import app, get_engine
from backend.schemas import Preferences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    engine = get_engine()
    if engine.movies:
        profile = Preferences(description="I would like a thoughtful story about friendship and discovery.")
        engine.recommend(profile)
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
