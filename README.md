# Movie Match

Movie Match is our ING5 NLP project at ECE for 2026-2027. It recommends films from English descriptions of the user's expectations, with an affinity index and an explanation of the matching criteria.

## What is implemented

- React and TypeScript questionnaire with guided text fields, genre checkboxes, single-choice era and method lists, and Likert importance scales.
- Cinema interface with local film posters, mood shortcuts, a searchable collection and a mobile search action.
- FastAPI backend with Pydantic validation and an English language check.
- CPU sentence embeddings, token chunking and local corpus-vector caching.
- Six-criterion weighted scoring, Top 3 ranking, strict filters and explicit missing-data handling.
- Preference summary, deterministic explanations, contribution details, a radar chart of the match profile and Recharts bar charts.
- Opt-in structured storage of questionnaire answers and their Top 3 in a local JSON Lines file, plus a voluntary JSON export.
- TMDB collection and validation commands, annotation guide, three-method evaluation and automated tests.
- Automatic relevance judgments by a local LLM (clearly separated from human judgments) to measure Precision@3 and nDCG@3 and to select the scoring configuration.
- Optional bonus: an "Ask the critic" button that rephrases a recommendation with a local LLM (Ollama). The ranking never depends on it.

The application includes 150 real films selected from the public TMDB 5000 snapshot. It works without a personal API token. Source keywords provide the themes; setting cues are extracted from the source text. Atmosphere and narrative labels are estimates selected from fixed descriptions using the semantic encoder. Missing or weakly supported descriptions stay `null`. This is automatic annotation, not a claimed human cross-review.

## Setup and launch

Use Python 3.12 and Node.js 24. From the project root, run these commands once in PowerShell. The model download requires internet access on the first run.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
.\.venv\Scripts\python.exe -m scripts.prepare_model
```

Start the API and the built frontend from the same terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open [Movie Match](http://127.0.0.1:8000) or the [API documentation](http://127.0.0.1:8000/docs). Press Ctrl+C in the terminal to stop the server. The included corpus works without a TMDB token or `.env` file.

On macOS or Linux:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
npm --prefix frontend ci
npm --prefix frontend run build
.venv/bin/python -m scripts.prepare_model
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The lock file installs the CPU build of PyTorch on Windows and Linux and the standard build on macOS. Exact dependency versions are recorded in `requirements.lock` and `frontend/package-lock.json`.

## Data and local model

The included corpus is `data/movies.json`. Its manifest records the public source, archive checksum, corpus fingerprint and annotation method. `data/annotation-evidence.json` records the original keywords, setting evidence and label similarities for each film. The snapshot describes films released through 2016; it is not a live catalog of new releases. Poster paths are unavailable in this snapshot and stay `null`. A separate local poster cache in `frontend/public/posters/` records its own public TMDB image sources, without changing the scoring corpus.

`scripts.prepare_model` downloads the encoder and caches corpus vectors locally. After preparation, ranking works offline. The model and cached vectors are excluded from Git.

To rebuild the corpus from the frozen public snapshot on another machine, first prepare the model, then run `python -m scripts.prepare_catalog`. The script keeps an existing corpus unchanged. It needs no token. The original `scripts.collect_movies` command remains available for an optional fresh TMDB collection with a personal token.

Run `python -m scripts.validate_corpus` to check the schema and 150-film count. The optional `--require-complete` flag is deliberately stricter: it reports unavailable fields rather than filling them with invented information. See [the annotation guide](docs/annotation-guide.md).

## Development commands

For frontend development with hot reload, use two terminals in the project root.

Backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
npm.cmd --prefix frontend run dev
```

Open http://localhost:5173. API documentation is at http://127.0.0.1:8000/docs. Vite proxies `/api` to the backend. Restart the backend after changing the corpus or `.env`. Posters and typography are cached locally, so the interface also works offline. Asset provenance and the font license are in `frontend/public/ASSETS.md`.

## Scoring choices

The principal encoder is `sentence-transformers/all-MiniLM-L6-v2`. It produces normalised vectors for the user text and each film criterion. Text exceeding the token budget is split before encoding; chunk vectors are averaged and normalised.

| Criterion | Base weight |
| --- | ---: |
| Story and synopsis | 0.35 |
| Themes | 0.25 |
| Atmosphere | 0.15 |
| Narrative style | 0.10 |
| Setting | 0.10 |
| Genres | 0.05 |

Only the ratios between weights matter. An alternative genre score, the share of the preferred genres found in the film's TMDB categories, can be enabled with `GENRE_MATCH=coverage` and `GENRE_WEIGHT`. It did not improve the validation results, so it is not the default.

For each available active criterion, multiply its weight by its importance from 1 to 5. The affinity is 100 times the weighted mean of the non-negative cosine similarities. Criteria missing from either side are excluded from the denominator and missing film descriptions are shown. This index is not a probability of enjoyment. The initial weights still need validation using the validation queries.

When the user ticks "Save my answers to the local study log", the API appends the answers, the method, the Top 3 and the corpus fingerprint to `data/responses/responses.jsonl` (excluded from Git; change the path with `RESPONSES_PATH`). Nothing is stored otherwise.

Release period, duration, excluded genres, excluded annotated topics and watched films are strict filters. Unknown duration does not pass a duration limit. Contradictory genre choices are rejected. Topic exclusions use annotations and cannot cover every scene. Unsupported non-English text is rejected when confidently detected; short and mixed-language requests remain a documented limitation.

## Verification and evaluation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
npm.cmd --prefix frontend test
npm.cmd --prefix frontend run build
```

Tests use fictional fixtures only inside the test suite. They verify technical behaviour and do not measure recommendation relevance. The optional real-model integration test runs when the local model has been prepared.

The [evaluation guide](evaluation/README.md) describes the 30 queries, common judgment pool and metrics. Technical verification can run without relevance judgments using `python -m scripts.check_application`. For an independent relevance evaluation:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate pool
# Complete evaluation/judgments.csv with the relevance judgments.
.\.venv\Scripts\python.exe -m scripts.evaluate run --split validation
.\.venv\Scripts\python.exe -m scripts.evaluate run --split test
```

Compare TF-IDF, global SBERT and criterion SBERT.

Without human annotators, a local LLM can produce automatic ("silver") judgments with the same 0 to 3 scale. It requires [Ollama](https://ollama.com) and the `qwen3:14b` model. The judge only sees source metadata, never our estimated labels:

```sh
.venv/bin/python -m scripts.llm_judge
.venv/bin/python -m scripts.tune_scoring            # validation split only
.venv/bin/python -m scripts.evaluate run --split test --judgments evaluation/judgments-llm.csv
```

These numbers are automatic estimates and are reported separately from human relevance measurements. Parameters are selected with the 10 validation queries and the 20 test queries are kept for the final comparison.

Current results with automatic LLM judgments on the 20 held-out test queries:

| Method | Precision@3 | nDCG@3 |
| --- | ---: | ---: |
| TF-IDF (keywords) | 0.367 | 0.366 |
| SBERT, one global vector | 0.450 | 0.485 |
| SBERT by criterion (default) | 0.550 | 0.527 |

See the evaluation section of [the documentation](docs/report.md) for the configuration selection and limits.

## Optional local critic

If Ollama is running with the `qwen3:8b` model (`ollama pull qwen3:8b`), each recommendation shows an "Ask the critic" button. The model rewrites the explanation in two or three sentences using only the user's summary, the synopsis and the matched criteria. Change the model with `CRITIC_MODEL` and the server with `OLLAMA_URL`. Without Ollama the button is hidden and everything else works offline.

## Repository organisation

```text
backend/           API, schemas, semantic encoder and ranking
frontend/          English React application and interface tests
scripts/           Collection, validation, model preparation and evaluation
data/              Reference corpus and data instructions
evaluation/        Queries, judgment instructions and local results
tests/             Backend and scoring tests
docs/              Annotation guide, report source and demonstration notes
output/pdf/        Generated PDF documentation
local-notes/       Personal documents excluded from Git
```

The full assignment is in [Projet_Sujet.pdf](Projet_Sujet.pdf). Read [the documentation](docs/report.md) and [the presentation notes](docs/demo.md). Regenerate the PDF with `python -m scripts.build_report` after adding the final measured results.

## Data credits and working rules

This product uses the TMDB API but is not endorsed or certified by TMDB.

Check [TMDB reuse and attribution conditions](https://developer.themoviedb.org/docs/faq) before publishing the collected corpus or images. Model details are in [the official model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

Keep code, comments and shared documentation in English. Keep secrets, personal responses, installed environments, local notes and model files outside Git. Write comments for the important processing steps and update the instructions when changing execution or data preparation.
