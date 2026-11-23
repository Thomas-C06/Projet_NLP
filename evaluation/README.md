# Ranking evaluation

`queries.json` contains 30 English requests: 10 validation queries and 20 final test queries. Use validation results to discuss weights. Keep the final test set out of parameter selection.

## Ready-to-run technical verification

Run `python -m scripts.check_application` to execute all 30 requests with all three methods on the real 150-film corpus. `technical-checks.json` records actual returned titles, constraint checks, score bounds, ranking order and warm search timings. This runs without a token or human judgments. These are technical measurements, not Precision@3 or nDCG relevance measurements.

## Create judgments

After freezing the reviewed corpus and preparing the local model:

```sh
python -m scripts.evaluate pool
```

This creates `judgments.csv` from the union of the Top 10 returned by TF-IDF, global SBERT and criterion SBERT for each request. Each method receives the same strict filters.

Complete every row with an annotator identifier and a relevance value:

- 0: unrelated, or fails an important expectation.
- 1: weak connection, with important expectations missing.
- 2: relevant, with most important expectations covered.
- 3: highly relevant, with a strong contextual match.

Prefer two independent annotators. To include a second annotation, add another row with the same query and movie, but a different annotator. The evaluator averages available judgments. An average of at least 2 counts as relevant for Precision@3. Do not add private names or identifying information.

## Compare methods

```sh
python -m scripts.evaluate run --split validation
python -m scripts.evaluate run --split test
```

Outputs are written to `evaluation/results/`. They include per-query results, method averages, corpus fingerprint, Precision@3, nDCG@3, constraint compliance and warm search latency. Precision@3 always divides by three. A method returning one film can therefore reach at most 1/3 for that query. Compliance is undefined for no returned films and is excluded from the compliance average; returned counts are reported separately.

nDCG@3 uses gains of `2^relevance - 1` and the ideal ordering of the shared judged pool. Unjudged films are not silently scored as irrelevant. Results depend on the pooling depth and the small query sample. Record disagreements, difficult cases and examples where paraphrases change the ranking.

The evaluator rejects incomplete judgments and a changed corpus. Regenerate the pool if the corpus, queries, ranking model or scoring configuration changes. Keep the selected weights unchanged during the final test.

## Automatic LLM judgments

`python -m scripts.llm_judge` fills the same pool with judgments from a local LLM (Ollama, `qwen3:14b`, temperature 0) and writes `judgments-llm.csv` with the annotator `llm:qwen3:14b` and a one-sentence reason per film. The judge sees only source metadata, never the estimated labels. Judgments are cached in `llm-judgments-cache.json` by query text, film and model, so a new configuration only judges new candidates.

`python -m scripts.tune_scoring` compares scoring configurations on the validation split and writes `scoring-selection.json`. `results-llm-validation.json` and `results-llm-test.json` contain the method comparison. `scoring-ablation-test.json` reports the same configurations on the test split for transparency only; it was produced after the choice was fixed.

`python -m scripts.tune_scoring` (default `--grid genres`) compares the semantic genre score with exact genre coverage at weights 0.05, 0.15 and 0.25, on the validation split, and writes `scoring-selection-genres.json`. `--grid story` reproduces the first selection with the original genre scoring. After changing the defaults in `backend/config.py`, refresh the LLM results:

```sh
python -m scripts.tune_scoring
python -m scripts.llm_judge
python -m scripts.evaluate run --split validation --judgments evaluation/judgments-llm.csv --output evaluation/results-llm-validation.json
python -m scripts.evaluate run --split test --judgments evaluation/judgments-llm.csv --output evaluation/results-llm-test.json
python -m scripts.check_application
```

The judgment cache only sends new query-film pairs to the LLM.

These are automatic estimates. Report them as LLM judgments, never as human annotations.

## Human study

`docs/human-study.md` describes the human protocol. `python -m scripts.human_study sheets` writes blind, shuffled sheets for two annotators, `merge` converts the completed sheets to `judgments-human.csv` for `scripts.evaluate`, and `agreement` measures agreement between annotators and with the LLM judge (`agreement-human-llm.json`).

No human relevance measurements are invented. The technical results are available immediately; independent relevance measurements use the explicit judgment workflow above.
