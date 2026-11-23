# Human relevance study

This protocol measures ranking quality with human judgments and checks the automatic LLM judge against them. It uses the same queries, pooling, 0 to 3 scale and metrics as the LLM evaluation, so the two results can be compared directly.

## Who and how much

- Two annotators, ideally people who did not build the scoring (classmates, friends). Each works alone and does not discuss items with the other until both have finished.
- Test split only (20 queries), pooled from the Top 3 of the three methods. This is every film that can appear in a reported Top 3, usually between 100 and 150 items, about 45 to 60 minutes per annotator.
- No private names in the files: annotators are identified by a letter (A, B).

## Steps

1. Freeze the configuration. Run the genre selection first (`python -m scripts.tune_scoring`) and update `backend/config.py` if needed. The sheets depend on the rankings: regenerate them after any scoring change.
2. Create the sheets:

   ```sh
   python -m scripts.human_study sheets --split test --depth 3
   ```

   This writes `evaluation/human/sheet-A.csv`, `sheet-B.csv` and `key.csv`. Send each annotator only their own sheet. The sheets are shuffled differently for each annotator and show neither the method, nor the rank, nor the LLM verdict. Keep `key.csv` to yourself.
3. Each annotator fills the `relevance` column (and `comment` when unsure), opening the sheet in Excel or LibreOffice and saving it as CSV.
4. Merge and evaluate:

   ```sh
   python -m scripts.human_study merge evaluation/human/sheet-A.csv evaluation/human/sheet-B.csv
   python -m scripts.evaluate run --split test --judgments evaluation/judgments-human.csv --output evaluation/results-human-test.json
   python -m scripts.human_study agreement
   ```

   `merge` refuses a sheet with a missing or invalid value. `agreement` writes `evaluation/agreement-human-llm.json`.

## Instructions for annotators

Read the request, then the film's title, genres, keywords and synopsis. You may use what you already know about the film. Judge only how well it fits the positive expectations: the duration limits and excluded genres were already applied.

| Value | Meaning |
| ---: | --- |
| 0 | Unrelated, or fails an important expectation (for example an adult comedy for a family request). |
| 1 | Weak connection: one part matches, important expectations are missing. |
| 2 | Relevant: most important expectations are covered. |
| 3 | Highly relevant: a strong match with the story, tone and context requested. |

Do not look up how other people rated the film and do not try to guess which system proposed it. When hesitating between two values, choose one and write a short comment.

## What to report

- Precision@3 and nDCG@3 per method from `results-human-test.json`, next to the LLM results. The human pool is the Top 3, so nDCG@3 is computed within that pool; the LLM pool used the Top 10.
- Agreement between the two annotators: exact agreement, linearly weighted Cohen's kappa on the 0 to 3 scale, and kappa on the relevant/not relevant decision (threshold 2).
- Agreement between the human mean and the LLM judge with the same measures, and a few of the largest disagreements listed in the agreement file, with the annotators' comments.
- Limits: two annotators, 20 queries, pooled judgments. Kappa reads roughly as slight below 0.2, fair up to 0.4, moderate up to 0.6, substantial up to 0.8 and almost perfect above.
