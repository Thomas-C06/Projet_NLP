# Reference corpus

The included corpus contains 150 real TMDB films with English descriptions, selected from the public [TMDB 5000 snapshot](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata). No API token is required to use the application.

`movies.manifest.json` records provenance, archive checksum, corpus fingerprint and annotation method. The snapshot covers films through 2016. Metadata, source keywords and overview text come from TMDB via the dataset publisher. No fictional entries are used in the catalog.

`annotation-evidence.json` records the original keyword tags, source setting cues and estimated-label similarities. Atmosphere and narrative labels are selected from fixed English descriptions with the sentence encoder. They are estimates, not reviewed facts. Unknown settings and low-confidence labels remain `null`; the ranking renormalises the available active criteria.

To validate and prepare vectors:

```sh
python -m scripts.validate_corpus
python -m scripts.prepare_model
```

To rebuild a missing corpus, run `python -m scripts.prepare_catalog` after preparing the model. It downloads the public archive without authentication, checks its checksum and creates the corpus and provenance files. It keeps an existing corpus unchanged. `scripts.collect_movies` remains available for an optional fresh API collection with a token.

The [annotation guide](../docs/annotation-guide.md) explains the automatic fields and how the group can review them. `--require-complete` is an optional stricter validation that reports missing fields. Restart the backend after modifying the corpus. Keep tokens and private responses outside Git. TMDB terms and attribution still apply; the dataset is not declared CC0 by this project.
