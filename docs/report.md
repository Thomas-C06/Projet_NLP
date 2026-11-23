# Movie Match project documentation

This project implements an English movie recommendation application for the ING5 Natural Language Processing assignment. It matches a user's written expectations with a structured film corpus and explains the three highest-ranked recommendations.

The repository includes the interface and a corpus of 150 real films from the public TMDB 5000 snapshot. The setup instructions explain how to build the frontend and download the model once. Uvicorn then serves the interface and API locally without a personal API token. The technical checks cover application behaviour, and independent human relevance evaluation is pending.

## Application architecture

The frontend uses React, TypeScript, Vite and Tailwind CSS. Recharts presents the global affinity and criterion scores. The backend uses Python 3.12, FastAPI and Pydantic. The REST API separates presentation from recommendation. NumPy handles vectors, scikit-learn provides the lexical comparison, and httpx collects source metadata.

GET /api/health reports corpus and local model availability. GET /api/catalog provides selectable genres, annotated topics and film identifiers. POST /api/recommend validates the questionnaire and returns the preference summary, ranked films, scoring details and warnings. Requests are processed in memory and are not saved to a server history. A JSON download is available only after the user's action.

## Questionnaire and user profile

The story description is required. Themes, atmosphere, narrative style, setting and preferred genres are optional. Guided text questions preserve contextual expectations. Checkboxes, multiple selections and importance scales from 1 to 5 provide structured information. Empty optional answers represent no preference.

Duration limits, excluded genres, excluded annotated topics and already watched films are separate from positive preferences. A genre cannot be both preferred and excluded. Unsupported non-English requests are rejected when language detection identifies them confidently. Very short or mixed-language input can remain ambiguous; the interface clearly asks for English.

## Reference corpus

The corpus contains 150 films from the public TMDB 5000 snapshot distributed on Kaggle by the dataset publisher. Selection balances genre and decade buckets and includes several well documented reference films. Titles, identifiers, overviews, genres, years, durations and keyword tags are source metadata. The manifest records the archive checksum, source URL and corpus fingerprint. The historical snapshot covers releases through 2016 and does not contain poster paths.

Themes use source keyword tags. Setting descriptions are extracted from explicit source cues. Atmosphere and narrative-focus labels are estimated by comparing source text with fixed English descriptions in the embedding space. A similarity below 0.15 leaves the label unknown. The evidence file records original tags, source setting cues and label similarities; no human cross-review is claimed. Pydantic validates all 150 records. Unknown values remain null, including 50 setting fields, 14 atmosphere fields and 4 narrative-style fields. The interface explains estimated attributes and missing-field handling.

---

## Semantic representation

The principal model is sentence-transformers/all-MiniLM-L6-v2. It maps English sentences and short paragraphs to 384-dimensional vectors. It is used as a pretrained encoder on CPU. No task-specific model training is performed.

Cleaning only normalises whitespace. It preserves negations, punctuation and sentence context. Text longer than the supported sequence length is split into token chunks with room reserved for special tokens. Chunk vectors are averaged and normalised. The same procedure encodes user preferences and film attributes.

The model is downloaded during preparation and stored locally. Corpus vectors are cached using text content, model files and processing settings. User text is not written to the disk cache. Model and corpus preparation are separate from recommendation, so no TMDB request or model download is needed during the demonstration.

## Scoring method

Each active user criterion is compared with the corresponding film field by cosine similarity. The initial base weights are story 0.35, themes 0.25, atmosphere 0.15, narrative style 0.10, setting 0.10 and genres 0.05. An importance value from 1 to 5 multiplies each base weight.

For a film f, let A(f) be the set of criteria both supplied by the user and available for that film. For criterion b, let w(b) be its base weight, i(b) its importance, and s(b,f) its cosine similarity. The score is:

Affinity(f) = 100 * sum[w(b) * i(b) * max(0, s(b,f))] / sum[w(b) * i(b)], for b in A(f).

If no usable criterion remains, the film is not scored. Missing fields are reported, and weights are renormalised over available active criteria. This can make a film with fewer annotations look strong on fewer criteria; the displayed missing-field warning and final corpus completeness check are therefore important.

The affinity index ranges from 0 to 100. It expresses textual alignment, not a calibrated probability of enjoyment. The initial weights are design assumptions that should be checked on the validation queries.

## Filters and explanations

Strict filters are applied before ranking. A duration limit excludes unknown durations. Excluded genres and already watched identifiers are removed. Topic exclusions match reviewed tags and exclude films with unknown topic annotations. They cannot guarantee that every scene avoids a topic. If fewer than three eligible films remain, the response explains this without relaxing a constraint.

The result includes film metadata, the global index, criterion similarities, normalised weights, point contributions and source attribute text. The two largest positive contributions support a deterministic explanation. No positive match and missing annotations are shown explicitly. Two bar charts compare global and criterion-level scores. Posters are cached separately from public TMDB pages with their image sources recorded. Both posters and typography are served locally for offline use.

The cinema interface provides mood shortcuts that guide optional preferences while preserving an existing story and strict limits. A searchable collection supports accent-insensitive title searches. The mobile action keeps search accessible while completing the questionnaire. The dark layout, poster cards, reduced-motion support and expanded scoring details keep the same recommendation logic visible to the user.

---

## Evaluation protocol

The comparison has three methods: TF-IDF with cosine similarity, SBERT with one global text embedding, and SBERT with criterion-level aggregation. All methods use the same corpus and strict filters. The lexical method is a reference for checking whether semantic representations help beyond shared vocabulary. The global method tests the value of criterion-level organisation.

The project provides 30 English requests: 10 validation queries for parameter selection and 20 held-out test queries. The queries include paraphrases, multiple expectations, difficult requests and strict constraints. A pooling command gathers the union of the Top 10 from the three methods for each query.

Annotators judge candidates from 0 to 3. Two independent annotators are preferred. Their available judgments are averaged; a value of at least 2 counts as relevant. Precision@3 uses a denominator of three. nDCG@3 uses exponential gains and the ideal order within the common judged pool. Constraint compliance and the number of returned films are reported, as well as warm search latency.

The evaluator rejects unfinished judgments and mismatched corpus fingerprints. Pooling is not exhaustive relevance assessment. The small test set limits conclusions and should be discussed alongside errors and qualitative examples. No relevance measurements are invented when judgments are unavailable.

## Verification and remaining work

Automated tests cover score aggregation, missing-field renormalisation, priorities, strict filters, unknown durations, contradictions, unsupported language, ranking metrics and API responses. Frontend tests cover request construction, contradictory genres and missing-corpus behaviour. Production compilation checks TypeScript and the Vite build. Semantic integration checks use the real local encoder with fictional fixtures, which are separate from the film corpus.

The ready-to-run technical check submits all 30 requests to each of the three methods on the real 150-film corpus. All 90 searches respected strict constraints, score bounds and ranking order. Average warm search times in this local run were 235.16 ms for TF-IDF, 178.48 ms for global SBERT and 190.57 ms for criterion SBERT. These timings describe one machine and run; they are not relevance scores.

The application works without completing an annotation exercise. For an independent academic relevance study, use the documented judgment pool, select parameters on validation data and run the held-out test. Such a study and an optional factual cross-review can assess the limitations of the automatic annotations. Every member should take part in the class presentation and demonstration.

## Sources

Course assignment: Sarah Malaeb, Natural Language Processing project, ECE ING5 Data and AI, 2026-2027, pages 8-9, 15-16 and 19.

TMDB API and attribution: https://developer.themoviedb.org/docs/faq

Public metadata snapshot: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Model description: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

FastAPI documentation: https://fastapi.tiangolo.com/

Sentence Transformers documentation: https://www.sbert.net/
