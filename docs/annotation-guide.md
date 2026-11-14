# Film annotation guide

We use the same rules for all 150 films. The objective is to compare informative English descriptions with user preferences.

## Included automatic annotations

The application ships with a frozen public TMDB snapshot and needs no manual preparation to run. `source-semantic-v1` means that source keywords supply themes, setting descriptions are extracted from the source, and mood or narrative-focus labels are estimated using the closest fixed description in the embedding space. A cosine similarity below 0.15 leaves the label unknown. This threshold is a processing rule, not a calibrated confidence level.

The annotation evidence file preserves keyword tags, setting cues and label similarities. No group cross-review is claimed for these records. The following procedure describes an optional factual review for the academic submission. Setting values stay null when no explicit source cue is found.

## Source and missing values

Keep the TMDB identifier, original source URL, collection manifest and available metadata. Use `null` when a value is unknown. Do not infer a film's setting from its release year. Do not replace an unknown duration with zero.

The optional live API collector writes available metadata and leaves manual fields empty. The public-snapshot preparation command additionally applies the documented automatic annotation rules. Raw source data stays in `data/raw/`. Keyword tags can be noisy and are not a complete account of a film's themes.

## Shared fields

- `themes`: three to six short English topic phrases, for example `identity`, `family relationships` or `social inequality`. Prefer the same phrase for the same concept across films.
- `mood`: one or two sentences about the atmosphere, such as tense, melancholic, playful or comforting. Describe the film, rather than claiming how every viewer will feel.
- `narrative_style`: one or two sentences about pace, viewpoint and story structure. Distinguish a slow character study from an investigation or a nonlinear narrative.
- `setting`: one or two sentences about location, period and environment. Keep uncertainty explicit.
- `annotation_version`: the agreed version, for example `v1-reviewed`, after another group member has checked the entry.

Use information supported by the film or a reliable synopsis. Avoid copying long descriptions from other websites. The app explanations repeat these annotations, so an inaccurate annotation can produce an inaccurate explanation.

## Review procedure

1. One member completes the missing fields in `data/movies.json`.
2. Another member checks the film identity, factual support, English wording and shared topic names.
3. Resolve disagreements before assigning the annotation version.
4. Keep a review record with the film identifier, annotator, reviewer and date in `data/annotation-reviews.csv`.
5. Validate the complete corpus, freeze its version, then regenerate the embeddings and judgment pool.

The automatic completeness check verifies populated fields, not factual accuracy or a real second review. Both remain the group's responsibility.

## Strict exclusions

Genre exclusions use exact genre categories. Topic exclusions use the reviewed theme tags. A film with unknown themes is excluded when a strict topic filter is used. Topic tags do not give a complete content warning for every scene; this limit is shown in the application.

## Attribution

Follow the [TMDB attribution rules](https://developer.themoviedb.org/docs/faq). Keep data provenance in the corpus and credits in the interface. Review reuse conditions before putting the collected data, images or review records on a public repository.
