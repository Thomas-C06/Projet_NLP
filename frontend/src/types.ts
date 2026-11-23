export const criteria = ['description', 'themes', 'mood', 'narrative_style', 'setting', 'genres'] as const;
export type Criterion = typeof criteria[number];
export const labels: Record<Criterion, string> = {
  description: 'Story', themes: 'Themes', mood: 'Atmosphere',
  narrative_style: 'Storytelling', setting: 'Setting', genres: 'Genres',
};

export interface Preferences {
  description: string;
  themes: string;
  mood: string;
  narrative_style: string;
  setting: string;
  preferred_genres: string[];
  importance: Record<Criterion, number>;
  max_runtime: number | null;
  excluded_genres: string[];
  excluded_themes: string[];
  seen_movie_ids: number[];
  release_period: ReleasePeriod;
  method: Method;
  save_response: boolean;
}

export type Method = 'tfidf' | 'sbert_global' | 'sbert_criteria';
export const methods: { value: Method; label: string; note: string }[] = [
  { value: 'sbert_criteria', label: 'Semantic, by criterion (recommended)', note: 'Each answer is compared with the matching film attribute, then weighted by your priorities.' },
  { value: 'sbert_global', label: 'Semantic, one overall profile', note: 'All your answers are merged and compared with each film as a whole. Priorities are not used.' },
  { value: 'tfidf', label: 'Keywords only (TF-IDF baseline)', note: 'Shared words only, without meaning. Useful to compare with the semantic methods.' },
];

export type ReleasePeriod = 'any' | 'before-1980' | '1980-1999' | '2000-later';
export const releasePeriods: { value: ReleasePeriod; label: string }[] = [
  { value: 'any', label: 'Any period' },
  { value: 'before-1980', label: 'Classics, before 1980' },
  { value: '1980-1999', label: '1980 to 1999' },
  { value: '2000-later', label: '2000 and later' },
];

export interface Movie {
  tmdb_id: number;
  title: string;
  overview: string | null;
  genres: string[];
  themes: string[] | null;
  mood: string | null;
  narrative_style: string | null;
  setting: string | null;
  release_year: number | null;
  runtime_minutes: number | null;
  poster_path: string | null;
  source: string;
  annotation_version: string | null;
}

export interface CriterionScore {
  criterion: string;
  label: string;
  similarity: number;
  weight: number;
  contribution: number;
  evidence: string;
}

export interface Recommendation {
  movie: Movie;
  affinity: number;
  criteria: CriterionScore[];
  missing_criteria: string[];
  main_contributions: string[];
  explanation: string;
}

export interface Results {
  profile_summary: string;
  recommendations: Recommendation[];
  eligible_count: number;
  catalog_size: number;
  method: Preferences['method'];
  warnings: string[];
  latency_ms: number;
  corpus_fingerprint: string;
  saved: boolean;
}

export interface Catalog {
  movies: { tmdb_id: number; title: string }[];
  genres: string[];
  themes: string[];
}

export interface Health {
  status: string;
  catalog_size: number;
  target_size: number;
  model_available: boolean;
  corpus_ready: boolean;
  annotated_count: number;
  estimated_annotations?: number;
  critic_available?: boolean;
}
