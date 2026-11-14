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
  method: 'tfidf' | 'sbert_global' | 'sbert_criteria';
}

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
}
