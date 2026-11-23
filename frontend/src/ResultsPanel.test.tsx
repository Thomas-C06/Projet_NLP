import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import ResultsPanel from './ResultsPanel';
import { initialPreferences } from './App';
import type { Results } from './types';

const criterion = (key: string, label: string, similarity: number) => ({
  criterion: key, label, similarity, weight: 1 / 3, contribution: 100 * similarity / 3, evidence: label + ' evidence',
});

const results: Results = {
  profile_summary: 'You are looking for: a test.', eligible_count: 2, catalog_size: 2, method: 'sbert_criteria',
  warnings: [], latency_ms: 12.3, corpus_fingerprint: 'abc', saved: true,
  recommendations: [1, 2].map(id => ({
    movie: { tmdb_id: id, title: 'Film ' + id, overview: 'A story.', genres: ['Drama'], themes: null, mood: null,
      narrative_style: null, setting: null, release_year: 2000, runtime_minutes: 100, poster_path: null,
      source: 'https://www.themoviedb.org/movie/' + id, annotation_version: null },
    affinity: 50 - id, missing_criteria: [], main_contributions: ['Story'], explanation: 'Main matching criteria: Story.',
    criteria: [criterion('description', 'Story', 0.6), criterion('themes', 'Themes', 0.4), criterion('mood', 'Atmosphere', 0.3)],
  })),
};

describe('Results', () => {
  it('shows the match profile radar, the method and the storage confirmation', () => {
    render(<ResultsPanel results={results} profile={{ ...initialPreferences(), description: 'a test' }} />);
    expect(screen.getByRole('heading', { name: /Match profile/ })).toBeInTheDocument();
    expect(screen.getByText(/Semantic, by criterion/)).toBeInTheDocument();
    expect(screen.getByText(/saved to the local study log/)).toBeInTheDocument();
    expect(screen.getAllByRole('article')).toHaveLength(2);
  });

  it('skips the radar when fewer than three criteria are compared', () => {
    const single = { ...results, saved: false, recommendations: results.recommendations.map(item => ({
      ...item, criteria: [criterion('description', 'Story', 0.5)] })) };
    render(<ResultsPanel results={single} profile={initialPreferences()} />);
    expect(screen.queryByRole('heading', { name: /Match profile/ })).not.toBeInTheDocument();
    expect(screen.queryByText(/saved to the local study log/)).not.toBeInTheDocument();
  });
});

describe('Critic', () => {
  it('asks the optional local critic only when it is available', async () => {
    const api = await import('./api');
    const spy = vi.spyOn(api, 'getCriticPitch').mockResolvedValue({ text: 'A thoughtful pick.', model: 'test-model' });
    render(<ResultsPanel results={results} profile={initialPreferences()} criticAvailable />);
    fireEvent.click(screen.getAllByRole('button', { name: /Ask the critic/ })[0]);
    expect(await screen.findByText('A thoughtful pick.')).toBeInTheDocument();
    expect(spy).toHaveBeenCalledWith(1, results.profile_summary, ['Story']);
  });
});
