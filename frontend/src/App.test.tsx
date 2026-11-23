import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import * as api from './api';

vi.mock('./api', () => ({ getHealth: vi.fn(), getCatalog: vi.fn(), getRecommendations: vi.fn(), getCriticPitch: vi.fn(), downloadPreferences: vi.fn() }));

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.getHealth).mockResolvedValue({ status: 'ok', catalog_size: 150, target_size: 150,
    model_available: true, corpus_ready: true, annotated_count: 150 });
  vi.mocked(api.getCatalog).mockResolvedValue({ movies: [], genres: ['Drama', 'Horror'], themes: [] });
});

describe('Questionnaire', () => {
  it('sends the text, priorities and constraints to the API', async () => {
    vi.mocked(api.getRecommendations).mockRejectedValue(new Error('Example error message'));
    render(<App />);
    const submit = screen.getByRole('button', { name: 'Find my films' });
    await waitFor(() => expect(submit).toBeEnabled());
    fireEvent.change(screen.getByLabelText(/Describe the kind/), { target: { value: 'I want a moving story about friendship.' } });
    fireEvent.change(screen.getByLabelText('Maximum duration in minutes'), { target: { value: '120' } });
    fireEvent.change(screen.getByLabelText(/Themes3\/5/), { target: { value: '5' } });
    fireEvent.click(submit);
    await waitFor(() => expect(api.getRecommendations).toHaveBeenCalledWith(expect.objectContaining({
      description: 'I want a moving story about friendship.', max_runtime: 120,
      importance: expect.objectContaining({ themes: 5 }), method: 'sbert_criteria',
    })));
    expect(await screen.findByRole('alert')).toHaveTextContent('Example error message');
  });

  it('sends the single-choice era, the comparison method and the storage consent', async () => {
    vi.mocked(api.getRecommendations).mockRejectedValue(new Error('Stop here'));
    render(<App />);
    const submit = screen.getByRole('button', { name: 'Find my films' });
    await waitFor(() => expect(submit).toBeEnabled());
    fireEvent.change(screen.getByLabelText(/Describe the kind/), { target: { value: 'I want a classic detective story.' } });
    fireEvent.change(screen.getByLabelText(/Which era/), { target: { value: 'before-1980' } });
    fireEvent.change(screen.getByLabelText(/How should we compare/), { target: { value: 'tfidf' } });
    expect(screen.getByText(/Shared words only/)).toBeVisible();
    fireEvent.click(screen.getByLabelText(/Save my answers/));
    fireEvent.click(submit);
    await waitFor(() => expect(api.getRecommendations).toHaveBeenCalledWith(expect.objectContaining({
      release_period: 'before-1980', method: 'tfidf', save_response: true,
    })));
  });

  it('asks for clarification when a genre is both preferred and excluded', async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Find my films' })).toBeEnabled());
    fireEvent.change(screen.getByLabelText(/Describe the kind/), { target: { value: 'I want a moving story about friendship.' } });
    const choices = screen.getAllByLabelText('Drama');
    choices.forEach(choice => fireEvent.click(choice));
    fireEvent.click(screen.getByRole('button', { name: 'Find my films' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('cannot be preferred and excluded');
    expect(api.getRecommendations).not.toHaveBeenCalled();
  });

  it('does not allow searching before the catalog is available', async () => {
    vi.mocked(api.getHealth).mockResolvedValue({ status: 'ok', catalog_size: 0, target_size: 150,
      model_available: false, corpus_ready: false, annotated_count: 0 });
    render(<App />);
    expect(await screen.findByRole('status')).toHaveTextContent('catalog is not ready');
    expect(screen.getByRole('button', { name: 'Find my films' })).toBeDisabled();
  });

  it('uses a mood as guidance while preserving an existing story and duration limit', async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Find my films' })).toBeEnabled());
    const story = screen.getByLabelText(/Describe the kind/);
    fireEvent.change(story, { target: { value: 'I want a film about a family reconnecting.' } });
    fireEvent.change(screen.getByLabelText('Maximum duration in minutes'), { target: { value: '110' } });
    fireEvent.click(screen.getByRole('button', { name: /Feel-good/ }));
    expect(story).toHaveValue('I want a film about a family reconnecting.');
    expect(screen.getByLabelText('Maximum duration in minutes')).toHaveValue(110);
    expect(screen.getByLabelText('What atmosphere are you looking for?')).toHaveValue('warm, uplifting and funny');
  });

  it('opens the collection and searches titles without requiring accents', async () => {
    vi.mocked(api.getCatalog).mockResolvedValue({ genres: [], themes: [], movies: [
      { tmdb_id: 194, title: 'Amélie' }, { tmdb_id: 152601, title: 'Her' },
    ] });
    render(<App />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Find my films' })).toBeEnabled());
    fireEvent.click(screen.getByRole('button', { name: /The collection/ }));
    expect(screen.getByRole('dialog')).toBeVisible();
    fireEvent.change(screen.getByLabelText('Search the collection'), { target: { value: 'Amelie' } });
    expect(screen.getByRole('link', { name: /Amélie/ })).toBeVisible();
    expect(screen.queryByRole('link', { name: /Her poster/ })).not.toBeInTheDocument();
  });
});
