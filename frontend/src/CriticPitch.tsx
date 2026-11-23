import { useState } from 'react';
import { LoaderCircle, MessageSquareQuote } from 'lucide-react';
import { getCriticPitch } from './api';

// Optional generative explanation. The ranking itself never depends on it.
export default function CriticPitch({ tmdbId, summary, matches }: { tmdbId: number; summary: string; matches: string[] }) {
  const [state, setState] = useState<{ status: 'idle' | 'loading' | 'done' | 'error'; text: string; model?: string }>({ status: 'idle', text: '' });

  async function ask() {
    setState({ status: 'loading', text: '' });
    try {
      const answer = await getCriticPitch(tmdbId, summary, matches);
      setState({ status: 'done', text: answer.text, model: answer.model });
    } catch (failure) {
      setState({ status: 'error', text: failure instanceof Error ? failure.message : 'The critic is not available.' });
    }
  }

  if (state.status === 'done') return <blockquote className="critic-pitch"><p>{state.text}</p>
    <small>Written by a local language model ({state.model}) from the synopsis and the matched criteria.</small></blockquote>;
  return <div className="critic-action">
    <button type="button" className="secondary-button" onClick={ask} disabled={state.status === 'loading'}>
      {state.status === 'loading' ? <><LoaderCircle size={14} className="spinner" /> The critic is writing...</>
        : <><MessageSquareQuote size={14} /> Ask the critic</>}</button>
    {state.status === 'error' && <p className="help" role="alert">{state.text}</p>}
  </div>;
}
