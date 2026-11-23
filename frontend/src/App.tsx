import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { ArrowDown, ArrowUpRight, Check, Clapperboard, Compass, Heart, LoaderCircle, Orbit, ShieldCheck, Shuffle, SlidersHorizontal, Sparkles, Sun, Zap } from 'lucide-react';
import { getCatalog, getHealth, getRecommendations } from './api';
import { criteria, labels, methods, releasePeriods } from './types';
import type { Catalog, Health, Method, Preferences, ReleasePeriod, Results } from './types';
import MoviePoster from './MoviePoster';
import CollectionDialog from './CollectionDialog';
import { moods } from './moods';

export function initialPreferences(): Preferences {
  return {
    description: '', themes: '', mood: '', narrative_style: '', setting: '',
    preferred_genres: [], excluded_genres: [], excluded_themes: [], seen_movie_ids: [],
    importance: { description: 3, themes: 3, mood: 3, narrative_style: 3, setting: 3, genres: 3 },
    max_runtime: null, release_period: 'any', method: 'sbert_criteria', save_response: false,
  };
}

const emptyCatalog: Catalog = { movies: [], genres: [], themes: [] };
const ResultsPanel = lazy(() => import('./ResultsPanel'));
const moodIcons = { orbit: Orbit, heart: Heart, sun: Sun, zap: Zap, compass: Compass, sparkles: Sparkles };
const featured = [{ id: 27205, title: 'Inception' }, { id: 152601, title: 'Her' }, { id: 157336, title: 'Interstellar' }];

export default function App() {
  const [profile, setProfile] = useState<Preferences>(initialPreferences);
  const [catalog, setCatalog] = useState<Catalog>(emptyCatalog);
  const [health, setHealth] = useState<Health | null>(null);
  const [connectionError, setConnectionError] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<Results | null>(null);
  const [submittedProfile, setSubmittedProfile] = useState<Preferences | null>(null);
  const [selectedMood, setSelectedMood] = useState<string | null>(null);
  const [collectionOpen, setCollectionOpen] = useState(false);
  const [quickActionVisible, setQuickActionVisible] = useState(false);
  const resultsRef = useRef<HTMLDivElement>(null);
  const formRef = useRef<HTMLFormElement>(null);

  useEffect(() => {
    let active = true;
    Promise.all([getHealth(), getCatalog()]).then(([status, films]) => {
      if (active) { setHealth(status); setCatalog(films); }
    }).catch(() => {
      if (active) setConnectionError('The film service is not available. Please try again when it is running.');
    });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    const updateQuickAction = () => {
      const form = formRef.current?.getBoundingClientRect();
      const submit = formRef.current?.querySelector('.search-button')?.getBoundingClientRect();
      setQuickActionVisible(!!form && !!submit && form.top < window.innerHeight * 0.8 && submit.top > window.innerHeight - 80);
    };
    window.addEventListener('scroll', updateQuickAction, { passive: true });
    window.addEventListener('resize', updateQuickAction);
    updateQuickAction();
    return () => { window.removeEventListener('scroll', updateQuickAction); window.removeEventListener('resize', updateQuickAction); };
  }, []);

  function update<K extends keyof Preferences>(key: K, value: Preferences[K]) {
    setProfile(previous => ({ ...previous, [key]: value }));
    if (key === 'mood') setSelectedMood(null);
  }

  function focusStory() {
    document.getElementById('discover')?.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
    document.getElementById('description')?.focus({ preventScroll: true });
  }

  function applyMood(index: number, replaceStory = false) {
    const preset = moods[index];
    setSelectedMood(preset.name);
    setProfile(previous => ({
      ...previous,
      description: replaceStory || !previous.description.trim() ? preset.description : previous.description,
      themes: replaceStory || !previous.themes.trim() ? preset.themes : previous.themes,
      mood: preset.mood,
      narrative_style: replaceStory || !previous.narrative_style.trim() ? preset.narrative_style : previous.narrative_style,
      preferred_genres: replaceStory || !previous.preferred_genres.length
        ? preset.genres.filter(genre => catalog.genres.includes(genre)) : previous.preferred_genres,
    }));
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setError('');
    const overlap = profile.preferred_genres.filter(genre => profile.excluded_genres.includes(genre));
    if (overlap.length) {
      setError('Please clarify your genres: ' + overlap.join(', ') + ' cannot be preferred and excluded.');
      return;
    }
    setLoading(true);
    setResults(null);
    try {
      const submitted = structuredClone(profile);
      const response = await getRecommendations(submitted);
      setSubmittedProfile(submitted);
      setResults(response);
      setTimeout(() => resultsRef.current?.scrollIntoView?.({ behavior: 'smooth', block: 'start' }), 80);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'The recommendations could not be loaded.');
    } finally { setLoading(false); }
  }

  const ready = !!health?.catalog_size && !!health?.model_available;
  const toggled = (values: string[], value: string) => values.includes(value)
    ? values.filter(item => item !== value) : [...values, value];
  const activeCount = [profile.description, profile.themes, profile.mood, profile.narrative_style, profile.setting, profile.preferred_genres.join(',')].filter(Boolean).length;

  return <>
    <a className="skip-link" href="#discover">Skip to the questionnaire</a>
    <header className="site-header"><a className="brand" href="#" aria-label="Movie Match home">
      <span className="brand-mark"><Clapperboard size={20} strokeWidth={1.8} /></span>movie<span>match</span><i>.</i></a>
      <nav aria-label="Main navigation"><a className="nav-active" href="#discover">Find a film</a>
        <button onClick={() => setCollectionOpen(true)}>The collection <ArrowUpRight size={13} /></button>
        <a href="#how-it-works">How it works</a></nav>
      <span className="catalog-status"><span className="status-dot" />{health ? health.catalog_size + ' films to discover' : 'Setting the scene'}</span>
    </header>

    <main>
      <section className="hero" aria-labelledby="hero-title">
        <div className="hero-copy"><p className="eyebrow"><span /> A LITTLE LESS SEARCHING. A LOT MORE CINEMA.</p>
          <h1 id="hero-title">Less scrolling.<br />More <span className="hero-word">feeling<svg viewBox="0 0 310 18" aria-hidden="true"><path d="M3 12 Q140 -2 305 8 M15 17 Q160 4 288 13" /></svg></span><span className="accent">.</span></h1>
          <p className="hero-description">The perfect movie night starts with a feeling.<br className="desktop-break" /> Tell us yours. We'll help you find the story.</p>
          <div className="hero-actions"><button className="primary-button hero-cta" onClick={focusStory}>Find my next film <ArrowUpRight size={18} /></button>
            <button className="text-button" onClick={() => { applyMood(Math.floor(Math.random() * moods.length), true); focusStory(); }}><Shuffle size={15} /> Surprise me</button></div>
          <div className="hero-footnote"><span className="small-stars">✦ ✦ ✦</span><span>Three thoughtful picks. All about your taste.</span></div>
        </div>
        <div className="hero-art" aria-label="Films from the collection">
          <div className="hero-orbit orbit-one" /><div className="hero-orbit orbit-two" />
          <span className="art-cross cross-one">+</span><span className="art-cross cross-two">+</span>
          <div className="poster-fan">{featured.map((movie, index) => <div className={'fan-film fan-film-' + index} key={movie.id}>
            <MoviePoster id={movie.id} title={movie.title} eager /><span>{movie.title}</span></div>)}</div>
          <div className="floating-note"><span><Sparkles size={17} /></span><div>Good stories.<strong>Your kind of feeling.</strong></div></div>
          <div className="art-caption"><span className="status-dot" /> A FEW STORIES FROM OUR COLLECTION <ArrowDown size={12} /></div>
        </div>
      </section>

      <section className="mood-section" aria-labelledby="mood-title">
        <div className="mood-heading"><div><span className="section-index">01 / SET THE TONE</span><h2 id="mood-title">Start with a feeling.</h2></div><p>Choose a mood, or write your own story below.</p></div>
        <div className="mood-grid">{moods.map((mood, index) => {
          const Icon = moodIcons[mood.icon];
          return <button className={'mood-tile ' + (selectedMood === mood.name ? 'selected' : '')} key={mood.name}
            aria-pressed={selectedMood === mood.name} onClick={() => applyMood(index)}>
            <span className={'mood-icon mood-icon-' + index}><Icon size={23} strokeWidth={1.4} /></span>
            <strong>{mood.name}</strong><small>{mood.note}</small><span className="mood-check">{selectedMood === mood.name ? <Check size={13} /> : <ArrowUpRight size={13} />}</span>
          </button>;
        })}</div>
      </section>

      {connectionError && <p className="notice" role="alert">{connectionError}</p>}
      {health && !health.catalog_size && <p className="notice" role="status">The film catalog is not ready yet. Please return when the movie data is available.</p>}
      {health && health.catalog_size > 0 && !health.model_available && <p className="notice" role="status">The recommendation model is being prepared. Please return when it is available.</p>}

      <section id="discover" className="discover-section" aria-labelledby="discover-title">
        <div className="discover-heading"><div><span className="section-index">02 / MAKE IT YOURS</span><h2 id="discover-title">Your story. Your rules<span className="accent">.</span></h2></div><span className="optional-note">A few words are all it takes.</span></div>
        <form ref={formRef} onSubmit={submit} className="questionnaire" aria-busy={loading}>
          <div className="form-main">
            <div className="story-label"><label htmlFor="description">Describe the kind of movie you would like to watch <span className="required">*</span></label><span><span className="status-dot" /> ENGLISH</span></div>
            <div className="story-input"><span className="input-quote" aria-hidden="true">“</span>
              <textarea id="description" required minLength={12} maxLength={4000} rows={4}
                value={profile.description} onChange={event => update('description', event.target.value)}
                onKeyDown={event => { if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) { event.preventDefault(); formRef.current?.requestSubmit(); } }}
                placeholder="Something like a journey through space, but really about the people we leave behind..." />
              <div className="story-input-footer"><span>Be specific. Be vague. Be you.</span><span>{profile.description.length}/4000</span></div>
            </div>
            <p className="help">Write a short sentence in English. Optional fields can stay empty. Put things you want to avoid in the exclusions below.</p>
            <div className="form-divider"><span>A LITTLE MORE CONTEXT</span><i /></div>
            <div className="two-columns">
              <div><label htmlFor="themes">Which themes would you like to explore?</label>
                <textarea id="themes" rows={2} maxLength={1500} value={profile.themes}
                  onChange={event => update('themes', event.target.value)} placeholder="Identity, unlikely friendships, second chances..." /></div>
              <div><label htmlFor="mood">What atmosphere are you looking for?</label>
                <textarea id="mood" rows={2} maxLength={1000} value={profile.mood}
                  onChange={event => update('mood', event.target.value)} placeholder="Quietly moving, a little strange, full of hope..." /></div>
              <div><label htmlFor="narrative-style">How should the story unfold?</label>
                <textarea id="narrative-style" rows={2} maxLength={1000} value={profile.narrative_style}
                  onChange={event => update('narrative_style', event.target.value)} placeholder="A slow burn, a twisty mystery, a grand adventure..." /></div>
              <div><label htmlFor="setting">A preferred place or historical period?</label>
                <textarea id="setting" rows={2} maxLength={1000} value={profile.setting}
                  onChange={event => update('setting', event.target.value)} placeholder="Tokyo at night, a distant future, a small town..." /></div>
            </div>
            <fieldset className="genre-field"><legend>Which genres do you enjoy?</legend><p className="help">Pick a few, or leave them all open.</p>
              <div className="genre-options">{catalog.genres.map(genre => <label className="genre-choice" key={genre}>
                <input type="checkbox" checked={profile.preferred_genres.includes(genre)}
                  onChange={() => update('preferred_genres', toggled(profile.preferred_genres, genre))} /><span>{profile.preferred_genres.includes(genre) && <Check size={12} />}{genre}</span></label>)}</div>
            </fieldset>
            <div className="two-columns choice-row">
              <div><label htmlFor="release-period">Which era should the film come from?</label>
                <select id="release-period" value={profile.release_period}
                  onChange={event => update('release_period', event.target.value as ReleasePeriod)}>
                  {releasePeriods.map(period => <option key={period.value} value={period.value}>{period.label}</option>)}</select>
                <p className="help">A strict filter on the release year.</p></div>
              <div><label htmlFor="method">How should we compare your words with the films?</label>
                <select id="method" value={profile.method} onChange={event => update('method', event.target.value as Method)}>
                  {methods.map(method => <option key={method.value} value={method.value}>{method.label}</option>)}</select>
                <p className="help">{methods.find(method => method.value === profile.method)?.note}</p></div>
            </div>
            <details className="constraints"><summary><span><SlidersHorizontal size={17} /> Set things you want to avoid</span><span className="summary-plus">+</span></summary>
              <p className="help">Your limits come first. We keep these filters even if fewer than three films remain.</p>
              <label htmlFor="max-runtime">Maximum duration in minutes</label>
              <input id="max-runtime" type="number" min={1} max={600} value={profile.max_runtime ?? ''}
                placeholder="No limit" onChange={event => update('max_runtime', event.target.value ? Number(event.target.value) : null)} />
              <fieldset><legend>Excluded genres</legend><div className="genre-options">{catalog.genres.map(genre =>
                <label className="genre-choice" key={genre}><input type="checkbox" checked={profile.excluded_genres.includes(genre)}
                  onChange={() => update('excluded_genres', toggled(profile.excluded_genres, genre))} /><span>{genre}</span></label>)}</div></fieldset>
              <label htmlFor="excluded-themes">Avoid these annotated topics</label>
              <select id="excluded-themes" multiple size={4} value={profile.excluded_themes}
                onChange={event => update('excluded_themes', Array.from(event.target.selectedOptions, option => option.value))}>
                {catalog.themes.map(theme => <option key={theme} value={theme}>{theme}</option>)}</select>
              <p className="help">Use Ctrl or Command for several topics. Topic annotations cannot cover every scene.</p>
              <label htmlFor="seen-films">Films you have already watched</label>
              <select id="seen-films" multiple size={5} value={profile.seen_movie_ids.map(String)}
                onChange={event => update('seen_movie_ids', Array.from(event.target.selectedOptions, option => Number(option.value)))}>
                {catalog.movies.map(movie => <option key={movie.tmdb_id} value={movie.tmdb_id}>{movie.title}</option>)}</select>
            </details>
          </div>
          <aside className="form-aside">
            <div className="aside-icon"><SlidersHorizontal size={20} /></div><span className="section-index">THE DIRECTOR'S CHAIR</span><h3>A little more<br />you.</h3>
            <p className="help">Turn up what matters. Leave the rest to the story.</p>
            <div className="importance-list">{criteria.map(criterion => <div className="importance" key={criterion}>
              <label htmlFor={'importance-' + criterion}>{labels[criterion]}<span>{profile.importance[criterion]}/5</span></label>
              <input id={'importance-' + criterion} type="range" min={1} max={5} value={profile.importance[criterion]}
                style={{ '--range-fill': ((profile.importance[criterion] - 1) / 4 * 100) + '%' } as React.CSSProperties}
                onChange={event => update('importance', { ...profile.importance, [criterion]: Number(event.target.value) })} />
            </div>)}</div>
            <div className="taste-status"><span className="status-dot" />{activeCount ? activeCount + ' preferences in the picture' : 'Waiting for your first words'}</div>
            <button className="primary-button search-button" type="submit" disabled={loading || !ready}>
              {loading ? <><LoaderCircle size={17} className="spinner" /> Finding your films...</> : <>Find my films <ArrowUpRight size={18} /></>}</button>
            <span className="shortcut-hint">or press <kbd>Ctrl</kbd> + <kbd>Enter</kbd></span>
            <label className="save-choice"><input type="checkbox" checked={profile.save_response}
              onChange={event => update('save_response', event.target.checked)} />
              <span>Save my answers to the local study log</span></label>
            <p className="privacy-note"><ShieldCheck size={14} /> {profile.save_response
              ? 'Your answers and picks will be added to a JSON file on this computer.'
              : 'Your answers stay in this search unless you choose to save or export them.'}</p>
            {error && <p className="error" role="alert">{error}</p>}
          </aside>
        </form>
      </section>

      <div ref={resultsRef} className="results-anchor"><Suspense fallback={<p role="status" className="loading-message">Setting the scene...</p>}>
        {results && submittedProfile && <ResultsPanel results={results} profile={submittedProfile} criticAvailable={!!health?.critic_available} />}
      </Suspense></div>

      <section className="how-section" id="how-it-works" aria-labelledby="how-title">
        <div><span className="section-index">THOUGHTFUL PICKS. NOTHING RANDOM.</span><h2 id="how-title">A good match<br />has a reason<span className="accent">.</span></h2><p>There is more to a film than its genre.</p></div>
        <div className="how-steps"><article><span>01</span><h3>You set the scene.</h3><p>Your words, your mood, your must-haves. Tell us what a good film means tonight.</p></article>
          <article><span>02</span><h3>We connect the stories.</h3><p>Your expectations are compared with the stories, themes and atmosphere in our collection.</p></article>
          <article><span>03</span><h3>You find your next watch.</h3><p>Three films, with a clear explanation of what makes each one a match.</p></article></div>
      </section>
      {!!health?.estimated_annotations && <p className="annotation-note">A note on our collection: atmosphere and storytelling labels are estimated from source descriptions. Unavailable attributes are left out of the score.</p>}
    </main>
    <footer><a className="brand footer-brand" href="#">movie<span>match</span><i>.</i></a><p>For the love of finding a good film.</p>
      <details><summary>Data & credits <ArrowUpRight size={12} /></summary>
        <img src="/tmdb-logo.svg" width="90" alt="TMDB" className="credits-logo" />
        <p>This product uses the TMDB API but is not endorsed or certified by TMDB.</p>
        <p>Film data and locally cached posters: <a href="https://www.themoviedb.org/" target="_blank" rel="noreferrer">TMDB</a>. ING5 NLP Project.</p>
      </details></footer>
    <CollectionDialog open={collectionOpen} close={() => setCollectionOpen(false)} catalog={catalog} />
    {quickActionVisible && <div className="mobile-action"><span>Your movie night<br /><strong>is a few words away.</strong></span>
      <button className="primary-button" onClick={() => formRef.current?.requestSubmit()} disabled={loading || !ready}>
        {loading ? <LoaderCircle size={16} className="spinner" /> : <>Get my picks <ArrowUpRight size={16} /></>}</button></div>}
  </>;
}
