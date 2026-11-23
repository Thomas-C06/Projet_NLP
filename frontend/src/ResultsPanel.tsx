import { Bar, BarChart, CartesianGrid, Legend, PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { ArrowDownToLine, ArrowRight, ArrowUpRight, Check, Film, SlidersHorizontal, Sparkles } from 'lucide-react';
import { downloadPreferences } from './api';
import { methods } from './types';
import type { Preferences, Results } from './types';
import MoviePoster from './MoviePoster';
import CriticPitch from './CriticPitch';

const colours = ['#d4fa82', '#b9a4f4', '#77cec0'];
const tooltipStyle = { background: '#1b1f1a', border: '1px solid #363d31', borderRadius: 10, color: '#f0f3e8', fontSize: 12 };

export default function ResultsPanel({ results, profile, criticAvailable = false }: { results: Results; profile: Preferences; criticAvailable?: boolean }) {
  const movies = results.recommendations;
  const totals = movies.map(({ movie, affinity }) => ({ title: movie.title, affinity: Number(affinity.toFixed(1)) }));
  const criterionLabels = [...new Set(movies.flatMap(item => item.criteria.map(criterion => criterion.label)))];
  const byCriterion = criterionLabels.map(label => Object.fromEntries([
    ['criterion', label], ...movies.map(({ movie, criteria }) => [
      'movie_' + movie.tmdb_id, criteria.find(item => item.label === label)?.similarity === undefined ? null
        : Number((criteria.find(item => item.label === label)!.similarity * 100).toFixed(1)),
    ]),
  ]));
  // A radar needs at least three axes to show a profile shape.
  const showRadar = byCriterion.length >= 3;
  const methodLabel = methods.find(method => method.value === results.method)?.label ?? results.method;
  const refine = () => {
    document.getElementById('discover')?.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
    document.getElementById('description')?.focus({ preventScroll: true });
  };

  return <section className="results" aria-labelledby="results-title">
    <div className="results-intro"><div><span className="section-index">03 / THE SHORTLIST</span>
      <h2 id="results-title">Stories worth exploring<span className="accent">.</span></h2>
      <p>Three different stories. A little bit of you in each one.</p></div>
      <div className="result-actions"><button className="secondary-button" onClick={refine}><SlidersHorizontal size={15} /> Refine my taste</button>
        <button className="icon-button" aria-label="Export my answers" title="Export my answers" onClick={() => downloadPreferences(profile)}><ArrowDownToLine size={18} /></button></div>
    </div>
    <div className="profile-ribbon"><span className="ribbon-icon"><Sparkles size={18} /></span><div><span>THE FEELING YOU ASKED FOR</span><p>{profile.description}</p></div>
      <span className="eligible-count">{results.eligible_count}<small>films fit your rules</small></span></div>
    <details className="profile-details"><summary>Your full preference summary <ArrowDownToLine size={12} /></summary><p>{results.profile_summary}</p>
      <p className="help">Method: {methodLabel} · {results.latency_ms.toFixed(0)} ms</p></details>
    {results.saved && <p className="saved-note" role="status"><Check size={13} /> Your answers were saved to the local study log.</p>}
    {results.warnings.map(message => <p key={message} className="notice" role="status">{message}</p>)}
    {movies.length ? <>
      <div className="movie-grid">{movies.map(({ movie, affinity, criteria, explanation, missing_criteria, main_contributions }, index) =>
        <article className={'movie-card ' + (index === 0 ? 'first-match' : '')} key={movie.tmdb_id}>
          <div className="movie-art"><MoviePoster id={movie.tmdb_id} title={movie.title} className="poster" />
            <div className="poster-shade" /><span className="movie-position">0{index + 1}</span>
            {index === 0 && <span className="closest-label"><Sparkles size={12} /> YOUR CLOSEST MATCH</span>}
            <div className="poster-bottom"><div className="affinity"><strong>{affinity.toFixed(1)}</strong><span>/100<br />affinity</span></div>
              <a className="source-link" href={movie.source} target="_blank" rel="noreferrer" aria-label={'View ' + movie.title + ' on TMDB'}><ArrowUpRight size={19} /></a></div>
          </div>
          <div className="movie-body"><div className="movie-title-line"><h3>{movie.title}</h3><span>{movie.release_year ?? '—'}</span></div>
            <p className="metadata">{movie.runtime_minutes ? movie.runtime_minutes + ' min' : 'Duration unknown'}<span>·</span>{movie.genres.slice(0, 3).join(' / ')}</p>
            <p className="synopsis">{movie.overview ?? 'Synopsis unavailable.'}</p>
            <details className="full-synopsis"><summary>Read the full story <ArrowRight size={12} /></summary><p>{movie.overview ?? 'Synopsis unavailable.'}</p></details>
            <div className="match-reasons"><span>WHY IT CONNECTS</span><div>{main_contributions.length
              ? main_contributions.map(item => <span key={item}><Check size={11} />{item}</span>)
              : <span>No positive match</span>}</div></div>
            {criticAvailable && <CriticPitch tmdbId={movie.tmdb_id} summary={results.profile_summary} matches={main_contributions} />}
            <details className="match-details"><summary>Understand this match <span>+</span></summary>
              <p className="explanation">{explanation}</p>
              {criteria.map(item => <div className="criterion-detail" key={item.criterion}>
                <div><strong>{item.label}</strong><span>{(item.similarity * 100).toFixed(1)}<small>/100</small></span></div>
                <progress max={100} value={item.similarity * 100} aria-label={item.label + ' affinity'} />
                <p>{item.evidence}</p><small>Weight {(item.weight * 100).toFixed(1)}% · Contribution {item.contribution.toFixed(1)} points</small>
              </div>)}
              {missing_criteria.length > 0 && <p className="help">Missing descriptions: {missing_criteria.join(', ')}.</p>}
            </details>
          </div>
        </article>)}
      </div>
      <div className="analysis-section"><div className="analysis-heading"><div><span className="section-index">BEHIND THE MATCH</span><h3>See the connection.</h3></div><p>Affinity measures textual alignment, not the probability that you will enjoy a film.</p></div>
        <div className="charts-grid">
          <div className="chart-card"><h3>Overall affinity<span>01</span></h3><div className="chart">
            <ResponsiveContainer width="100%" height="100%"><BarChart data={totals} layout="vertical" margin={{ left: 8, right: 16 }} barSize={22}>
              <CartesianGrid stroke="#2b3128" strokeDasharray="3 3" horizontal={false} /><XAxis type="number" domain={[0, 100]} axisLine={false} tickLine={false} tick={{ fill: '#8e9987', fontSize: 11 }} />
              <YAxis dataKey="title" type="category" width={110} axisLine={false} tickLine={false} tick={{ fill: '#cbd2c3', fontSize: 11 }} />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: '#ffffff04' }} /><Bar dataKey="affinity" name="Affinity" fill={colours[0]} radius={[0, 5, 5, 0]} isAnimationActive={false} />
            </BarChart></ResponsiveContainer></div></div>
          {showRadar && <div className="chart-card"><h3>Match profile<span>02</span></h3><div className="chart radar-chart">
            <ResponsiveContainer width="100%" height="100%"><RadarChart data={byCriterion} outerRadius="72%">
              <PolarGrid stroke="#2f3a28" /><PolarAngleAxis dataKey="criterion" tick={{ fill: '#cbd2c3', fontSize: 11 }} />
              <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
              <Tooltip contentStyle={tooltipStyle} />
              {movies.map((item, index) => <Radar key={item.movie.tmdb_id} dataKey={'movie_' + item.movie.tmdb_id} name={item.movie.title}
                stroke={colours[index]} fill={colours[index]} fillOpacity={index === 0 ? 0.22 : 0.08} strokeWidth={index === 0 ? 2 : 1.2} isAnimationActive={false} />)}
            </RadarChart></ResponsiveContainer></div><p className="help">Semantic similarity per criterion, from 0 at the centre to 100 at the edge.</p></div>}
          <div className={'chart-card ' + (showRadar ? 'chart-wide' : '')}><h3>Affinity by criterion<span>{showRadar ? '03' : '02'}</span></h3><div className="chart" style={{ height: Math.max(250, byCriterion.length * 65) }}>
            <ResponsiveContainer width="100%" height="100%"><BarChart data={byCriterion} layout="vertical" margin={{ right: 16 }}>
              <CartesianGrid stroke="#2b3128" strokeDasharray="3 3" horizontal={false} /><XAxis type="number" domain={[0, 100]} axisLine={false} tickLine={false} tick={{ fill: '#8e9987', fontSize: 11 }} />
              <YAxis type="category" dataKey="criterion" width={85} axisLine={false} tickLine={false} tick={{ fill: '#cbd2c3', fontSize: 10 }} />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: '#ffffff04' }} /><Legend wrapperStyle={{ fontSize: 11, paddingTop: 16 }} />
              {movies.map((item, index) => <Bar key={item.movie.tmdb_id} dataKey={'movie_' + item.movie.tmdb_id}
                name={item.movie.title} fill={colours[index]} radius={[0, 3, 3, 0]} isAnimationActive={false} />)}
            </BarChart></ResponsiveContainer></div><p className="help">Missing criteria appear as gaps.</p></div>
        </div>
      </div>
    </> : <div className="empty-results"><Film size={36} strokeWidth={1} /><h3>A little too specific for this collection.</h3><p>No film fits all your current rules. Your exclusions have been kept.</p>
      <button className="secondary-button" onClick={refine}>Adjust my preferences <ArrowUpRight size={16} /></button></div>}
  </section>;
}
