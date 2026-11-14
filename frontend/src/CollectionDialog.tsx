import { useEffect, useRef, useState } from 'react';
import { ArrowUpRight, Search, X } from 'lucide-react';
import type { Catalog } from './types';
import MoviePoster from './MoviePoster';

export default function CollectionDialog({ open, close, catalog }: {
  open: boolean; close: () => void; catalog: Catalog;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const [search, setSearch] = useState('');
  useEffect(() => {
    if (open) {
      if (ref.current?.showModal) ref.current.showModal();
      else ref.current?.setAttribute('open', '');
      ref.current?.querySelector('input')?.focus();
    } else if (ref.current?.open) {
      if (ref.current.close) ref.current.close();
      else ref.current.removeAttribute('open');
    }
  }, [open]);
  const normalise = (text: string) => text.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
  const movies = catalog.movies.filter(movie => normalise(movie.title).includes(normalise(search)));

  return <dialog className="collection-dialog" ref={ref} onCancel={close}
    onClick={event => { if (event.target === event.currentTarget) close(); }} aria-labelledby="collection-title">
    <div className="dialog-heading"><div><p className="eyebrow">A SMALL COLLECTION. A WIDE WORLD.</p>
      <h2 id="collection-title">The collection<span className="accent">.</span></h2></div>
      <button className="icon-button" onClick={close} aria-label="Close collection"><X size={22} /></button></div>
    <div className="collection-search"><Search size={18} /><input aria-label="Search the collection"
      placeholder="Find a title you love..." value={search} onChange={event => setSearch(event.target.value)} />
      <span>{movies.length} {movies.length === 1 ? 'film' : 'films'}</span></div>
    <div className="collection-grid">{movies.map(movie => <a className="collection-film" key={movie.tmdb_id}
      href={'https://www.themoviedb.org/movie/' + movie.tmdb_id} target="_blank" rel="noreferrer">
      <MoviePoster id={movie.tmdb_id} title={movie.title} /><span>{movie.title}<ArrowUpRight size={13} /></span></a>)}</div>
    {movies.length === 0 && <p className="empty-collection">No title matches this search.</p>}
  </dialog>;
}
