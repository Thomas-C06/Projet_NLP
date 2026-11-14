import { useState } from 'react';
import { Film } from 'lucide-react';

export default function MoviePoster({ id, title, className = '', eager = false }: {
  id: number; title: string; className?: string; eager?: boolean;
}) {
  const [unavailable, setUnavailable] = useState(false);
  if (unavailable) return <div className={'poster-fallback ' + className}>
    <Film size={34} strokeWidth={1} /><span>{title}</span><small>Poster unavailable</small>
  </div>;
  return <img className={className} src={'/posters/' + id + '.jpg'} alt={title + ' poster'}
    loading={eager ? 'eager' : 'lazy'} onError={() => setUnavailable(true)} />;
}
