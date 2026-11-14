import type { Catalog, Health, Preferences, Results } from './types';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api/${path}`, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = body.detail;
    const message = Array.isArray(detail)
      ? detail.map((item: { msg: string }) => item.msg).join(' ')
      : typeof detail === 'string' ? detail : 'The request could not be completed.';
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const getHealth = () => request<Health>('health');
export const getCatalog = () => request<Catalog>('catalog');
export const getRecommendations = (preferences: Preferences) => request<Results>('recommend', {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(preferences),
});

export function downloadPreferences(profile: Preferences): void {
  const blob = new Blob([JSON.stringify(profile, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'movie-preferences.json';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
