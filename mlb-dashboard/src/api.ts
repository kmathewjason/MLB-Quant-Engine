/**
 * Thin fetch helpers for the MLB Quant Engine API.
 * All requests are relative — Vite proxies them to the FastAPI backend.
 */

import type {
  Envelope,
  DailyGame,
  GameSimData,
  SGPLeg,
  SGPData,
  BacktestData,
} from './types';

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const b = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText}: ${b}`);
  }
  return res.json() as Promise<T>;
}

export async function fetchDailyPredictions(
  date?: string,
  nSims = 20_000,
): Promise<Envelope<DailyGame[]>> {
  const params = new URLSearchParams({ n_sims: String(nSims) });
  if (date) params.set('date', date);
  return get<Envelope<DailyGame[]>>(`/api/predictions/daily?${params}`);
}

export async function fetchGameSimulation(
  gameId: number,
  opts: { nSims?: number; totalLine?: number; runLine?: number; bins?: number } = {},
): Promise<Envelope<GameSimData>> {
  const params = new URLSearchParams({ n_sims: String(opts.nSims ?? 50_000) });
  if (opts.totalLine !== undefined) params.set('total_line', String(opts.totalLine));
  if (opts.runLine !== undefined) params.set('run_line', String(opts.runLine));
  if (opts.bins !== undefined) params.set('bins', String(opts.bins));
  return get<Envelope<GameSimData>>(`/api/games/${gameId}/simulation?${params}`);
}

export async function fetchSGP(
  gameId: number,
  legs: SGPLeg[],
  bankroll = 1000,
  nSims = 20_000,
): Promise<Envelope<SGPData>> {
  return post<Envelope<SGPData>>('/api/predictions/sgp', {
    game_id: gameId,
    legs,
    bankroll,
    n_sims: nSims,
  });
}

export async function fetchBacktestReport(): Promise<Envelope<BacktestData>> {
  return get<Envelope<BacktestData>>('/api/backtest/report');
}
