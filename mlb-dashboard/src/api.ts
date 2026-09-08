/**
 * Axios API client for the MLB Quant Engine backend.
 * All URLs are relative — Vite's dev proxy forwards /api/* to FastAPI.
 */
import axios from 'axios'
import type {
  Envelope,
  DailyGame,
  GameSimData,
  SGPLeg,
  SGPData,
  BacktestData,
} from './types'

const client = axios.create({ baseURL: '/' })

// ── helpers ────────────────────────────────────────────────────────────────

function apiError(err: unknown): never {
  if (axios.isAxiosError(err)) {
    const detail = (err.response?.data as { detail?: string })?.detail
    throw new Error(detail ?? err.message)
  }
  throw err
}

// ── endpoints ──────────────────────────────────────────────────────────────

export async function fetchDailyPredictions(
  date?: string,
  nSims = 20_000,
): Promise<Envelope<DailyGame[]>> {
  try {
    const { data } = await client.get<Envelope<DailyGame[]>>('/api/predictions/daily', {
      params: { n_sims: nSims, ...(date ? { date } : {}) },
    })
    return data
  } catch (e) {
    return apiError(e)
  }
}

export async function fetchGameSimulation(
  gameId: number,
  opts: { nSims?: number; totalLine?: number; runLine?: number; bins?: number } = {},
): Promise<Envelope<GameSimData>> {
  try {
    const { data } = await client.get<Envelope<GameSimData>>(
      `/api/games/${gameId}/simulation`,
      {
        params: {
          n_sims: opts.nSims ?? 50_000,
          ...(opts.totalLine !== undefined ? { total_line: opts.totalLine } : {}),
          ...(opts.runLine  !== undefined ? { run_line:   opts.runLine  } : {}),
          ...(opts.bins     !== undefined ? { bins:       opts.bins     } : {}),
        },
      },
    )
    return data
  } catch (e) {
    return apiError(e)
  }
}

export async function fetchSGP(
  gameId: number,
  legs: SGPLeg[],
  bankroll = 1000,
  nSims = 20_000,
): Promise<Envelope<SGPData>> {
  try {
    const { data } = await client.post<Envelope<SGPData>>('/api/predictions/sgp', {
      game_id: gameId,
      legs,
      bankroll,
      n_sims: nSims,
    })
    return data
  } catch (e) {
    return apiError(e)
  }
}

export async function fetchBacktestReport(): Promise<Envelope<BacktestData>> {
  try {
    const { data } = await client.get<Envelope<BacktestData>>('/api/backtest/report')
    return data
  } catch (e) {
    return apiError(e)
  }
}
