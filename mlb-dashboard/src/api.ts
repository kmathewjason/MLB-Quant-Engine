/**
 * Axios API client for the MLB Quant Engine backend.
 * All URLs are relative — Vite's dev proxy forwards /api/* to FastAPI.
 *
 * Timeouts
 * --------
 * DAILY_TIMEOUT_MS  : 25 s — covers schedule fetch + 15 games × simulation.
 *                     No retries because the odds call is now try_once on the
 *                     backend, so total wall time is ~2 s without odds or
 *                     ~3 s with a live odds hit.
 * SIM_TIMEOUT_MS    : 30 s — individual game deep-sim (50 k paths).
 * DEFAULT_TIMEOUT_MS: 10 s — everything else.
 */
import axios from 'axios'
import type {
  Envelope,
  GameStub,
  BoardData,
  DailyGame,
  GameSimData,
  SGPLeg,
  SGPData,
  BacktestData,
  GameLegsData,
  ParlayLegInput,
  ParlayEvalResult,
  SuggestedParlaysData,
} from './types'

const DAILY_TIMEOUT_MS  = 25_000
const SIM_TIMEOUT_MS    = 30_000
const DEFAULT_TIMEOUT_MS = 10_000

const client = axios.create({
  baseURL: '/',
  timeout: DEFAULT_TIMEOUT_MS,
})

// ── helpers ────────────────────────────────────────────────────────────────

function apiError(err: unknown): never {
  if (axios.isAxiosError(err)) {
    if (err.code === 'ECONNABORTED') throw new Error('Request timed out — is the API server running?')
    const detail = (err.response?.data as { detail?: string })?.detail
    throw new Error(detail ?? err.message)
  }
  throw err
}

// ── endpoints ──────────────────────────────────────────────────────────────

/**
 * Fetch today's game list (schedule only — no simulation, no odds).
 * Designed to respond in <1 s.
 */
export async function fetchGamesToday(date?: string): Promise<Envelope<GameStub[]>> {
  try {
    const { data } = await client.get<Envelope<GameStub[]>>('/api/games/today', {
      params: date ? { date } : {},
      timeout: 8_000,
    })
    return data
  } catch (e) {
    console.error('[fetchGamesToday] error:', e)
    return apiError(e)
  }
}

/**
 * Fetch the full market board for a single game (simulation + odds + props).
 * May take 5-30 s depending on n_sims and whether odds API is reachable.
 */
export async function fetchGameBoard(
  gameId: number,
  opts: { bankroll?: number; devig?: string; nSims?: number; nBoot?: number } = {},
): Promise<Envelope<BoardData>> {
  try {
    const { data } = await client.get<Envelope<BoardData>>(
      `/api/games/${gameId}/board`,
      {
        params: {
          ...(opts.bankroll !== undefined ? { bankroll: opts.bankroll } : {}),
          ...(opts.devig    !== undefined ? { devig:    opts.devig    } : {}),
          ...(opts.nSims    !== undefined ? { n_sims:   opts.nSims    } : {}),
          ...(opts.nBoot    !== undefined ? { n_boot:   opts.nBoot    } : {}),
        },
        timeout: SIM_TIMEOUT_MS,
      },
    )
    return data
  } catch (e) {
    console.error('[fetchGameBoard] error:', e)
    return apiError(e)
  }
}

/**
 * Fetch today's (or a specific date's) slate of games with simulation data.
 * Uses n_sims=5000 on the initial load for speed; user can re-run with more.
 */
export async function fetchDailyPredictions(
  date?: string,
  nSims = 5_000,
): Promise<Envelope<DailyGame[]>> {
  try {
    const { data } = await client.get<Envelope<DailyGame[]>>('/api/predictions/daily', {
      params: { n_sims: nSims, ...(date ? { date } : {}) },
      timeout: DAILY_TIMEOUT_MS,
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
          n_sims: opts.nSims ?? 20_000,
          ...(opts.totalLine !== undefined ? { total_line: opts.totalLine } : {}),
          ...(opts.runLine  !== undefined ? { run_line:   opts.runLine  } : {}),
          ...(opts.bins     !== undefined ? { bins:       opts.bins     } : {}),
        },
        timeout: SIM_TIMEOUT_MS,
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
  nSims = 10_000,
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

export async function fetchGameLegs(
  gameId: number,
  opts: { bankroll?: number; nSims?: number } = {},
): Promise<Envelope<GameLegsData>> {
  try {
    const { data } = await client.get<Envelope<GameLegsData>>(
      `/api/games/${gameId}/legs`,
      {
        params: {
          ...(opts.bankroll !== undefined ? { bankroll: opts.bankroll } : {}),
          ...(opts.nSims    !== undefined ? { n_sims:   opts.nSims    } : {}),
        },
        timeout: SIM_TIMEOUT_MS,
      },
    )
    return data
  } catch (e) {
    return apiError(e)
  }
}

export async function fetchEvaluateParlay(
  legs: ParlayLegInput[],
  bankroll = 1000,
  nSims = 10_000,
): Promise<Envelope<ParlayEvalResult>> {
  try {
    const { data } = await client.post<Envelope<ParlayEvalResult>>(
      '/api/parlays/evaluate',
      { legs, bankroll, n_sims: nSims },
      { timeout: SIM_TIMEOUT_MS },
    )
    return data
  } catch (e) {
    return apiError(e)
  }
}

export async function fetchSuggestedParlays(
  mode: 'sgp' | 'crossgame',
  bankroll = 1000,
  opts: { date?: string; nSims?: number } = {},
): Promise<Envelope<SuggestedParlaysData>> {
  try {
    const { data } = await client.get<Envelope<SuggestedParlaysData>>(
      '/api/parlays/suggested',
      {
        params: {
          mode,
          bankroll,
          ...(opts.date  ? { date:   opts.date  } : {}),
          ...(opts.nSims ? { n_sims: opts.nSims } : {}),
        },
        timeout: SIM_TIMEOUT_MS,
      },
    )
    return data
  } catch (e) {
    return apiError(e)
  }
}

export async function fetchBacktestReport(): Promise<Envelope<BacktestData>> {
  try {
    const { data } = await client.get<Envelope<BacktestData>>('/api/backtest/report', {
      timeout: DEFAULT_TIMEOUT_MS,
    })
    return data
  } catch (e) {
    return apiError(e)
  }
}
