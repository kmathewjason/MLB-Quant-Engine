/** Shared TypeScript types mirroring the FastAPI response shapes. */

export interface Meta {
  generated_at: string
  version: string
}

export interface Envelope<T> {
  data: T
  meta: Meta
}

// ── /api/predictions/daily ─────────────────────────────────────────────────

export interface RunDist {
  mean: number
  std: number
  p10: number
  p25: number
  p50: number
  p75: number
  p90: number
}

export interface SimulationBlock {
  n_sims: number
  home_win_prob: number
  away_win_prob: number
  home_runs: RunDist
  away_runs: RunDist
  total_runs: RunDist
  spread: Record<string, number>
  totals: Record<string, number>
}

export interface MarketRow {
  market: string
  label: string
  model_prob: number
  market_prob: number | null
  market_odds: number | null
  edge: number | null
  ev: number | null
  kelly_full: number | null
  kelly_quarter: number | null
  stake_units: number | null
}

export interface DailyGame {
  game_pk: number
  game_date: string
  away_team: string
  home_team: string
  away_probable_pitcher: string
  home_probable_pitcher: string
  status: string
  simulation: SimulationBlock
  markets: MarketRow[]
}

// ── /api/games/{id}/simulation ─────────────────────────────────────────────

export interface Histogram {
  edges: number[]
  counts: number[]
}

export interface DistWithHist extends RunDist {
  histogram: Histogram
}

export interface MarginDist {
  histogram: Histogram
  mean: number
  std: number
}

export interface GameSimData {
  game_id: number
  n_sims: number
  home_win_prob: number
  away_win_prob: number
  spread_cover_prob: number
  over_prob: number
  under_prob: number
  home_runs: DistWithHist
  away_runs: DistWithHist
  total_runs: DistWithHist
  margin_dist: MarginDist
}

// ── /api/predictions/sgp ───────────────────────────────────────────────────

export interface SGPLeg {
  side: 'home' | 'away'
  outcome: 'moneyline' | 'over' | 'under' | 'spread'
  line?: number
  odds?: number
  label?: string
}

export interface SGPLegResult {
  label?: string
  side: string
  outcome: string
  line?: number | null
  odds?: number | null
  sim_prob: number
  naive_prob: number
}

export interface KellyVariant {
  ev: number
  f_star: number
  f_quarter: number
  stake_units: number
}

export interface SGPData {
  game_id: number
  n_sims: number
  legs: SGPLegResult[]
  joint_prob_corr_adjusted: number
  joint_prob_naive: number
  parlay_net_payout: number
  correlation_matrix: number[][]
  kelly: { corr_adjusted: KellyVariant; naive: KellyVariant }
}

// ── /api/backtest/report ───────────────────────────────────────────────────

export interface WFSummary {
  n_folds: number
  n_oos_predictions: number
  oos_log_loss: number
  oos_accuracy: number
  mean_fold_log_loss: number
  std_fold_log_loss: number
  mean_fold_brier: number
}

export interface FoldMetric {
  fold: number
  date_start: string
  date_end: string
  n_train: number
  n_test: number
  log_loss: number
  accuracy: number
  brier: number
}

export interface BrierDecomposition {
  brier_score: number
  reliability: number
  resolution: number
  uncertainty: number
  brier_skill_score: number
}

export interface ReliabilityBin {
  bin_centre: number
  mean_pred: number
  obs_freq: number
  count: number
  ci_lo_95: number
  ci_hi_95: number
}

export interface ReliabilityCurve {
  class_name: string
  bins: ReliabilityBin[]
  ece: number
}

export interface CalibrationReport {
  report: Record<string, number | string>[]
  brier_decomposition?: BrierDecomposition
  reliability_curves?: ReliabilityCurve[]
  available: boolean
}

export interface CLVSummary {
  n_bets: number
  mean_clv_prob: number
  mean_clv_log_odds: number
  pct_positive_clv: number
  clv_tstat: number
  clv_pvalue: number
  clv_result_corr: number
  roi_pct: number
  total_pnl_units: number
}

export interface BacktestData {
  walk_forward: { summary?: WFSummary; fold_metrics?: FoldMetric[]; available: boolean }
  calibration: CalibrationReport
  clv: { summary?: CLVSummary; available: boolean }
}
