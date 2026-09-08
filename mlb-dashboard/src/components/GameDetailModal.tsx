/**
 * GameDetailModal — full distribution view for a single game.
 * Opens when a game card is clicked.
 */
import { useState, useEffect } from 'react'
import { fetchGameSimulation } from '../api'
import type { DailyGame, GameSimData, MarketRow } from '../types'

const S = {
  label: {
    fontSize: 9, fontWeight: 700 as const, letterSpacing: '0.07em',
    color: '#5a6072', textTransform: 'uppercase' as const,
  },
}

// ── inline histogram bar chart ─────────────────────────────────────────────

function HistBar({ edges, counts, color, mean }: {
  edges: number[]; counts: number[]; color: string; mean: number
}) {
  const max = Math.max(...counts)
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 1, height: 48 }}>
      {counts.map((c, i) => {
        const edge = edges[i] ?? i
        const isMean = edges[i] <= mean && (edges[i + 1] ?? Infinity) > mean
        return (
          <div
            key={i}
            title={`${edge}–${edges[i + 1] ?? '+'}: ${c.toLocaleString()}`}
            style={{
              flex: 1, height: `${Math.max(2, (c / max) * 100)}%`,
              background: isMean ? '#fff' : color, borderRadius: '2px 2px 0 0',
              opacity: isMean ? 1 : 0.65,
              minWidth: 3,
            }}
          />
        )
      })}
    </div>
  )
}

// ── confidence meter ───────────────────────────────────────────────────────

function ConfMeter({ prob, label }: { prob: number; label: string }) {
  const pct = Math.round(prob * 100)
  const conf = Math.abs(pct - 50) // 0–50
  const bar = Math.min(100, conf * 2) // 0–100 scale
  const color = conf >= 15 ? '#34d399' : conf >= 8 ? '#fbbf24' : '#6b7280'

  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
        <span style={{ fontSize: 12.5, fontWeight: 600, color: '#d1d5db' }}>{label}</span>
        <span style={{ fontSize: 14, fontWeight: 800, color: '#fff' }}>{pct}%</span>
      </div>
      <div style={{ height: 7, background: '#1f2230', borderRadius: 6, overflow: 'hidden' }}>
        <div style={{
          width: `${pct}%`, height: '100%',
          background: `linear-gradient(90deg, #1e5fc2, ${color})`,
          borderRadius: 6, transition: 'width 0.5s',
        }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 3 }}>
        <span style={{ fontSize: 9, color: '#3d4455' }}>0%</span>
        <span style={{ fontSize: 9, color: color }}>
          Model confidence: {conf >= 15 ? 'HIGH' : conf >= 8 ? 'MEDIUM' : 'LOW'}
          {' '}({bar.toFixed(0)}%)
        </span>
        <span style={{ fontSize: 9, color: '#3d4455' }}>100%</span>
      </div>
    </div>
  )
}

// ── ranked market table ────────────────────────────────────────────────────

function MarketTable({ markets }: { markets: MarketRow[] }) {
  const ranked = [...markets].sort((a, b) => {
    // EV first (if available), then model confidence
    if (a.ev !== null && b.ev !== null) return b.ev - a.ev
    if (a.ev !== null) return -1
    if (b.ev !== null) return 1
    return Math.abs(b.model_prob - 0.5) - Math.abs(a.model_prob - 0.5)
  })

  return (
    <div style={{
      background: '#12141b', borderRadius: 10, border: '1px solid #1f2230',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '22px 1fr 52px 52px 52px 52px 56px',
        gap: 6, padding: '7px 12px',
        background: '#0f1117', borderBottom: '1px solid #1f2230',
      }}>
        {['#', 'Market', 'Model', 'Edge', 'EV', 'Odds', 'Kelly $'].map(h => (
          <span key={h} style={S.label}>{h}</span>
        ))}
      </div>

      {ranked.map((m, i) => {
        const rankColor = i === 0 ? '#fbbf24' : i === 1 ? '#9ca3af' : i === 2 ? '#cd7c2e' : '#3a4055'
        const evPos = (m.ev ?? 0) > 0.005
        const conf = Math.round(m.model_prob * 100)

        return (
          <div
            key={m.market}
            style={{
              display: 'grid',
              gridTemplateColumns: '22px 1fr 52px 52px 52px 52px 56px',
              gap: 6, padding: '8px 12px', alignItems: 'center',
              borderBottom: '1px solid #1a1d24',
              background: evPos ? '#0a1a12' : 'transparent',
            }}
          >
            {/* Rank badge */}
            <span style={{
              width: 18, height: 18, borderRadius: '50%',
              background: rankColor, color: i < 3 ? '#000' : '#6b7280',
              fontSize: 9, fontWeight: 700,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              {i + 1}
            </span>

            {/* Label */}
            <span style={{ fontSize: 12.5, color: '#d1d5db', fontWeight: 500 }}>
              {m.label}
              {evPos && (
                <span style={{
                  marginLeft: 6, fontSize: 8, background: '#0d3326',
                  color: '#34d399', padding: '1px 5px', borderRadius: 3,
                  fontWeight: 700, letterSpacing: '0.05em',
                }}>
                  +EV
                </span>
              )}
            </span>

            {/* Model % */}
            <span style={{
              fontSize: 12, fontWeight: 700, color: '#fff', textAlign: 'right',
            }}>
              {conf}%
            </span>

            {/* Edge */}
            <span style={{
              fontSize: 11, textAlign: 'right', fontWeight: 600,
              color: m.edge === null ? '#2d3040'
                : (m.edge ?? 0) >= 0 ? '#34d399' : '#f87171',
            }}>
              {m.edge === null ? '—'
                : `${(m.edge ?? 0) >= 0 ? '+' : ''}${(m.edge! * 100).toFixed(1)}%`}
            </span>

            {/* EV */}
            <span style={{
              fontSize: 11, textAlign: 'right', fontWeight: 600,
              color: m.ev === null ? '#2d3040'
                : (m.ev ?? 0) >= 0.01 ? '#34d399' : '#6b7280',
            }}>
              {m.ev === null ? '—'
                : `${(m.ev ?? 0) >= 0 ? '+' : ''}${(m.ev! * 100).toFixed(1)}%`}
            </span>

            {/* Odds */}
            <span style={{
              fontSize: 11, textAlign: 'right', color: '#9ca3af',
            }}>
              {m.market_odds === null ? '—'
                : m.market_odds > 0 ? `+${m.market_odds}` : m.market_odds}
            </span>

            {/* Stake */}
            <span style={{
              fontSize: 12, textAlign: 'right', fontWeight: 700,
              color: m.stake_units !== null ? '#4faeff' : '#2d3040',
            }}>
              {m.stake_units !== null ? `$${Math.round(m.stake_units)}` : '—'}
            </span>
          </div>
        )
      })}
    </div>
  )
}

// ── main modal ─────────────────────────────────────────────────────────────

interface Props {
  game: DailyGame
  defaultTotalLine: number
  onClose: () => void
}

export default function GameDetailModal({ game, defaultTotalLine: _dtl, onClose }: Props) {
  const [simData, setSimData] = useState<GameSimData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    fetchGameSimulation(game.game_pk, { nSims: 20_000 })
      .then(env => setSimData(env.data))
      .catch(() => setSimData(null))
      .finally(() => setLoading(false))
  }, [game.game_pk])

  const sim = game.simulation

  return (
    /* Overlay */
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 50,
        background: 'rgba(0,0,0,0.72)', display: 'flex',
        alignItems: 'flex-start', justifyContent: 'center',
        padding: '40px 20px', overflowY: 'auto',
      }}
    >
      {/* Panel */}
      <div
        onClick={e => e.stopPropagation()}
        style={{
          width: '100%', maxWidth: 720,
          background: '#16181f', borderRadius: 16,
          border: '1px solid #2a2d3a',
          overflow: 'hidden',
        }}
      >
        {/* ── Header ─────────────────────────────────────────────── */}
        <div style={{
          padding: '18px 22px', borderBottom: '1px solid #1f2230',
          display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
          background: '#12141b',
        }}>
          <div>
            <div style={{ fontSize: 17, fontWeight: 800, color: '#fff', marginBottom: 3 }}>
              {game.away_team}
              <span style={{ color: '#3d4455', margin: '0 8px', fontWeight: 400 }}>@</span>
              {game.home_team}
            </div>
            <div style={{ fontSize: 11.5, color: '#4b5563' }}>
              {game.game_date} · {game.status}
              {game.away_probable_pitcher !== 'TBD' && (
                <span style={{ marginLeft: 10 }}>
                  {game.away_probable_pitcher} vs {game.home_probable_pitcher}
                </span>
              )}
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: '#1f2230', border: 'none', color: '#9ca3af',
              width: 28, height: 28, borderRadius: 8, fontSize: 14,
              cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}
          >
            ✕
          </button>
        </div>

        <div style={{ padding: '20px 22px' }}>

          {/* ── Win probability ──────────────────────────────────── */}
          <div style={{ marginBottom: 20 }}>
            <div style={{ ...S.label, marginBottom: 10 }}>Win probability (Monte Carlo)</div>
            <ConfMeter prob={sim.away_win_prob} label={`${game.away_team} (Away)`} />
            <ConfMeter prob={sim.home_win_prob} label={`${game.home_team} (Home)`} />
          </div>

          {/* ── Run distributions ─────────────────────────────────── */}
          <div style={{ marginBottom: 20 }}>
            <div style={{ ...S.label, marginBottom: 8 }}>Projected run distribution</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              {[
                { label: game.away_team, dist: sim.away_runs, color: '#4faeff', hist: simData?.away_runs },
                { label: game.home_team, dist: sim.home_runs, color: '#34d399', hist: simData?.home_runs },
              ].map(({ label, dist, color, hist }) => (
                <div key={label} style={{
                  background: '#12141b', border: '1px solid #1f2230',
                  borderRadius: 10, padding: '12px 14px',
                }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: '#9ca3af', marginBottom: 8 }}>
                    {label.split(' ').slice(-1)[0]}
                  </div>
                  {hist && !loading ? (
                    <HistBar
                      edges={hist.histogram.edges}
                      counts={hist.histogram.counts}
                      color={color}
                      mean={dist.mean}
                    />
                  ) : (
                    <div style={{
                      height: 48, background: '#1a1d27', borderRadius: 4,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                    }}>
                      <span style={{ fontSize: 10, color: '#3d4455' }}>
                        {loading ? 'Loading distribution…' : 'Summary only'}
                      </span>
                    </div>
                  )}
                  <div style={{
                    display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)',
                    gap: 4, marginTop: 8,
                  }}>
                    {[
                      { l: 'Mean', v: dist.mean.toFixed(1) },
                      { l: 'P25', v: dist.p25.toFixed(0) },
                      { l: 'P50', v: dist.p50.toFixed(0) },
                      { l: 'P90', v: dist.p90.toFixed(0) },
                    ].map(s => (
                      <div key={s.l} style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: 12.5, fontWeight: 700, color }}>
                          {s.v}
                        </div>
                        <div style={{ ...S.label }}>{s.l}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* ── Game totals row ───────────────────────────────────── */}
          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10,
            marginBottom: 20,
          }}>
            {[
              { label: 'Total Runs Mean', val: sim.total_runs.mean.toFixed(1) },
              { label: 'Total Runs P50', val: sim.total_runs.p50.toFixed(0) },
              { label: 'Over 8.5 prob', val: `${Math.round((sim.totals['over_8_5'] ?? 0) * 100)}%` },
              { label: '-1.5 Cover prob', val: `${Math.round((sim.spread['home_cover_prob_m1.5'] ?? 0.5) * 100)}%` },
            ].map(s => (
              <div key={s.label} style={{
                background: '#12141b', border: '1px solid #1f2230',
                borderRadius: 10, padding: '11px 14px', textAlign: 'center',
              }}>
                <div style={{ fontSize: 16, fontWeight: 800, color: '#e8eaf0' }}>{s.val}</div>
                <div style={{ ...S.label, marginTop: 3 }}>{s.label}</div>
              </div>
            ))}
          </div>

          {/* ── Ranked betting opportunities ──────────────────────── */}
          <div>
            <div style={{ ...S.label, marginBottom: 8 }}>
              Betting opportunities — ranked by model confidence
              <span style={{ color: '#1e3a2a', marginLeft: 8, fontWeight: 400 }}>
                ● green background = positive expected value
              </span>
            </div>
            <MarketTable markets={game.markets} />
          </div>

          {/* ── Footer ───────────────────────────────────────────── */}
          <div style={{
            marginTop: 16, paddingTop: 12, borderTop: '1px solid #1f2230',
            fontSize: 10, color: '#2d3040', display: 'flex', justifyContent: 'space-between',
          }}>
            <span>
              {sim.n_sims.toLocaleString()} Monte Carlo simulations · Bayesian shrinkage priors
            </span>
            <span>Click outside to close</span>
          </div>
        </div>
      </div>
    </div>
  )
}
