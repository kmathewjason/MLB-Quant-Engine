/**
 * GameDetailModal — full distribution view for a single game.
 * Markets grouped by category with tab navigation.
 */
import { useState, useEffect, useMemo } from 'react'
import { fetchGameSimulation } from '../api'
import type { DailyGame, GameSimData, MarketRow } from '../types'

const S = {
  label: {
    fontSize: 9, fontWeight: 700 as const, letterSpacing: '0.07em',
    color: '#5a6072', textTransform: 'uppercase' as const,
  },
}

// ── design tokens ─────────────────────────────────────────────────────────
const C = {
  bg:     '#0f1117', surface: '#16181f', card:   '#1c1f29',
  card2:  '#12141b', border:  '#1f2230', border2: '#2a2d38',
  text:   '#e8eaf0', muted:   '#6b7280', muted2: '#3d4455',
  accent: '#4faeff', pos:     '#34d399', neg:    '#f87171', amber: '#fbbf24',
}

// ── category config ────────────────────────────────────────────────────────
const CATEGORIES = [
  { key: 'all',         label: 'All Markets' },
  { key: 'moneyline',   label: 'Moneyline'   },
  { key: 'spread',      label: 'Run Line'    },
  { key: 'total',       label: 'Totals'      },
  { key: 'player_prop', label: 'Player Props'},
]

// ── histogram bar chart ────────────────────────────────────────────────────
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
          <div key={i}
            title={`${edge}–${edges[i + 1] ?? '+'}: ${c.toLocaleString()}`}
            style={{
              flex: 1, height: `${Math.max(2, (c / max) * 100)}%`,
              background: isMean ? '#fff' : color, borderRadius: '2px 2px 0 0',
              opacity: isMean ? 1 : 0.65, minWidth: 3,
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
  const conf = Math.abs(pct - 50)
  const bar  = Math.min(100, conf * 2)
  const color = conf >= 15 ? C.pos : conf >= 8 ? C.amber : C.muted
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
          borderRadius: 6,
        }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 3 }}>
        <span style={{ fontSize: 9, color: C.muted2 }}>0%</span>
        <span style={{ fontSize: 9, color }}>
          Confidence: {conf >= 15 ? 'HIGH' : conf >= 8 ? 'MEDIUM' : 'LOW'}
          {' '}({bar.toFixed(0)}%)
        </span>
        <span style={{ fontSize: 9, color: C.muted2 }}>100%</span>
      </div>
    </div>
  )
}

// ── market table ───────────────────────────────────────────────────────────
function MarketTable({ markets }: { markets: MarketRow[] }) {
  const sorted = useMemo(() => [...markets].sort((a, b) => {
    if (a.ev !== null && b.ev !== null) return b.ev - a.ev
    if (a.ev !== null) return -1
    if (b.ev !== null) return 1
    return Math.abs(b.model_prob - 0.5) - Math.abs(a.model_prob - 0.5)
  }), [markets])

  if (sorted.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: '24px 0', color: C.muted, fontSize: 12 }}>
        No markets available in this category
      </div>
    )
  }

  return (
    <div style={{ background: C.card2, borderRadius: 10, border: `1px solid ${C.border}`, overflow: 'hidden' }}>
      {/* Header */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '22px 1fr 52px 52px 52px 52px 58px',
        gap: 6, padding: '7px 12px',
        background: C.bg, borderBottom: `1px solid ${C.border}`,
      }}>
        {['#', 'Market', 'Model', 'Edge', 'EV', 'Odds', 'Kelly $'].map(h => (
          <span key={h} style={S.label}>{h}</span>
        ))}
      </div>

      {sorted.map((m, i) => {
        const rankColor = i === 0 ? C.amber : i === 1 ? '#9ca3af' : i === 2 ? '#cd7c2e' : '#3a4055'
        const evPos = (m.ev ?? 0) > 0.005
        const conf  = Math.round(m.model_prob * 100)
        const catColor: Record<string, string> = {
          moneyline: '#4faeff', spread: '#a78bfa', total: '#fbbf24',
          player_prop: '#34d399', game: '#9ca3af',
        }
        const dotColor = catColor[m.category] ?? '#9ca3af'
        return (
          <div key={`${m.market}-${i}`} style={{
            display: 'grid',
            gridTemplateColumns: '22px 1fr 52px 52px 52px 52px 58px',
            gap: 6, padding: '8px 12px', alignItems: 'center',
            borderBottom: `1px solid #1a1d24`,
            background: evPos ? '#0a1a12' : 'transparent',
          }}>
            <span style={{
              width: 18, height: 18, borderRadius: '50%',
              background: rankColor, color: i < 3 ? '#000' : C.muted,
              fontSize: 9, fontWeight: 700,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>{i + 1}</span>

            <span style={{ fontSize: 12, color: '#d1d5db', fontWeight: 500, overflow: 'hidden' }}>
              <span style={{
                display: 'inline-block', width: 6, height: 6, borderRadius: '50%',
                background: dotColor, marginRight: 5, flexShrink: 0,
                verticalAlign: 'middle',
              }} />
              {m.label}
              {evPos && (
                <span style={{
                  marginLeft: 6, fontSize: 8, background: '#0d3326',
                  color: C.pos, padding: '1px 5px', borderRadius: 3,
                  fontWeight: 700, letterSpacing: '0.05em',
                }}>+EV</span>
              )}
            </span>

            <span style={{ fontSize: 12, fontWeight: 700, color: '#fff', textAlign: 'right' }}>
              {conf}%
            </span>

            <span style={{
              fontSize: 11, textAlign: 'right', fontWeight: 600,
              color: m.edge === null ? C.muted2 : (m.edge ?? 0) >= 0 ? C.pos : C.neg,
            }}>
              {m.edge === null ? '—'
                : `${(m.edge ?? 0) >= 0 ? '+' : ''}${(m.edge! * 100).toFixed(1)}%`}
            </span>

            <span style={{
              fontSize: 11, textAlign: 'right', fontWeight: 600,
              color: m.ev === null ? C.muted2 : (m.ev ?? 0) >= 0.01 ? C.pos : C.muted,
            }}>
              {m.ev === null ? '—'
                : `${(m.ev ?? 0) >= 0 ? '+' : ''}${(m.ev! * 100).toFixed(1)}%`}
            </span>

            <span style={{ fontSize: 11, textAlign: 'right', color: '#9ca3af' }}>
              {m.market_odds === null ? '—'
                : m.market_odds > 0 ? `+${m.market_odds}` : m.market_odds}
            </span>

            <span style={{
              fontSize: 12, textAlign: 'right', fontWeight: 700,
              color: m.stake_units !== null ? C.accent : C.muted2,
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
  const [simData,  setSimData]  = useState<GameSimData | null>(null)
  const [loading,  setLoading]  = useState(true)
  const [activeTab, setTab]     = useState('all')

  useEffect(() => {
    setLoading(true)
    fetchGameSimulation(game.game_pk, { nSims: 20_000 })
      .then(env => setSimData(env.data))
      .catch(() => setSimData(null))
      .finally(() => setLoading(false))
  }, [game.game_pk])

  const sim = game.simulation

  // ── Market counts per category for tab labels ──────────────────────────
  const counts = useMemo(() => {
    const c: Record<string, number> = {}
    for (const m of game.markets) {
      c[m.category] = (c[m.category] ?? 0) + 1
    }
    return c
  }, [game.markets])

  const filteredMarkets = useMemo(() => {
    if (activeTab === 'all') return game.markets
    return game.markets.filter(m => m.category === activeTab)
  }, [game.markets, activeTab])

  // Visible tabs: only show categories that have markets
  const visibleTabs = CATEGORIES.filter(
    cat => cat.key === 'all' || (counts[cat.key] ?? 0) > 0
  )

  const spreadCoverKey = Object.keys(sim.spread).find(k => k.includes('m1')) ?? ''

  return (
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, zIndex: 50,
      background: 'rgba(0,0,0,0.72)', display: 'flex',
      alignItems: 'flex-start', justifyContent: 'center',
      padding: '40px 20px', overflowY: 'auto',
    }}>
      <div onClick={e => e.stopPropagation()} style={{
        width: '100%', maxWidth: 760,
        background: C.surface, borderRadius: 16,
        border: `1px solid ${C.border2}`, overflow: 'hidden',
      }}>

        {/* ── Header ──────────────────────────────────────────────── */}
        <div style={{
          padding: '18px 22px', borderBottom: `1px solid ${C.border}`,
          display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
          background: C.card2,
        }}>
          <div>
            <div style={{ fontSize: 17, fontWeight: 800, color: '#fff', marginBottom: 3 }}>
              {game.away_team}
              <span style={{ color: C.muted2, margin: '0 8px', fontWeight: 400 }}>@</span>
              {game.home_team}
            </div>
            <div style={{ fontSize: 11.5, color: '#4b5563' }}>
              {game.game_date} · {game.status}
              {game.away_probable_pitcher !== 'TBD' && (
                <span style={{ marginLeft: 10 }}>
                  SP: {game.away_probable_pitcher} vs {game.home_probable_pitcher}
                </span>
              )}
            </div>
          </div>
          <button onClick={onClose} style={{
            background: '#1f2230', border: 'none', color: '#9ca3af',
            width: 28, height: 28, borderRadius: 8, fontSize: 14,
            cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>✕</button>
        </div>

        <div style={{ padding: '20px 22px' }}>

          {/* ── Win probability ─────────────────────────────────────── */}
          <div style={{ marginBottom: 20 }}>
            <div style={{ ...S.label, marginBottom: 10 }}>Win probability (Monte Carlo)</div>
            <ConfMeter prob={sim.away_win_prob} label={`${game.away_team} (Away)`} />
            <ConfMeter prob={sim.home_win_prob} label={`${game.home_team} (Home)`} />
          </div>

          {/* ── Run distributions ───────────────────────────────────── */}
          <div style={{ marginBottom: 20 }}>
            <div style={{ ...S.label, marginBottom: 8 }}>Projected run distribution</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              {[
                { label: game.away_team, dist: sim.away_runs, color: C.accent, hist: simData?.away_runs },
                { label: game.home_team, dist: sim.home_runs, color: C.pos,    hist: simData?.home_runs },
              ].map(({ label, dist, color, hist }) => (
                <div key={label} style={{
                  background: C.card2, border: `1px solid ${C.border}`,
                  borderRadius: 10, padding: '12px 14px',
                }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: '#9ca3af', marginBottom: 8 }}>
                    {label.split(' ').slice(-1)[0]}
                  </div>
                  {hist && !loading ? (
                    <HistBar edges={hist.histogram.edges} counts={hist.histogram.counts}
                      color={color} mean={dist.mean} />
                  ) : (
                    <div style={{
                      height: 48, background: '#1a1d27', borderRadius: 4,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                    }}>
                      <span style={{ fontSize: 10, color: C.muted2 }}>
                        {loading ? 'Loading…' : 'No histogram data'}
                      </span>
                    </div>
                  )}
                  <div style={{
                    display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)',
                    gap: 4, marginTop: 8,
                  }}>
                    {[
                      { l: 'Mean', v: dist.mean.toFixed(1) },
                      { l: 'P25',  v: dist.p25.toFixed(0)  },
                      { l: 'P50',  v: dist.p50.toFixed(0)  },
                      { l: 'P90',  v: dist.p90.toFixed(0)  },
                    ].map(s => (
                      <div key={s.l} style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: 12.5, fontWeight: 700, color }}>{s.v}</div>
                        <div style={{ ...S.label }}>{s.l}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* ── Game totals strip ────────────────────────────────────── */}
          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)',
            gap: 8, marginBottom: 22,
          }}>
            {[
              { label: 'Total Mean',   val: sim.total_runs.mean.toFixed(1) },
              { label: 'Total Median', val: sim.total_runs.p50.toFixed(0)  },
              { label: 'Over 8.5',     val: `${Math.round((sim.totals['over_8_5'] ?? 0) * 100)}%` },
              { label: 'Home -1.5',    val: `${Math.round((sim.spread[spreadCoverKey] ?? 0.5) * 100)}%` },
            ].map(s => (
              <div key={s.label} style={{
                background: C.card2, border: `1px solid ${C.border}`,
                borderRadius: 10, padding: '11px 14px', textAlign: 'center',
              }}>
                <div style={{ fontSize: 16, fontWeight: 800, color: C.text }}>{s.val}</div>
                <div style={{ ...S.label, marginTop: 3 }}>{s.label}</div>
              </div>
            ))}
          </div>

          {/* ── Market tabs ──────────────────────────────────────────── */}
          <div style={{ ...S.label, marginBottom: 10 }}>
            Betting markets — ranked by model confidence
          </div>

          {/* Category tab bar */}
          <div style={{ display: 'flex', gap: 6, marginBottom: 12, flexWrap: 'wrap' }}>
            {visibleTabs.map(tab => {
              const cnt = tab.key === 'all' ? game.markets.length : (counts[tab.key] ?? 0)
              const active = activeTab === tab.key
              return (
                <button
                  key={tab.key}
                  onClick={() => setTab(tab.key)}
                  style={{
                    padding: '5px 13px', borderRadius: 99, fontSize: 11,
                    fontWeight: 600, border: 'none', cursor: 'pointer',
                    background: active ? C.accent : C.border2,
                    color: active ? '#fff' : C.muted,
                  }}
                >
                  {tab.label}
                  <span style={{
                    marginLeft: 5, fontSize: 10,
                    color: active ? 'rgba(255,255,255,0.7)' : C.muted2,
                  }}>{cnt}</span>
                </button>
              )
            })}
          </div>

          {/* Market table */}
          <MarketTable markets={filteredMarkets} />

          {/* ── Legend ────────────────────────────────────────────────── */}
          <div style={{
            display: 'flex', gap: 14, flexWrap: 'wrap',
            marginTop: 10, paddingTop: 8, borderTop: `1px solid ${C.border}`,
          }}>
            {[
              { color: '#4faeff',  label: 'Moneyline' },
              { color: '#a78bfa',  label: 'Run Line'  },
              { color: '#fbbf24',  label: 'Total'     },
              { color: '#34d399',  label: 'Player Prop'},
            ].map(({ color, label }) => (
              <span key={label} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 10, color: C.muted }}>
                <span style={{ width: 7, height: 7, borderRadius: '50%', background: color, flexShrink: 0 }} />
                {label}
              </span>
            ))}
            <span style={{ marginLeft: 'auto', fontSize: 10, color: C.muted2 }}>
              {sim.n_sims.toLocaleString()} sims · log5 matchup-adjusted
            </span>
          </div>

          <div style={{ marginTop: 8, fontSize: 10, color: C.muted2, textAlign: 'right' }}>
            Click outside to close
          </div>
        </div>
      </div>
    </div>
  )
}
