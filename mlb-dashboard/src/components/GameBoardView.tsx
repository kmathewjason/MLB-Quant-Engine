/**
 * GameBoardView — full market board for a single game.
 * Called from SlateTab when a game is clicked.
 * Fetches GET /api/games/{game_id}/board and renders a sortable,
 * filterable table with confidence badges.
 */
import { useState, useEffect, useMemo } from 'react'
import { fetchGameBoard } from '../api'
import type { GameStub, BoardData, BoardRow, BoardCategory } from '../types'

// ── design tokens ─────────────────────────────────────────────────────────
const C = {
  bg:      '#0f1117',
  surface: '#16181f',
  card:    '#1c1f29',
  card2:   '#12141b',
  border:  '#22252e',
  border2: '#2a2d38',
  text:    '#e8eaf0',
  muted:   '#8892a4',
  muted2:  '#4b5563',
  accent:  '#4faeff',
  pos:     '#34d399',
  neg:     '#f87171',
  amber:   '#fbbf24',
}
const lbl = {
  fontSize: 9, fontWeight: 700 as const,
  letterSpacing: '0.06em', color: C.muted2, textTransform: 'uppercase' as const,
}

// ── category config ───────────────────────────────────────────────────────
type FilterCat = 'all' | BoardCategory
const CATS: { key: FilterCat; label: string; color: string }[] = [
  { key: 'all',          label: 'All',           color: C.accent  },
  { key: 'moneyline',    label: 'Moneyline',      color: '#4faeff' },
  { key: 'spread',       label: 'Run Line',       color: '#a78bfa' },
  { key: 'total',        label: 'Totals',         color: C.amber   },
  { key: 'batter_prop',  label: 'Batter Props',   color: C.pos     },
  { key: 'pitcher_prop', label: 'Pitcher Props',  color: '#f97316' },
]
const CAT_COLOR: Record<string, string> = {
  moneyline: '#4faeff', spread: '#a78bfa', total: C.amber,
  batter_prop: C.pos, pitcher_prop: '#f97316',
}

// ── sort columns ──────────────────────────────────────────────────────────
type SortKey = 'confidence_score' | 'model_prob' | 'edge' | 'ev_per_dollar' | 'kelly_stake' | 'ci_width'

// ── confidence badge bar ──────────────────────────────────────────────────
function ConfBadge({ row }: { row: BoardRow }) {
  const { point_estimate: pe, ci_low, ci_high, ci_width } = row.confidence
  const pct = Math.round(pe * 100)
  const conf = Math.abs(pct - 50)
  // Narrow CI = saturated colour; wide CI = pale
  const maxWidth = 0.3
  const saturation = Math.max(0, 1 - ci_width / maxWidth)
  const baseColor = conf >= 15 ? [52, 211, 153] : conf >= 8 ? [251, 191, 36] : [107, 114, 128]
  const r = Math.round(baseColor[0] * saturation + 50 * (1 - saturation))
  const g = Math.round(baseColor[1] * saturation + 50 * (1 - saturation))
  const b = Math.round(baseColor[2] * saturation + 50 * (1 - saturation))
  const color = `rgb(${r},${g},${b})`
  const barW  = Math.round(Math.min(100, conf * 2))

  return (
    <div style={{ minWidth: 80 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
        <span style={{ fontSize: 10, fontWeight: 700, color }}>{pct}%</span>
        <span style={{ fontSize: 9, color: C.muted2 }}>
          [{Math.round(ci_low * 100)}–{Math.round(ci_high * 100)}%]
        </span>
      </div>
      <div style={{ height: 4, background: C.border2, borderRadius: 3 }}>
        <div style={{
          width: `${barW}%`, height: '100%',
          background: color, borderRadius: 3, opacity: 0.85 + saturation * 0.15,
        }} />
      </div>
    </div>
  )
}

// ── skeleton table ─────────────────────────────────────────────────────────
function SkeletonTable() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {[1,2,3,4,5,6,7,8].map(i => (
        <div key={i} style={{
          height: 44, background: C.card, border: `1px solid ${C.border}`,
          borderRadius: 8, opacity: 0.3 + i * 0.06,
        }} />
      ))}
    </div>
  )
}

// ── section error ─────────────────────────────────────────────────────────
function SectionError({ msg, onRetry }: { msg: string; onRetry: () => void }) {
  return (
    <div style={{
      background: '#1e0e0e', border: `1px solid #6b2020`,
      borderRadius: 10, padding: '18px 20px',
    }}>
      <div style={{ color: C.neg, fontWeight: 700, fontSize: 13, marginBottom: 6 }}>
        Failed to load market board
      </div>
      <div style={{ color: '#e8938a', fontSize: 12, marginBottom: 12 }}>{msg}</div>
      <button onClick={onRetry} style={{
        background: '#3a1010', border: `1px solid #7f2020`, borderRadius: 6,
        color: C.neg, padding: '6px 14px', fontSize: 12, fontWeight: 600, cursor: 'pointer',
      }}>
        Retry
      </button>
    </div>
  )
}

// ── sort arrow ────────────────────────────────────────────────────────────
function SortArrow({ active, asc }: { active: boolean; asc: boolean }) {
  if (!active) return <span style={{ color: C.muted2, fontSize: 8 }}> ⇅</span>
  return <span style={{ color: C.accent, fontSize: 8 }}> {asc ? '↑' : '↓'}</span>
}

// ── main component ────────────────────────────────────────────────────────
interface Props {
  game: GameStub
  onBack: () => void
}

export default function GameBoardView({ game, onBack }: Props) {
  const [board,   setBoard]   = useState<BoardData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState<string | null>(null)

  // Board controls
  const [catFilter, setCatFilter]   = useState<FilterCat>('all')
  const [minEdge,   setMinEdge]     = useState(0)        // slider 0–20 (%)
  const [sortKey,   setSortKey]     = useState<SortKey>('confidence_score')
  const [sortAsc,   setSortAsc]     = useState(false)
  const [nSims,     setNSims]       = useState(20_000)
  const [bankroll,  setBankroll]    = useState(1_000)

  const load = () => {
    setLoading(true); setError(null)
    fetchGameBoard(game.game_id, { nSims, bankroll })
      .then(env => setBoard(env.data))
      .catch(e => {
        console.error('[GameBoardView] fetchGameBoard failed:', e)
        setError(String(e))
      })
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [game.game_id])  // eslint-disable-line react-hooks/exhaustive-deps

  // ── Filtered + sorted rows ───────────────────────────────────────────
  const rows = useMemo(() => {
    if (!board) return []
    let r = board.rows
    if (catFilter !== 'all') r = r.filter(row => row.category === catFilter)
    if (minEdge > 0) r = r.filter(row => row.edge !== null && row.edge * 100 >= minEdge)
    return [...r].sort((a, b) => {
      const resolve = (row: typeof a): number => {
        if (sortKey === 'ci_width') return row.confidence.ci_width ?? -Infinity
        const v = (row as unknown as Record<string, number | null>)[sortKey]
        return v ?? -Infinity
      }
      const av = resolve(a)
      const bv = resolve(b)
      return sortAsc ? av - bv : bv - av
    })
  }, [board, catFilter, minEdge, sortKey, sortAsc])

  // Counts per category for filter chips
  const catCounts = useMemo(() => {
    if (!board) return {}
    const c: Record<string, number> = {}
    for (const r of board.rows) c[r.category] = (c[r.category] ?? 0) + 1
    return c
  }, [board])

  // Column header click
  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc(a => !a)
    else { setSortKey(key); setSortAsc(false) }
  }

  const positiveEV = board?.rows.filter(r => (r.ev_per_dollar ?? 0) > 0).length ?? 0

  return (
    <div style={{ maxWidth: 960 }}>

      {/* ── Back + title ─────────────────────────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 14, marginBottom: 20 }}>
        <button
          onClick={onBack}
          style={{
            background: C.card, border: `1px solid ${C.border2}`, borderRadius: 8,
            color: C.muted, padding: '7px 14px', fontSize: 12, cursor: 'pointer',
            fontWeight: 600, flexShrink: 0,
          }}
        >
          ← Back
        </button>
        <div>
          <h1 style={{ fontSize: 20, fontWeight: 800, color: '#fff', margin: '0 0 3px' }}>
            {game.away_team}
            <span style={{ color: C.muted2, margin: '0 9px', fontWeight: 400 }}>@</span>
            {game.home_team}
          </h1>
          <div style={{ fontSize: 11, color: C.muted2 }}>
            {game.game_date} · {game.start_time_local} · {game.park_name}
            {game.probable_pitcher_away !== 'TBD' && (
              <span style={{ marginLeft: 10 }}>
                SP: {game.probable_pitcher_away.split(' ').pop()} vs {game.probable_pitcher_home.split(' ').pop()}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* ── Stats strip (once board loaded) ──────────────────────────── */}
      {board && !loading && (
        <div style={{
          display: 'flex', gap: 10, marginBottom: 18, flexWrap: 'wrap',
        }}>
          {[
            { label: 'Markets',     val: board.rows.length.toString() },
            { label: '+EV Markets', val: positiveEV.toString(), accent: positiveEV > 0 },
            { label: 'Simulations', val: board.n_sims.toLocaleString() },
            { label: 'Omitted',     val: board.omitted.length.toString(), warn: board.omitted.length > 0 },
          ].map(s => (
            <div key={s.label} style={{
              background: C.card, border: `1px solid ${C.border}`,
              borderRadius: 9, padding: '10px 16px', textAlign: 'center', minWidth: 110,
            }}>
              <div style={{
                fontSize: 18, fontWeight: 800,
                color: s.accent ? C.pos : s.warn && s.val !== '0' ? C.amber : C.text,
              }}>
                {s.val}
              </div>
              <div style={{ ...lbl, marginTop: 2 }}>{s.label}</div>
            </div>
          ))}
        </div>
      )}

      {/* ── Controls ─────────────────────────────────────────────────── */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 14, flexWrap: 'wrap', alignItems: 'flex-end' }}>

        {/* Sim quality */}
        <div>
          <div style={{ ...lbl, marginBottom: 5 }}>Sim quality</div>
          <select
            value={nSims}
            onChange={e => setNSims(Number(e.target.value))}
            style={{
              background: C.card, border: `1px solid ${C.border2}`, borderRadius: 7,
              padding: '7px 10px', color: C.text, fontSize: 12, cursor: 'pointer',
            }}
          >
            <option value={5_000}>5 k — fast</option>
            <option value={20_000}>20 k — standard</option>
            <option value={50_000}>50 k — precise</option>
          </select>
        </div>

        {/* Bankroll */}
        <div>
          <div style={{ ...lbl, marginBottom: 5 }}>Bankroll $</div>
          <input
            type="number" value={bankroll}
            onChange={e => setBankroll(Number(e.target.value))}
            style={{
              width: 90, background: C.card, border: `1px solid ${C.border2}`,
              borderRadius: 7, padding: '7px 10px', color: C.text, fontSize: 12, outline: 'none',
            }}
          />
        </div>

        {/* Min edge slider */}
        <div>
          <div style={{ ...lbl, marginBottom: 5 }}>Min edge: {minEdge}%</div>
          <input
            type="range" min={0} max={20} step={1} value={minEdge}
            onChange={e => setMinEdge(Number(e.target.value))}
            style={{ width: 120, cursor: 'pointer', accentColor: C.accent }}
          />
        </div>

        {/* Run board button */}
        <button
          onClick={load}
          disabled={loading}
          style={{
            background: loading ? '#1e3358' : C.accent, color: loading ? '#4faeff66' : '#fff',
            border: 'none', borderRadius: 7, padding: '8px 16px',
            fontWeight: 700, fontSize: 12.5, cursor: loading ? 'not-allowed' : 'pointer',
            alignSelf: 'flex-end',
          }}
        >
          {loading ? '⏳ Simulating…' : '▶ Run Board'}
        </button>
      </div>

      {/* ── Category filter chips ─────────────────────────────────────── */}
      {board && (
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 14 }}>
          {CATS.filter(c => c.key === 'all' || (catCounts[c.key] ?? 0) > 0).map(c => {
            const cnt = c.key === 'all' ? board.rows.length : (catCounts[c.key] ?? 0)
            const active = catFilter === c.key
            return (
              <button
                key={c.key}
                onClick={() => setCatFilter(c.key)}
                style={{
                  padding: '5px 13px', borderRadius: 99, fontSize: 11,
                  fontWeight: 600, border: 'none', cursor: 'pointer',
                  background: active ? c.color : C.border2,
                  color: active ? (c.key === 'total' ? '#111' : '#fff') : C.muted,
                }}
              >
                {c.label}
                <span style={{
                  marginLeft: 5, fontSize: 10,
                  color: active ? 'rgba(0,0,0,0.4)' : C.muted2,
                }}>
                  {cnt}
                </span>
              </button>
            )
          })}
        </div>
      )}

      {/* ── Error ────────────────────────────────────────────────────── */}
      {error && !loading && (
        <div style={{ marginBottom: 16 }}>
          <SectionError msg={error} onRetry={load} />
        </div>
      )}

      {/* ── Loading skeleton ─────────────────────────────────────────── */}
      {loading && (
        <div>
          <div style={{ color: C.muted, fontSize: 12, marginBottom: 12 }}>
            Running {nSims.toLocaleString()} simulations + bootstrap CI…
          </div>
          <SkeletonTable />
        </div>
      )}

      {/* ── Market table ─────────────────────────────────────────────── */}
      {board && !loading && (
        <>
          {rows.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '40px 0', color: C.muted2, fontSize: 13 }}>
              No markets match the current filters.
            </div>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{
                width: '100%', borderCollapse: 'collapse', fontSize: 12,
              }}>
                <thead>
                  <tr style={{ background: C.card2, borderBottom: `1px solid ${C.border}` }}>
                    {/* Category dot */}
                    <th style={{ width: 8, padding: '8px 6px' }} />
                    {[
                      { key: null as SortKey | null, label: 'Market',    align: 'left'   },
                      { key: 'model_prob'   as SortKey, label: 'Model %', align: 'right' },
                      { key: null,           label: 'Confidence',         align: 'left'  },
                      { key: 'edge'         as SortKey, label: 'Edge',    align: 'right' },
                      { key: 'ev_per_dollar' as SortKey, label: 'EV',     align: 'right' },
                      { key: null,           label: 'Odds',               align: 'right' },
                      { key: 'kelly_stake'  as SortKey, label: 'Kelly $', align: 'right' },
                      { key: 'confidence_score' as SortKey, label: 'Score', align: 'right' },
                    ].map(col => (
                      <th
                        key={col.label}
                        onClick={col.key ? () => toggleSort(col.key!) : undefined}
                        style={{
                          padding: '8px 10px', textAlign: col.align as any,
                          color: sortKey === col.key ? C.accent : C.muted,
                          fontWeight: 600, fontSize: 10, letterSpacing: '0.05em',
                          textTransform: 'uppercase', whiteSpace: 'nowrap',
                          cursor: col.key ? 'pointer' : 'default',
                          userSelect: 'none',
                        }}
                      >
                        {col.label}
                        {col.key && <SortArrow active={sortKey === col.key} asc={sortAsc} />}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, i) => {
                    const evPos = (row.ev_per_dollar ?? 0) > 0.005
                    const evNeg = row.ev_per_dollar !== null && (row.ev_per_dollar ?? 0) < -0.02
                    const dot   = CAT_COLOR[row.category] ?? C.muted
                    return (
                      <tr
                        key={`${row.label}-${row.side}-${i}`}
                        style={{
                          borderBottom: `1px solid ${C.border}`,
                          background: evPos ? '#091a0f' : evNeg ? '#1a0909' : i % 2 === 0 ? 'transparent' : '#ffffff04',
                        }}
                      >
                        {/* Colour dot */}
                        <td style={{ padding: '0 6px', textAlign: 'center' }}>
                          <span style={{
                            display: 'inline-block', width: 7, height: 7,
                            borderRadius: '50%', background: dot,
                          }} />
                        </td>

                        {/* Label */}
                        <td style={{ padding: '9px 10px', color: C.text, fontWeight: 500, whiteSpace: 'nowrap', maxWidth: 220 }}>
                          {row.label}
                          {evPos && (
                            <span style={{
                              marginLeft: 6, fontSize: 8, background: '#0d3326',
                              color: C.pos, padding: '1px 5px', borderRadius: 3,
                              fontWeight: 700, letterSpacing: '0.05em',
                            }}>+EV</span>
                          )}
                        </td>

                        {/* Model % */}
                        <td style={{ padding: '9px 10px', textAlign: 'right', fontWeight: 700, color: '#fff', whiteSpace: 'nowrap' }}>
                          {Math.round(row.model_prob * 100)}%
                        </td>

                        {/* Confidence badge */}
                        <td style={{ padding: '9px 10px' }}>
                          <ConfBadge row={row} />
                        </td>

                        {/* Edge */}
                        <td style={{
                          padding: '9px 10px', textAlign: 'right', fontWeight: 600,
                          color: row.edge === null ? C.muted2 : row.edge >= 0 ? C.pos : C.neg,
                          whiteSpace: 'nowrap',
                        }}>
                          {row.edge === null ? '—'
                            : `${row.edge >= 0 ? '+' : ''}${(row.edge * 100).toFixed(1)}%`}
                        </td>

                        {/* EV */}
                        <td style={{
                          padding: '9px 10px', textAlign: 'right', fontWeight: 600,
                          color: row.ev_per_dollar === null ? C.muted2
                            : evPos ? C.pos : C.muted,
                          whiteSpace: 'nowrap',
                        }}>
                          {row.ev_per_dollar === null ? '—'
                            : `${row.ev_per_dollar >= 0 ? '+' : ''}${(row.ev_per_dollar * 100).toFixed(1)}%`}
                        </td>

                        {/* Odds */}
                        <td style={{ padding: '9px 10px', textAlign: 'right', color: C.muted, whiteSpace: 'nowrap' }}>
                          {row.market_odds === null ? '—'
                            : row.market_odds > 0 ? `+${row.market_odds}` : row.market_odds}
                        </td>

                        {/* Kelly $ */}
                        <td style={{
                          padding: '9px 10px', textAlign: 'right', fontWeight: 700,
                          color: row.kelly_stake !== null ? C.accent : C.muted2,
                          whiteSpace: 'nowrap',
                        }}>
                          {row.kelly_stake !== null ? `$${row.kelly_stake.toFixed(0)}` : '—'}
                        </td>

                        {/* Confidence score */}
                        <td style={{
                          padding: '9px 10px', textAlign: 'right',
                          color: row.confidence_score > 0.5 ? C.pos : C.muted2,
                          fontVariantNumeric: 'tabular-nums', fontSize: 11,
                        }}>
                          {row.confidence_score.toFixed(2)}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* Omitted props warning */}
          {board.omitted.length > 0 && (
            <details style={{ marginTop: 14, cursor: 'pointer' }}>
              <summary style={{ fontSize: 11, color: C.amber, fontWeight: 600 }}>
                ⚠ {board.omitted.length} market{board.omitted.length !== 1 ? 's' : ''} omitted due to errors
              </summary>
              <div style={{
                marginTop: 8, background: C.card2, border: `1px solid ${C.border}`,
                borderRadius: 8, padding: '10px 14px',
              }}>
                {board.omitted.map((o, i) => (
                  <div key={i} style={{ fontSize: 11, color: C.muted, marginBottom: 4 }}>
                    <strong style={{ color: C.amber }}>{o.label}</strong>: {o.reason}
                  </div>
                ))}
              </div>
            </details>
          )}

          {/* Footer */}
          <div style={{ marginTop: 12, fontSize: 10, color: C.muted2, display: 'flex', justifyContent: 'space-between' }}>
            <span>
              Confidence interval: bootstrap 5th–95th percentile ({board.n_sims.toLocaleString()} sims, 500 resamples)
            </span>
            <span>Sort: {sortKey.replace(/_/g, ' ')} {sortAsc ? '↑' : '↓'}</span>
          </div>
        </>
      )}
    </div>
  )
}
