/**
 * SGPTab — redesigned parlay builder.
 *
 * Layout
 * ──────
 * 1. GamePicker   — reuses today's slate (same fetchGamesToday as SlateTab),
 *                   search-filterable, click to select a game.
 * 2. Two top tabs: "Build Your Own" | "Suggested Parlays"
 *
 * Build Your Own tab
 * ──────────────────
 * - Loads GET /api/games/{id}/legs as a checkable list.
 * - User picks 2+ legs → Calculate → POST /api/parlays/evaluate.
 * - Shows naive vs corr-adjusted side-by-side, pct diff callout,
 *   per-leg marginal table, group breakdown.
 *
 * Suggested Parlays tab
 * ──────────────────────
 * - Two sub-tabs: Same-Game | Cross-Game.
 * - Each calls GET /api/parlays/suggested?mode=sgp|crossgame.
 * - Cards ranked by EV; shows legs, joint prob, CI bar, Kelly stake.
 * - Empty/error states surfaced explicitly.
 */

import { useEffect, useMemo, useState } from 'react'
import {
  fetchGamesToday,
  fetchGameLegs,
  fetchEvaluateParlay,
  fetchSuggestedParlays,
} from '../api'
import type {
  GameStub,
  GameLeg,
  ParlayLegInput,
  ParlayEvalResult,
  SuggestedParlay,
  SuggestedParlaysData,
  KellyVariant,
} from '../types'

// ── design tokens ────────────────────────────────────────────────────────────
const C = {
  bg:      '#0f1117',
  surface: '#16181f',
  card:    '#1c1f29',
  border:  '#22252e',
  border2: '#2a2d38',
  text:    '#e8eaf0',
  muted:   '#8892a4',
  muted2:  '#5a6072',
  accent:  '#4faeff',
  pos:     '#34d399',
  neg:     '#f87171',
  amber:   '#fbbf24',
}

// ── tiny helpers ─────────────────────────────────────────────────────────────
function pct(v: number, d = 1) { return `${(v * 100).toFixed(d)}%` }
function fmtOdds(o: number | null) {
  if (o === null) return '—'
  return o >= 0 ? `+${o}` : `${o}`
}

// ── shared primitives ────────────────────────────────────────────────────────
function Spinner() {
  return (
    <div style={{ display: 'flex', justifyContent: 'center', padding: '32px 0' }}>
      <div style={{
        width: 28, height: 28, borderRadius: '50%',
        border: `3px solid ${C.border2}`,
        borderTopColor: C.accent,
        animation: 'spin 0.7s linear infinite',
      }} />
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}

function ErrorCard({ msg, onRetry }: { msg: string; onRetry?: () => void }) {
  return (
    <div style={{
      background: '#2a1515', border: '1px solid #6b2020', borderRadius: 8,
      color: C.neg, fontSize: 13, padding: '12px 16px',
      display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12,
    }}>
      <span>⚠ {msg}</span>
      {onRetry && (
        <button onClick={onRetry} style={{
          background: 'none', border: `1px solid ${C.neg}`, borderRadius: 6,
          color: C.neg, fontSize: 11, padding: '3px 10px', cursor: 'pointer',
        }}>Retry</button>
      )}
    </div>
  )
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      fontSize: 10, fontWeight: 700, color: C.muted, textTransform: 'uppercase',
      letterSpacing: '0.08em', marginBottom: 10,
    }}>{children}</div>
  )
}

function TabPill({
  label, active, onClick,
}: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick} style={{
      background: active ? C.accent : 'transparent',
      color: active ? '#fff' : C.muted,
      border: active ? 'none' : `1px solid ${C.border2}`,
      borderRadius: 20, padding: '5px 16px', fontSize: 12,
      fontWeight: active ? 700 : 400, cursor: 'pointer',
    }}>{label}</button>
  )
}

// ── Kelly comparison ─────────────────────────────────────────────────────────
function KellyComparison({
  corrVariant, naiveVariant, bankroll, pctDiff,
}: {
  corrVariant: KellyVariant
  naiveVariant: KellyVariant
  bankroll: number
  pctDiff: number
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 16 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        <div style={{
          background: C.card, border: `2px solid ${C.accent}`,
          borderRadius: 10, padding: '14px 16px',
        }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: C.accent, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>
            Corr-adjusted
          </div>
          <div style={{ fontSize: 26, fontWeight: 800, color: C.accent, fontVariantNumeric: 'tabular-nums' }}>
            ${corrVariant.stake_units.toFixed(0)}
          </div>
          <div style={{ fontSize: 11, color: C.muted, marginTop: 3 }}>
            f* = {pct(corrVariant.f_star)} → ¼K
          </div>
          <div style={{ fontSize: 12, fontWeight: 600, marginTop: 5,
            color: corrVariant.ev >= 0 ? C.pos : C.neg }}>
            EV {corrVariant.ev >= 0 ? '+' : ''}{(corrVariant.ev * 100).toFixed(2)}%
          </div>
        </div>
        <div style={{
          background: C.card, border: `1px solid ${C.border}`,
          borderRadius: 10, padding: '14px 16px',
        }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>
            Naive (independent)
          </div>
          <div style={{ fontSize: 26, fontWeight: 800, color: C.muted, fontVariantNumeric: 'tabular-nums' }}>
            ${naiveVariant.stake_units.toFixed(0)}
          </div>
          <div style={{ fontSize: 11, color: C.muted2, marginTop: 3 }}>
            f* = {pct(naiveVariant.f_star)} → ¼K
          </div>
          <div style={{ fontSize: 12, fontWeight: 600, marginTop: 5,
            color: naiveVariant.ev >= 0 ? C.pos : C.neg }}>
            EV {naiveVariant.ev >= 0 ? '+' : ''}{(naiveVariant.ev * 100).toFixed(2)}%
          </div>
        </div>
      </div>
      <div style={{
        background: Math.abs(pctDiff) > 5 ? '#26200e' : C.card,
        border: `1px solid ${Math.abs(pctDiff) > 5 ? C.amber : C.border}`,
        borderRadius: 8, padding: '8px 14px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <span style={{ fontSize: 12, color: C.text }}>Correlation model value</span>
        <span style={{ fontSize: 14, fontWeight: 800,
          color: pctDiff > 0 ? C.pos : pctDiff < 0 ? C.neg : C.muted }}>
          {pctDiff >= 0 ? '+' : ''}{pctDiff.toFixed(1)}% vs naive
        </span>
        <span style={{ fontSize: 10, color: C.muted2 }}>{bankroll} unit bankroll</span>
      </div>
    </div>
  )
}

// ── GamePicker ────────────────────────────────────────────────────────────────
function GamePicker({
  selected,
  onSelect,
}: {
  selected: GameStub | null
  onSelect: (g: GameStub) => void
}) {
  const [games,   setGames]   = useState<GameStub[]>([])
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState<string | null>(null)
  const [query,   setQuery]   = useState('')

  const load = () => {
    setLoading(true); setError(null)
    fetchGamesToday()
      .then(env => setGames(env.data))
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  const filtered = useMemo(() => {
    if (!query.trim()) return games
    const q = query.toLowerCase()
    return games.filter(g =>
      g.home_team.toLowerCase().includes(q) ||
      g.away_team.toLowerCase().includes(q),
    )
  }, [games, query])

  return (
    <div style={{
      background: C.surface, border: `1px solid ${C.border}`,
      borderRadius: 10, padding: '14px 16px', marginBottom: 16,
    }}>
      <SectionLabel>Select a game</SectionLabel>

      <input
        value={query}
        onChange={e => setQuery(e.target.value)}
        placeholder="Search teams…"
        style={{
          width: '100%', boxSizing: 'border-box',
          background: C.card, border: `1px solid ${C.border2}`,
          borderRadius: 6, padding: '6px 10px', fontSize: 13,
          color: C.text, marginBottom: 10, outline: 'none',
        }}
      />

      {loading && <Spinner />}
      {error   && <ErrorCard msg={error} onRetry={load} />}

      {!loading && !error && filtered.length === 0 && (
        <div style={{ fontSize: 12, color: C.muted, textAlign: 'center', padding: '10px 0' }}>
          No games found{query ? ` matching "${query}"` : ' for today'}.
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {filtered.map(g => {
          const isActive = selected?.game_id === g.game_id
          return (
            <button key={g.game_id} onClick={() => onSelect(g)} style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              background: isActive ? '#1e8aff18' : C.card,
              border: `1px solid ${isActive ? C.accent : C.border}`,
              borderRadius: 7, padding: '8px 12px', cursor: 'pointer', textAlign: 'left',
            }}>
              <span style={{ fontSize: 13, fontWeight: 600, color: C.text }}>
                {g.away_team} @ {g.home_team}
              </span>
              <span style={{ fontSize: 11, color: C.muted }}>
                {g.start_time_local} · {g.park_name}
              </span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

// ── Build Your Own ────────────────────────────────────────────────────────────
function BuildYourOwn({
  game,
  bankroll,
}: {
  game: GameStub
  bankroll: number
}) {
  const [legs,     setLegs]     = useState<GameLeg[]>([])
  const [loading,  setLoading]  = useState(true)
  const [error,    setError]    = useState<string | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [result,   setResult]   = useState<ParlayEvalResult | null>(null)
  const [calcBusy, setCalcBusy] = useState(false)
  const [calcErr,  setCalcErr]  = useState<string | null>(null)

  const loadLegs = () => {
    setLoading(true); setError(null); setSelected(new Set()); setResult(null)
    fetchGameLegs(game.game_id, { nSims: 10_000 })
      .then(env => setLegs(env.data.legs))
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }

  useEffect(() => { loadLegs() }, [game.game_id])

  const toggle = (id: string) => {
    setSelected(prev => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }

  const calculate = () => {
    const picked = legs.filter(l => selected.has(l.leg_id))
    if (picked.length < 2) { setCalcErr('Select at least 2 legs.'); return }
    const payload: ParlayLegInput[] = picked.map(l => ({
      leg_id:      l.leg_id,
      game_id:     game.game_id,
      description: l.description,
      side:        l.side,
      market_type: l.market_type,
      model_prob:  l.model_prob,
      market_odds: l.current_odds,
      line:        l.line,
    }))
    setCalcBusy(true); setCalcErr(null); setResult(null)
    fetchEvaluateParlay(payload, bankroll)
      .then(env => setResult(env.data))
      .catch(e => setCalcErr(String(e)))
      .finally(() => setCalcBusy(false))
  }

  if (loading) return <Spinner />
  if (error)   return <ErrorCard msg={error} onRetry={loadLegs} />
  if (legs.length === 0) {
    return (
      <div style={{ textAlign: 'center', color: C.muted, padding: '24px 0', fontSize: 13 }}>
        No legs available for this game yet. The simulation may still be warming up.
      </div>
    )
  }

  return (
    <div>
      {/* Leg checklist */}
      <div style={{
        background: C.surface, border: `1px solid ${C.border}`,
        borderRadius: 10, padding: '14px 16px', marginBottom: 12,
      }}>
        <SectionLabel>Available legs — click to select ({selected.size} selected)</SectionLabel>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {legs.map(leg => {
            const checked = selected.has(leg.leg_id)
            const hasEdge = leg.edge !== null && leg.edge > 0
            return (
              <label key={leg.leg_id} style={{
                display: 'flex', alignItems: 'center', gap: 10,
                background: checked ? '#1e8aff12' : C.card,
                border: `1px solid ${checked ? C.accent : C.border}`,
                borderRadius: 7, padding: '7px 12px', cursor: 'pointer',
              }}>
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => toggle(leg.leg_id)}
                  style={{ accentColor: C.accent, width: 14, height: 14 }}
                />
                <span style={{ flex: 1, fontSize: 13, color: C.text }}>
                  {leg.description}
                </span>
                <span style={{ fontSize: 11, color: C.muted2, minWidth: 40, textAlign: 'right' }}>
                  {fmtOdds(leg.current_odds)}
                </span>
                <span style={{ fontSize: 11, fontWeight: 600,
                  color: hasEdge ? C.pos : C.muted,
                  minWidth: 52, textAlign: 'right' }}>
                  {leg.edge !== null ? `edge ${pct(leg.edge, 1)}` : pct(leg.model_prob)}
                </span>
              </label>
            )
          })}
        </div>
        <div style={{ marginTop: 12, display: 'flex', gap: 10, alignItems: 'center' }}>
          <button
            onClick={calculate}
            disabled={calcBusy || selected.size < 2}
            style={{
              background: selected.size >= 2 ? C.accent : C.border2,
              color: '#fff', border: 'none', borderRadius: 7,
              padding: '8px 24px', fontWeight: 700, fontSize: 13,
              cursor: selected.size >= 2 ? 'pointer' : 'not-allowed',
              opacity: calcBusy ? 0.7 : 1,
            }}
          >
            {calcBusy ? 'Calculating…' : `Calculate (${selected.size} leg${selected.size !== 1 ? 's' : ''})`}
          </button>
          {selected.size > 0 && (
            <button onClick={() => setSelected(new Set())} style={{
              background: 'none', border: `1px solid ${C.border2}`, borderRadius: 7,
              padding: '7px 14px', fontSize: 12, color: C.muted, cursor: 'pointer',
            }}>Clear</button>
          )}
        </div>
        {calcErr && <div style={{ marginTop: 8 }}><ErrorCard msg={calcErr} /></div>}
      </div>

      {/* Results */}
      {result && !calcBusy && (
        <div>
          {/* Summary strip */}
          <div style={{
            background: C.surface, border: `1px solid ${C.border}`,
            borderRadius: 10, padding: '12px 16px',
            display: 'flex', flexWrap: 'wrap', gap: 20, marginBottom: 12,
          }}>
            {[
              { label: 'Corr-adj prob',  v: pct(result.joint_prob_corr_adjusted), color: C.accent },
              { label: 'Naive (indep)',  v: pct(result.joint_prob_naive),          color: C.muted  },
              { label: 'Net payout (b)', v: `${result.parlay_net_payout.toFixed(2)}×`, color: C.pos },
              { label: 'Sims',           v: result.n_sims.toLocaleString(),        color: C.muted2 },
            ].map(item => (
              <div key={item.label}>
                <div style={{ fontSize: 22, fontWeight: 800, color: item.color, fontVariantNumeric: 'tabular-nums' }}>
                  {item.v}
                </div>
                <div style={{ fontSize: 10, color: C.muted, marginTop: 2 }}>{item.label}</div>
              </div>
            ))}
          </div>

          <KellyComparison
            corrVariant={result.kelly.corr_adjusted}
            naiveVariant={result.kelly.naive}
            bankroll={bankroll}
            pctDiff={result.pct_diff_corr_vs_naive}
          />

          {/* Per-leg table */}
          <div style={{
            background: C.surface, border: `1px solid ${C.border}`,
            borderRadius: 10, padding: '12px 16px', marginBottom: 12,
          }}>
            <SectionLabel>Per-leg marginal probabilities</SectionLabel>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                  {['Leg', 'Side', 'Market', 'Odds', 'Sim prob'].map(h => (
                    <th key={h} style={{ padding: '5px 10px', textAlign: 'left', color: C.muted, fontWeight: 600 }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.legs.map((l, i) => (
                  <tr key={i} style={{
                    borderBottom: `1px solid ${C.border}`,
                    background: i % 2 === 0 ? 'transparent' : '#ffffff06',
                  }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600, color: C.text }}>{l.description}</td>
                    <td style={{ padding: '6px 10px', color: C.muted }}>{l.side}</td>
                    <td style={{ padding: '6px 10px', color: C.muted }}>{l.market_type}</td>
                    <td style={{ padding: '6px 10px', color: C.text, fontVariantNumeric: 'tabular-nums' }}>
                      {fmtOdds(l.market_odds)}
                    </td>
                    <td style={{ padding: '6px 10px', fontWeight: 700, color: C.accent, fontVariantNumeric: 'tabular-nums' }}>
                      {pct(l.marginal_prob)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Group breakdown (only interesting for cross-game) */}
          {result.groups.length > 1 && (
            <div style={{
              background: C.surface, border: `1px solid ${C.border}`,
              borderRadius: 10, padding: '12px 16px',
            }}>
              <SectionLabel>Group breakdown</SectionLabel>
              {result.groups.map((g, i) => (
                <div key={i} style={{
                  display: 'flex', justifyContent: 'space-between',
                  fontSize: 12, padding: '5px 0',
                  borderBottom: i < result.groups.length - 1 ? `1px solid ${C.border}` : 'none',
                }}>
                  <span style={{ color: C.muted }}>
                    Game {g.game_id} ({g.leg_ids.length} leg{g.leg_ids.length !== 1 ? 's' : ''})
                    {g.note ? <em style={{ color: C.muted2 }}> — {g.note}</em> : null}
                  </span>
                  <span style={{ color: C.accent, fontWeight: 600 }}>
                    corr {pct(g.joint_prob_corr)} / naive {pct(g.joint_prob_naive)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Suggested parlay card ─────────────────────────────────────────────────────
function ParlayCard({ p }: { p: SuggestedParlay }) {
  const ciWidth = p.ci_high - p.ci_low
  const ciBar   = Math.min(ciWidth / 0.1, 1)   // saturate at 10% CI width
  const evColor = p.ev > 0 ? C.pos : p.ev < 0 ? C.neg : C.muted

  return (
    <div style={{
      background: C.card, border: `1px solid ${C.border}`,
      borderRadius: 10, padding: '14px 16px', marginBottom: 10,
    }}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <div style={{
          background: '#1e8aff28', color: C.accent,
          borderRadius: 12, padding: '2px 10px', fontSize: 11, fontWeight: 700,
        }}>
          #{p.rank}
        </div>
        <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
          <span style={{ fontSize: 12, color: evColor, fontWeight: 700 }}>
            EV {p.ev >= 0 ? '+' : ''}{(p.ev * 100).toFixed(2)}%
          </span>
          <span style={{ fontSize: 12, color: C.pos, fontWeight: 700 }}>
            Kelly ${p.kelly_stake.toFixed(0)}
          </span>
          <span style={{ fontSize: 11, color: C.muted }}>
            {pct(p.joint_prob_corr)} joint
          </span>
        </div>
      </div>

      {/* Legs */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginBottom: 10 }}>
        {p.legs.map((l, i) => (
          <div key={i} style={{
            display: 'flex', justifyContent: 'space-between',
            background: C.surface, borderRadius: 6, padding: '5px 10px', fontSize: 12,
          }}>
            <span style={{ color: C.text }}>{l.description}</span>
            <span style={{ color: C.muted2, display: 'flex', gap: 12 }}>
              <span>{fmtOdds(l.market_odds)}</span>
              <span style={{ color: C.accent }}>{pct(l.model_prob)}</span>
            </span>
          </div>
        ))}
      </div>

      {/* Stats footer */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        borderTop: `1px solid ${C.border}`, paddingTop: 8,
      }}>
        <div>
          <span style={{ fontSize: 10, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Net payout
          </span>
          <span style={{ fontSize: 12, fontWeight: 700, color: C.text, marginLeft: 6 }}>
            {p.parlay_net_payout.toFixed(2)}×
          </span>
        </div>
        {/* CI bar */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 3, alignItems: 'flex-end' }}>
          <div style={{ width: 100, height: 4, background: C.border2, borderRadius: 2 }}>
            <div style={{
              width: `${(1 - ciBar) * 100}%`, height: '100%',
              background: ciBar < 0.4 ? C.pos : ciBar < 0.7 ? C.amber : C.neg,
              borderRadius: 2,
            }} />
          </div>
          <span style={{ fontSize: 10, color: C.muted2 }}>
            CI [{pct(p.ci_low, 1)}, {pct(p.ci_high, 1)}]
          </span>
        </div>
      </div>
    </div>
  )
}

// ── Suggested Parlays pane ────────────────────────────────────────────────────
function SuggestedPane({
  mode, bankroll,
}: {
  mode: 'sgp' | 'crossgame'
  bankroll: number
}) {
  const [data,    setData]    = useState<SuggestedParlaysData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState<string | null>(null)

  const load = () => {
    setLoading(true); setError(null); setData(null)
    fetchSuggestedParlays(mode, bankroll, { nSims: 6_000 })
      .then(env => setData(env.data))
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }

  // Auto-load when mode changes
  useEffect(() => { load() }, [mode])

  if (loading) return <Spinner />
  if (error)   return <ErrorCard msg={error} onRetry={load} />

  if (!data) {
    return (
      <div style={{ textAlign: 'center', color: C.muted, padding: '24px 0', fontSize: 13 }}>
        <button onClick={load} style={{
          background: C.accent, color: '#fff', border: 'none', borderRadius: 7,
          padding: '8px 20px', fontSize: 13, fontWeight: 700, cursor: 'pointer',
        }}>
          Generate suggestions
        </button>
      </div>
    )
  }

  return (
    <div>
      {/* Skipped games notice */}
      {data.skipped_games.length > 0 && (
        <details style={{ marginBottom: 12 }}>
          <summary style={{ fontSize: 11, color: C.muted, cursor: 'pointer' }}>
            {data.skipped_games.length} game(s) skipped
          </summary>
          <div style={{ marginTop: 6, display: 'flex', flexDirection: 'column', gap: 4 }}>
            {data.skipped_games.map((s, i) => (
              <div key={i} style={{ fontSize: 11, color: C.muted2, paddingLeft: 12 }}>
                Game {s.game_id}: {s.reason}
              </div>
            ))}
          </div>
        </details>
      )}

      {/* Empty state */}
      {data.parlays.length === 0 ? (
        <div style={{
          background: C.surface, border: `1px solid ${C.border}`,
          borderRadius: 10, padding: '24px 16px', textAlign: 'center',
        }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: C.muted, marginBottom: 8 }}>
            No parlays generated
          </div>
          <div style={{ fontSize: 12, color: C.muted2, maxWidth: 360, margin: '0 auto' }}>
            {data.message ??
              (mode === 'sgp'
                ? 'The simulation cache may not be built yet for today\'s games. Try loading the board for a specific game first, then come back here.'
                : 'Need odds data for at least 2 games to build cross-game parlays.')}
          </div>
          <button onClick={load} style={{
            marginTop: 14, background: 'none', border: `1px solid ${C.border2}`,
            borderRadius: 7, padding: '7px 16px', fontSize: 12, color: C.muted, cursor: 'pointer',
          }}>Retry</button>
        </div>
      ) : (
        <div>
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10,
          }}>
            <SectionLabel>
              {data.parlays.length} suggestion{data.parlays.length !== 1 ? 's' : ''} · ranked by EV
            </SectionLabel>
            <button onClick={load} style={{
              background: 'none', border: `1px solid ${C.border2}`, borderRadius: 6,
              padding: '4px 12px', fontSize: 11, color: C.muted, cursor: 'pointer',
            }}>Refresh</button>
          </div>
          {data.parlays.map(p => (
            <ParlayCard key={p.rank} p={p} />
          ))}
        </div>
      )}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────
export default function SGPTab() {
  const [selectedGame, setSelectedGame] = useState<GameStub | null>(null)
  const [topTab,       setTopTab]       = useState<'build' | 'suggested'>('build')
  const [suggMode,     setSuggMode]     = useState<'sgp' | 'crossgame'>('sgp')
  const [bankroll,     setBankroll]     = useState(1000)

  return (
    <div style={{ maxWidth: 720 }}>
      {/* Title */}
      <div style={{ marginBottom: 18 }}>
        <div style={{ fontSize: 20, fontWeight: 800, color: C.text, marginBottom: 4 }}>
          SGP / Parlay Builder
        </div>
        <div style={{ fontSize: 12, color: C.muted }}>
          Build your own or get optimizer-generated suggestions — correlation-adjusted via Monte Carlo.
        </div>
      </div>

      {/* Bankroll input — global */}
      <div style={{
        background: C.surface, border: `1px solid ${C.border}`,
        borderRadius: 8, padding: '10px 14px', marginBottom: 12,
        display: 'flex', alignItems: 'center', gap: 12,
      }}>
        <span style={{ fontSize: 12, color: C.muted }}>Bankroll $</span>
        <input
          type="number" min={1} value={bankroll}
          onChange={e => setBankroll(Math.max(1, parseFloat(e.target.value) || 1000))}
          style={{
            width: 90, background: C.card, border: `1px solid ${C.border2}`,
            borderRadius: 6, padding: '5px 8px', fontSize: 13, color: C.text,
          }}
        />
      </div>

      {/* Game picker */}
      <GamePicker selected={selectedGame} onSelect={setSelectedGame} />

      {/* Top tab selector */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
        <TabPill label="Build Your Own"      active={topTab === 'build'}     onClick={() => setTopTab('build')} />
        <TabPill label="Suggested Parlays"   active={topTab === 'suggested'} onClick={() => setTopTab('suggested')} />
      </div>

      {/* Build Your Own */}
      {topTab === 'build' && (
        selectedGame
          ? <BuildYourOwn game={selectedGame} bankroll={bankroll} />
          : (
            <div style={{
              background: C.surface, border: `1px solid ${C.border}`,
              borderRadius: 10, padding: '28px 16px', textAlign: 'center',
              color: C.muted, fontSize: 13,
            }}>
              Select a game above to see available legs.
            </div>
          )
      )}

      {/* Suggested Parlays */}
      {topTab === 'suggested' && (
        <div>
          {/* Sub-tabs */}
          <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
            <TabPill label="Same-Game (SGP)"  active={suggMode === 'sgp'}       onClick={() => setSuggMode('sgp')} />
            <TabPill label="Cross-Game"        active={suggMode === 'crossgame'} onClick={() => setSuggMode('crossgame')} />
          </div>
          <SuggestedPane mode={suggMode} bankroll={bankroll} />
        </div>
      )}
    </div>
  )
}
