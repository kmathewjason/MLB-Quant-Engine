/**
 * SlateTab — Today's game slate with search, game cards, and ranked
 * betting opportunities sorted by model confidence.
 */
import { useState, useEffect, useMemo } from 'react'
import { fetchDailyPredictions } from '../api'
import type { DailyGame, MarketRow } from '../types'
import GameDetailModal from './GameDetailModal'

// ── colour helpers ─────────────────────────────────────────────────────────

const S = {
  page:        { color: '#e8eaf0' },
  card:        { background: '#1a1d27', border: '1px solid #262a36', borderRadius: 12 },
  cardHover:   { background: '#1e2130' },
  label:       { fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', color: '#5a6072', textTransform: 'uppercase' as const },
  muted:       { color: '#6b7280', fontSize: 12 },
  accent:      { color: '#4faeff' },
  green:       { color: '#34d399' },
  red:         { color: '#f87171' },
  amber:       { color: '#fbbf24' },
  surface:     { background: '#12141b' },
  divider:     { borderTop: '1px solid #1f2230' },
}

// ── confidence badge ───────────────────────────────────────────────────────

function ConfBadge({ prob }: { prob: number }) {
  const pct = Math.round(prob * 100)
  // Confidence = how far the model is from 50/50 (0–50 scale, capped at 50)
  const conf = Math.min(50, Math.abs(pct - 50))
  let bg: string, fg: string, label: string
  if (conf >= 15) { bg = '#0d3326'; fg = '#34d399'; label = 'HIGH CONF' }
  else if (conf >= 8) { bg = '#2a1f07'; fg = '#fbbf24'; label = 'MEDIUM' }
  else { bg = '#1e1e2e'; fg = '#6b7280'; label = 'COIN FLIP' }
  return (
    <span style={{
      background: bg, color: fg, borderRadius: 4, padding: '2px 6px',
      fontSize: 9, fontWeight: 700, letterSpacing: '0.07em',
    }}>
      {label}
    </span>
  )
}

// ── win probability bar ────────────────────────────────────────────────────

function WinBar({ home, homeTeam, awayTeam }: {
  home: number; homeTeam: string; awayTeam: string
}) {
  const hp = Math.round(home * 100)
  const ap = 100 - hp
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: 11, color: '#9ca3af' }}>{awayTeam.split(' ').pop()} {ap}%</span>
        <span style={{ fontSize: 11, color: '#9ca3af' }}>{hp}% {homeTeam.split(' ').pop()}</span>
      </div>
      <div style={{ height: 6, borderRadius: 6, background: '#262a36', overflow: 'hidden', display: 'flex' }}>
        <div style={{
          width: `${ap}%`, background: hp >= ap ? '#374151' : '#4faeff',
          transition: 'width 0.4s', borderRadius: '6px 0 0 6px',
        }} />
        <div style={{
          width: `${hp}%`, background: hp >= ap ? '#4faeff' : '#374151',
          transition: 'width 0.4s', borderRadius: '0 6px 6px 0',
        }} />
      </div>
    </div>
  )
}

// ── ranked bet row ─────────────────────────────────────────────────────────

function BetRow({ m, rank }: { m: MarketRow; rank: number }) {
  const hasOdds = m.ev !== null
  const conf = Math.round(m.model_prob * 100)
  const edgePct = m.edge !== null ? (m.edge * 100).toFixed(1) : null
  const evPct   = m.ev   !== null ? (m.ev   * 100).toFixed(1) : null

  const rankColor = rank === 1 ? '#fbbf24' : rank === 2 ? '#9ca3af' : rank === 3 ? '#cd7c2e' : '#4b5563'

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 10,
      padding: '8px 12px', borderBottom: '1px solid #1a1d27',
    }}>
      {/* Rank */}
      <span style={{
        width: 20, height: 20, borderRadius: '50%', background: rankColor,
        color: rank <= 3 ? '#000' : '#9ca3af', fontSize: 10, fontWeight: 700,
        display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
      }}>
        {rank}
      </span>

      {/* Label */}
      <span style={{ flex: 1, fontSize: 12.5, color: '#d1d5db', fontWeight: 500 }}>
        {m.label}
      </span>

      {/* Model % */}
      <span style={{ width: 42, textAlign: 'right', fontSize: 12, fontWeight: 700, color: '#e8eaf0' }}>
        {conf}%
      </span>

      {/* Edge */}
      {hasOdds ? (
        <span style={{
          width: 52, textAlign: 'right', fontSize: 11.5, fontWeight: 600,
          color: (m.edge ?? 0) >= 0 ? '#34d399' : '#f87171',
        }}>
          {(m.edge ?? 0) >= 0 ? '+' : ''}{edgePct}%
        </span>
      ) : (
        <span style={{ width: 52, textAlign: 'right', fontSize: 11, color: '#3d4455' }}>
          no odds
        </span>
      )}

      {/* EV */}
      {hasOdds ? (
        <span style={{
          width: 48, textAlign: 'right', fontSize: 11, fontWeight: 600,
          color: (m.ev ?? 0) >= 0.01 ? '#34d399' : '#6b7280',
        }}>
          {(m.ev ?? 0) >= 0 ? '+' : ''}{evPct}%
        </span>
      ) : (
        <span style={{ width: 48, textAlign: 'right', color: '#2d3040', fontSize: 10 }}>—</span>
      )}

      {/* Stake */}
      {m.stake_units !== null ? (
        <span style={{
          width: 50, textAlign: 'right', fontSize: 12, fontWeight: 700, color: '#4faeff',
        }}>
          ${Math.round(m.stake_units)}
        </span>
      ) : (
        <span style={{ width: 50, textAlign: 'right', color: '#2d3040', fontSize: 10 }}>—</span>
      )}
    </div>
  )
}

// ── game card ──────────────────────────────────────────────────────────────

function GameCard({ game, onSelect }: { game: DailyGame; onSelect: () => void }) {
  const sim = game.simulation
  const topBets = useMemo(() => {
    return [...game.markets]
      .sort((a, b) => {
        // Sort by model confidence distance from 50%
        const ca = Math.abs(a.model_prob - 0.5)
        const cb = Math.abs(b.model_prob - 0.5)
        if (a.ev !== null && b.ev !== null) return (b.ev - a.ev)
        if (a.ev !== null) return -1
        if (b.ev !== null) return 1
        return cb - ca
      })
      .slice(0, 4)
  }, [game.markets])

  const homeWin = sim.home_win_prob >= 0.5
  const favProb = homeWin ? sim.home_win_prob : sim.away_win_prob

  return (
    <div
      onClick={onSelect}
      style={{
        ...S.card,
        cursor: 'pointer',
        transition: 'border-color 0.15s, background 0.15s',
        overflow: 'hidden',
      }}
      onMouseEnter={e => {
        (e.currentTarget as HTMLElement).style.borderColor = '#3a4259'
        ;(e.currentTarget as HTMLElement).style.background = '#1e2130'
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLElement).style.borderColor = '#262a36'
        ;(e.currentTarget as HTMLElement).style.background = '#1a1d27'
      }}
    >
      {/* Header row */}
      <div style={{ padding: '14px 16px 12px', borderBottom: '1px solid #1f2230' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
          {/* Matchup */}
          <div>
            <div style={{ fontSize: 13.5, fontWeight: 700, color: '#fff', marginBottom: 2 }}>
              {game.away_team} <span style={{ color: '#4b5563', fontWeight: 400 }}>@</span> {game.home_team}
            </div>
            <div style={{ fontSize: 11, color: '#4b5563' }}>
              {game.game_date}
              {game.away_probable_pitcher !== 'TBD' && (
                <span style={{ marginLeft: 8 }}>
                  {game.away_probable_pitcher.split(' ').pop()} vs {game.home_probable_pitcher.split(' ').pop()}
                </span>
              )}
            </div>
          </div>
          {/* Status + confidence */}
          <div style={{ textAlign: 'right', display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4 }}>
            <span style={{
              fontSize: 9, background: '#1c2230', color: '#4faeff',
              borderRadius: 4, padding: '2px 6px', fontWeight: 600, letterSpacing: '0.05em',
            }}>
              {game.status.toUpperCase()}
            </span>
            <ConfBadge prob={favProb} />
          </div>
        </div>

        {/* Win probability bar */}
        <WinBar
          home={sim.home_win_prob}
          homeTeam={game.home_team} awayTeam={game.away_team}
        />
      </div>

      {/* Run totals row */}
      <div style={{
        display: 'flex', padding: '8px 16px', gap: 20,
        borderBottom: '1px solid #1a1d27', background: '#15171f',
      }}>
        {[
          { label: `${game.away_team.split(' ').pop()} Runs`, val: sim.away_runs.mean.toFixed(1) },
          { label: `${game.home_team.split(' ').pop()} Runs`, val: sim.home_runs.mean.toFixed(1) },
          { label: 'Total', val: sim.total_runs.mean.toFixed(1) },
          { label: 'Spread Cover', val: `${Math.round((sim.spread['home_cover_prob_m1.5'] ?? 0.5) * 100)}%` },
        ].map(item => (
          <div key={item.label} style={{ flex: 1, textAlign: 'center' }}>
            <div style={{ fontSize: 13.5, fontWeight: 700, color: '#e8eaf0' }}>{item.val}</div>
            <div style={{ ...S.label, marginTop: 1 }}>{item.label}</div>
          </div>
        ))}
      </div>

      {/* Ranked bets */}
      <div>
        {/* Column header */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10,
          padding: '6px 12px', background: '#12141b',
        }}>
          <span style={{ width: 20 }} />
          <span style={{ flex: 1, ...S.label }}>Market</span>
          <span style={{ width: 42, textAlign: 'right', ...S.label }}>Model</span>
          <span style={{ width: 52, textAlign: 'right', ...S.label }}>Edge</span>
          <span style={{ width: 48, textAlign: 'right', ...S.label }}>EV</span>
          <span style={{ width: 50, textAlign: 'right', ...S.label }}>Stake</span>
        </div>
        {topBets.map((m, i) => <BetRow key={m.market} m={m} rank={i + 1} />)}
      </div>

      {/* Footer */}
      <div style={{
        padding: '7px 14px', background: '#12141b',
        fontSize: 10.5, color: '#3d4455', textAlign: 'right',
      }}>
        {sim.n_sims.toLocaleString()} simulations · click for full distribution →
      </div>
    </div>
  )
}

// ── main component ─────────────────────────────────────────────────────────

export default function SlateTab() {
  const [games,   setGames]   = useState<DailyGame[]>([])
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState<string | null>(null)
  const [date,    setDate]    = useState('')
  const [search,  setSearch]  = useState('')
  const [detail,  setDetail]  = useState<DailyGame | null>(null)
  const [nSims,   setNSims]   = useState(5_000)

  const load = (sims = nSims) => {
    setLoading(true); setError(null)
    fetchDailyPredictions(date || undefined, sims)
      .then(env => setGames(env.data))
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const filtered = useMemo(() => {
    const q = search.toLowerCase().trim()
    if (!q) return games
    return games.filter(g =>
      g.home_team.toLowerCase().includes(q) ||
      g.away_team.toLowerCase().includes(q) ||
      g.home_probable_pitcher.toLowerCase().includes(q) ||
      g.away_probable_pitcher.toLowerCase().includes(q)
    )
  }, [games, search])

  // Sort: highest confidence (furthest from 50/50) first
  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      const ca = Math.abs(a.simulation.home_win_prob - 0.5)
      const cb = Math.abs(b.simulation.home_win_prob - 0.5)
      return cb - ca
    })
  }, [filtered])

  const totalMarkets = games.reduce((s, g) => s + g.markets.length, 0)
  const positiveEV   = games.reduce((s, g) =>
    s + g.markets.filter(m => (m.ev ?? 0) > 0.01).length, 0)

  return (
    <div style={{ maxWidth: 900 }}>
      {/* ── Page header ─────────────────────────────────────────────── */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#fff', margin: 0 }}>
          Today's Slate
        </h1>
        <p style={{ color: '#5a6072', fontSize: 13, marginTop: 4 }}>
          Monte Carlo game simulations · Bayesian matchup priors · Kelly-sized opportunities
        </p>
      </div>

      {/* ── Summary strip ───────────────────────────────────────────── */}
      {!loading && games.length > 0 && (
        <div style={{
          display: 'flex', gap: 12, marginBottom: 20,
        }}>
          {[
            { label: 'Games today',    val: games.length },
            { label: 'Markets analysed', val: totalMarkets },
            { label: '+EV opportunities', val: positiveEV, color: positiveEV > 0 ? '#34d399' : '#6b7280' },
            { label: 'Simulations / game', val: nSims.toLocaleString() },
          ].map(s => (
            <div key={s.label} style={{
              flex: 1, background: '#1a1d27', border: '1px solid #262a36',
              borderRadius: 10, padding: '12px 16px',
            }}>
              <div style={{ fontSize: 18, fontWeight: 800, color: (s as any).color ?? '#fff' }}>
                {s.val}
              </div>
              <div style={{ ...S.label, marginTop: 2 }}>{s.label}</div>
            </div>
          ))}
        </div>
      )}

      {/* ── Controls ────────────────────────────────────────────────── */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 20, flexWrap: 'wrap', alignItems: 'flex-end' }}>

        {/* Search */}
        <div style={{ flex: 2, minWidth: 180 }}>
          <div style={{ ...S.label, marginBottom: 5 }}>Search team or pitcher</div>
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="e.g. Dodgers, Cole, Yankees…"
            style={{
              width: '100%', background: '#1a1d27', border: '1px solid #2e3345',
              borderRadius: 8, padding: '8px 12px', color: '#e8eaf0', fontSize: 13,
              outline: 'none', boxSizing: 'border-box',
            }}
          />
        </div>

        {/* Date */}
        <div style={{ flex: 1, minWidth: 140 }}>
          <div style={{ ...S.label, marginBottom: 5 }}>Date</div>
          <input
            type="date" value={date} onChange={e => setDate(e.target.value)}
            style={{
              width: '100%', background: '#1a1d27', border: '1px solid #2e3345',
              borderRadius: 8, padding: '8px 10px', color: '#e8eaf0', fontSize: 13,
              outline: 'none', boxSizing: 'border-box',
              colorScheme: 'dark',
            }}
          />
        </div>

        {/* Sim quality */}
        <div>
          <div style={{ ...S.label, marginBottom: 5 }}>Sim quality</div>
          <select
            value={nSims}
            onChange={e => setNSims(Number(e.target.value))}
            style={{
              background: '#1a1d27', border: '1px solid #2e3345',
              borderRadius: 8, padding: '8px 10px', color: '#e8eaf0',
              fontSize: 13, cursor: 'pointer', outline: 'none',
            }}
          >
            <option value={1000}>1 k — fast preview</option>
            <option value={5000}>5 k — standard</option>
            <option value={20000}>20 k — precise</option>
            <option value={50000}>50 k — max precision</option>
          </select>
        </div>

        {/* Run button */}
        <button
          onClick={() => load(nSims)}
          disabled={loading}
          style={{
            background: loading ? '#1e3358' : '#1e5fc2',
            color: loading ? '#4faeff66' : '#fff',
            border: 'none', borderRadius: 8, padding: '8px 20px',
            fontWeight: 700, fontSize: 13.5, cursor: loading ? 'not-allowed' : 'pointer',
            alignSelf: 'flex-end', whiteSpace: 'nowrap',
          }}
        >
          {loading ? '⏳ Simulating…' : '▶ Run Analysis'}
        </button>
      </div>

      {/* ── Error ───────────────────────────────────────────────────── */}
      {error && (
        <div style={{
          background: '#2a0f0f', border: '1px solid #7f1d1d', borderRadius: 8,
          padding: '10px 14px', color: '#f87171', fontSize: 13, marginBottom: 16,
        }}>
          ⚠ {error}
        </div>
      )}

      {/* ── Loading skeleton ────────────────────────────────────────── */}
      {loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {[1, 2, 3].map(i => (
            <div key={i} style={{
              ...S.card, height: 200,
              background: 'linear-gradient(90deg, #1a1d27 25%, #1e2130 50%, #1a1d27 75%)',
              backgroundSize: '200% 100%',
              animation: 'pulse 1.4s ease-in-out infinite',
            }} />
          ))}
          <style>{`@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }`}</style>
        </div>
      )}

      {/* ── Empty state ──────────────────────────────────────────────── */}
      {!loading && games.length === 0 && !error && (
        <div style={{
          textAlign: 'center', padding: '60px 20px',
          color: '#4b5563', fontSize: 14,
        }}>
          <div style={{ fontSize: 40, marginBottom: 12 }}>⚾</div>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>No games loaded</div>
          <div style={{ fontSize: 12 }}>Choose a date and click Run Analysis</div>
        </div>
      )}

      {/* ── No search results ────────────────────────────────────────── */}
      {!loading && games.length > 0 && sorted.length === 0 && (
        <div style={{ textAlign: 'center', padding: '40px 20px', color: '#4b5563' }}>
          No games matching "{search}"
        </div>
      )}

      {/* ── Game cards — sorted by confidence ──────────────────────── */}
      {!loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {sorted.map(g => (
            <GameCard key={g.game_pk} game={g} onSelect={() => setDetail(g)} />
          ))}
        </div>
      )}

      {/* ── Detail modal ─────────────────────────────────────────────── */}
      {detail && (
        <GameDetailModal
          game={detail}
          defaultTotalLine={8.5}
          onClose={() => setDetail(null)}
        />
      )}
    </div>
  )
}
