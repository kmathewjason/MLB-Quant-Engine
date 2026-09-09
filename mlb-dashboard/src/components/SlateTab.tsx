/**
 * SlateTab — fast game list from GET /api/games/today.
 * No simulation, no odds — loads in <1 s.
 * Clicking a game opens GameBoardView.
 */
import { useState, useEffect, useMemo } from 'react'
import { fetchGamesToday } from '../api'
import type { GameStub } from '../types'
import GameBoardView from './GameBoardView'

// ── design tokens ─────────────────────────────────────────────────────────
const C = {
  bg:      '#0f1117',
  surface: '#16181f',
  card:    '#1a1d27',
  border:  '#262a36',
  border2: '#2a2d38',
  text:    '#e8eaf0',
  muted:   '#8892a4',
  muted2:  '#4b5563',
  accent:  '#4faeff',
  green:   '#34d399',
  amber:   '#fbbf24',
  red:     '#f87171',
}
const label = {
  fontSize: 9, fontWeight: 700 as const,
  letterSpacing: '0.06em', color: C.muted2, textTransform: 'uppercase' as const,
}

// ── status badge ──────────────────────────────────────────────────────────
function StatusBadge({ status }: { status: string }) {
  const s = status.toLowerCase()
  const isLive = s.includes('progress') || s.includes('live')
  const isDone = s.includes('final') || s.includes('complete')
  const bg  = isLive ? '#1a2e1a' : isDone ? '#1a1a2e' : '#1c2230'
  const fg  = isLive ? C.green   : isDone ? '#9ca3af' : C.accent
  const dot = isLive ? C.green   : null
  return (
    <span style={{
      background: bg, color: fg, borderRadius: 4,
      padding: '2px 7px', fontSize: 9, fontWeight: 700, letterSpacing: '0.06em',
      display: 'inline-flex', alignItems: 'center', gap: 4,
    }}>
      {dot && <span style={{ width: 5, height: 5, borderRadius: '50%', background: dot, flexShrink: 0 }} />}
      {status.toUpperCase()}
    </span>
  )
}

// ── skeleton card ─────────────────────────────────────────────────────────
function SkeletonCard() {
  return (
    <div style={{
      background: C.card, border: `1px solid ${C.border}`,
      borderRadius: 12, padding: '16px 18px', height: 110,
      opacity: 0.5,
    }}>
      <div style={{ width: '45%', height: 14, background: C.border2, borderRadius: 4, marginBottom: 10 }} />
      <div style={{ width: '60%', height: 11, background: C.border2, borderRadius: 4, marginBottom: 8 }} />
      <div style={{ width: '30%', height: 11, background: C.border2, borderRadius: 4 }} />
    </div>
  )
}

// ── game card ─────────────────────────────────────────────────────────────
function GameCard({ game, onClick }: { game: GameStub; onClick: () => void }) {
  const [hov, setHov] = useState(false)
  const hasPitchers = game.probable_pitcher_away !== 'TBD' || game.probable_pitcher_home !== 'TBD'
  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        background: hov ? '#1e2130' : C.card,
        border: `1px solid ${hov ? '#3a4259' : C.border}`,
        borderRadius: 12, padding: '15px 18px',
        cursor: 'pointer', transition: 'border-color 0.12s, background 0.12s',
      }}
    >
      {/* Top row: matchup + time + status */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 9 }}>
        <div>
          <div style={{ fontSize: 14, fontWeight: 700, color: '#fff', marginBottom: 2 }}>
            <span style={{ color: C.muted2, fontSize: 12, fontWeight: 400 }}>
              {game.away_team.split(' ').pop()}
            </span>
            <span style={{ color: C.muted2, margin: '0 7px' }}>@</span>
            <span style={{ color: '#fff' }}>
              {game.home_team.split(' ').pop()}
            </span>
          </div>
          <div style={{ fontSize: 11, color: C.muted2 }}>
            {game.away_team} @ {game.home_team}
          </div>
        </div>
        <div style={{ textAlign: 'right', display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 5 }}>
          <StatusBadge status={game.status || 'Scheduled'} />
          <span style={{ fontSize: 11, color: C.muted }}>{game.start_time_local}</span>
        </div>
      </div>

      {/* Bottom row: pitchers + park */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ fontSize: 11, color: C.muted }}>
          {hasPitchers ? (
            <>
              <span style={{ color: C.muted2, marginRight: 4 }}>SP:</span>
              {game.probable_pitcher_away.split(' ').pop()} vs {game.probable_pitcher_home.split(' ').pop()}
            </>
          ) : (
            <span style={{ color: C.muted2 }}>Pitchers TBD</span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ fontSize: 10, color: C.muted2 }}>{game.park_name}</span>
          <span style={{
            fontSize: 9, background: '#1c2230', color: C.accent,
            borderRadius: 3, padding: '1px 6px', fontWeight: 600,
          }}>
            View Board →
          </span>
        </div>
      </div>
    </div>
  )
}

// ── inline error card ─────────────────────────────────────────────────────
function ErrorCard({ msg, onRetry }: { msg: string; onRetry: () => void }) {
  return (
    <div style={{
      background: '#1e0e0e', border: `1px solid #6b2020`,
      borderRadius: 10, padding: '18px 20px', color: C.red,
    }}>
      <div style={{ fontWeight: 700, marginBottom: 6, fontSize: 13 }}>Failed to load games</div>
      <div style={{ fontSize: 12, marginBottom: 12, color: '#e8938a' }}>{msg}</div>
      <button
        onClick={onRetry}
        style={{
          background: '#3a1010', border: `1px solid #7f2020`, borderRadius: 6,
          color: C.red, padding: '6px 14px', fontSize: 12, fontWeight: 600, cursor: 'pointer',
        }}
      >
        Retry
      </button>
    </div>
  )
}

// ── main ──────────────────────────────────────────────────────────────────
export default function SlateTab() {
  const [games,   setGames]   = useState<GameStub[]>([])
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState<string | null>(null)
  const [date,    setDate]    = useState('')
  const [search,  setSearch]  = useState('')
  const [selected, setSelected] = useState<GameStub | null>(null)

  const load = (d?: string) => {
    setLoading(true); setError(null)
    fetchGamesToday(d || undefined)
      .then(env => setGames(env.data))
      .catch(e => {
        console.error('[SlateTab] fetchGamesToday failed:', e)
        setError(String(e))
      })
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])  // eslint-disable-line react-hooks/exhaustive-deps

  const filtered = useMemo(() => {
    const q = search.toLowerCase().trim()
    if (!q) return games
    return games.filter(g =>
      g.home_team.toLowerCase().includes(q) ||
      g.away_team.toLowerCase().includes(q) ||
      g.probable_pitcher_home.toLowerCase().includes(q) ||
      g.probable_pitcher_away.toLowerCase().includes(q) ||
      g.park_name.toLowerCase().includes(q)
    )
  }, [games, search])

  // If a game is selected, render the board view instead
  if (selected) {
    return (
      <GameBoardView
        game={selected}
        onBack={() => setSelected(null)}
      />
    )
  }

  return (
    <div style={{ maxWidth: 820 }}>

      {/* ── Page header ─────────────────────────────────────────────── */}
      <div style={{ marginBottom: 22 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#fff', margin: '0 0 4px' }}>
          Today's Slate
        </h1>
        <p style={{ color: C.muted2, fontSize: 13, margin: 0 }}>
          {games.length > 0
            ? `${games.length} game${games.length !== 1 ? 's' : ''} · click any game to open the full market board`
            : 'Click a game to run the full simulation and view betting markets'}
        </p>
      </div>

      {/* ── Controls ────────────────────────────────────────────────── */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 20, flexWrap: 'wrap', alignItems: 'flex-end' }}>

        {/* Search */}
        <div style={{ flex: 2, minWidth: 180 }}>
          <div style={{ ...label, marginBottom: 5 }}>Search team or pitcher</div>
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="e.g. Dodgers, Cole, Yankees…"
            style={{
              width: '100%', background: C.card, border: `1px solid ${C.border2}`,
              borderRadius: 8, padding: '8px 12px', color: C.text, fontSize: 13,
              outline: 'none', boxSizing: 'border-box',
            }}
          />
        </div>

        {/* Date */}
        <div style={{ flex: 1, minWidth: 140 }}>
          <div style={{ ...label, marginBottom: 5 }}>Date</div>
          <input
            type="date" value={date}
            onChange={e => { setDate(e.target.value); load(e.target.value) }}
            style={{
              width: '100%', background: C.card, border: `1px solid ${C.border2}`,
              borderRadius: 8, padding: '8px 10px', color: C.text, fontSize: 13,
              outline: 'none', boxSizing: 'border-box', colorScheme: 'dark',
            }}
          />
        </div>

        {/* Refresh */}
        <button
          onClick={() => load(date || undefined)}
          disabled={loading}
          style={{
            background: loading ? '#1e3358' : C.accent,
            color: loading ? '#4faeff66' : '#fff',
            border: 'none', borderRadius: 8, padding: '8px 18px',
            fontWeight: 700, fontSize: 13, cursor: loading ? 'not-allowed' : 'pointer',
            alignSelf: 'flex-end', whiteSpace: 'nowrap',
          }}
        >
          {loading ? '⏳ Loading…' : '↻ Refresh'}
        </button>
      </div>

      {/* ── Error ────────────────────────────────────────────────────── */}
      {error && !loading && (
        <div style={{ marginBottom: 16 }}>
          <ErrorCard msg={error} onRetry={() => load(date || undefined)} />
        </div>
      )}

      {/* ── Loading skeletons ─────────────────────────────────────────── */}
      {loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {[1, 2, 3, 4, 5].map(i => <SkeletonCard key={i} />)}
        </div>
      )}

      {/* ── Empty state ───────────────────────────────────────────────── */}
      {!loading && !error && games.length === 0 && (
        <div style={{
          textAlign: 'center', padding: '60px 20px', color: C.muted2,
        }}>
          <div style={{ fontSize: 36, marginBottom: 12 }}>⚾</div>
          <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 6, color: C.muted }}>
            No games found
          </div>
          <div style={{ fontSize: 12 }}>
            Try a different date, or check that the API server is running.
          </div>
        </div>
      )}

      {/* ── No search match ───────────────────────────────────────────── */}
      {!loading && games.length > 0 && filtered.length === 0 && (
        <div style={{ textAlign: 'center', padding: '40px 20px', color: C.muted2 }}>
          No games matching "{search}"
        </div>
      )}

      {/* ── Game list ─────────────────────────────────────────────────── */}
      {!loading && filtered.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {filtered.map(g => (
            <GameCard
              key={g.game_id}
              game={g}
              onClick={() => setSelected(g)}
            />
          ))}
        </div>
      )}

      {/* Footer note */}
      {!loading && games.length > 0 && (
        <div style={{ marginTop: 16, fontSize: 10, color: C.muted2, textAlign: 'right' }}>
          Schedule data via MLB Stats API · click any game for market board
        </div>
      )}

    </div>
  )
}
