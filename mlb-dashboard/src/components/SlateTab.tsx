/**
 * SlateTab — daily slate with sortable/filterable market table.
 *
 * Filters  : market type (h2h / total / spread), min edge %, min EV %
 * Sort     : click any column header; second click reverses
 * Game detail: click a row → GameDetailModal opens for that game_pk
 */
import { useState, useEffect, useMemo } from 'react'
import { fetchDailyPredictions } from '../api'
import type { DailyGame, MarketRow } from '../types'
import GameDetailModal from './GameDetailModal'

// ── helpers ────────────────────────────────────────────────────────────────

function pct(v: number | null, decimals = 1) {
  if (v === null) return <span className="text-gray-300">—</span>
  return <span>{(v * 100).toFixed(decimals)}%</span>
}

function signed(v: number | null) {
  if (v === null) return <span className="text-gray-300">—</span>
  const cls = v >= 0 ? 'text-pos bg-pos-bg' : 'text-neg bg-neg-bg'
  return (
    <span className={`inline-block px-1.5 py-0.5 rounded text-xs font-semibold ${cls}`}>
      {v >= 0 ? '+' : ''}{(v * 100).toFixed(1)}%
    </span>
  )
}

function odds(v: number | null) {
  if (v === null) return <span className="text-gray-300">—</span>
  return <span>{v > 0 ? `+${v}` : v}</span>
}

type SortKey = keyof MarketRow | 'game'
type SortDir = 'asc' | 'desc'

interface FlatRow extends MarketRow {
  game_pk: number
  matchup: string
  game_date: string
}

const MARKET_TYPE_OPTIONS = [
  { value: '',       label: 'All markets' },
  { value: 'h2h',    label: 'Moneyline (h2h)' },
  { value: 'total',  label: 'Totals' },
  { value: 'spread', label: 'Spread' },
]

// ── sub-components ─────────────────────────────────────────────────────────

function SortHeader({
  label, col, sortCol, sortDir, onSort,
}: {
  label: string; col: SortKey; sortCol: SortKey; sortDir: SortDir
  onSort: (k: SortKey) => void
}) {
  const active = sortCol === col
  return (
    <th
      className="px-3 py-2 text-left text-xs font-semibold text-muted uppercase tracking-wide cursor-pointer select-none whitespace-nowrap hover:text-gray-700"
      onClick={() => onSort(col)}
    >
      {label}
      {active && <span className="ml-1 opacity-60">{sortDir === 'asc' ? '↑' : '↓'}</span>}
    </th>
  )
}

// ── main component ─────────────────────────────────────────────────────────

export default function SlateTab() {
  const [games,    setGames]    = useState<DailyGame[]>([])
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState<string | null>(null)
  const [date,     setDate]     = useState('')
  const [detail,   setDetail]   = useState<{ game: DailyGame; totalLine: number } | null>(null)

  // filters
  const [mktType,  setMktType]  = useState('')
  const [minEdge,  setMinEdge]  = useState('')
  const [minEV,    setMinEV]    = useState('')
  const [minStake, setMinStake] = useState('')

  // sort
  const [sortCol,  setSortCol]  = useState<SortKey>('ev')
  const [sortDir,  setSortDir]  = useState<SortDir>('desc')

  const load = () => {
    setLoading(true); setError(null)
    fetchDailyPredictions(date || undefined, 10_000)
      .then(env => setGames(env.data))
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // flatten games × markets into rows
  const flat: FlatRow[] = useMemo(() => {
    const rows: FlatRow[] = []
    for (const g of games) {
      for (const m of g.markets) {
        rows.push({
          ...m,
          game_pk:   g.game_pk,
          matchup:   `${g.away_team} @ ${g.home_team}`,
          game_date: g.game_date,
        })
      }
    }
    return rows
  }, [games])

  // filter
  const filtered = useMemo(() => {
    const edgeMin  = parseFloat(minEdge)  / 100
    const evMin    = parseFloat(minEV)    / 100
    const stakeMin = parseFloat(minStake)
    return flat.filter(r => {
      if (mktType && !r.market.startsWith(mktType)) return false
      if (!isNaN(edgeMin)  && (r.edge  ?? -Infinity) < edgeMin)  return false
      if (!isNaN(evMin)    && (r.ev    ?? -Infinity) < evMin)    return false
      if (!isNaN(stakeMin) && (r.stake_units ?? -Infinity) < stakeMin) return false
      return true
    })
  }, [flat, mktType, minEdge, minEV, minStake])

  // sort
  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      let av: number | string | null = null
      let bv: number | string | null = null
      if (sortCol === 'game') { av = a.matchup; bv = b.matchup }
      else { av = a[sortCol] as number | null; bv = b[sortCol] as number | null }
      if (av === null || av === undefined) return 1
      if (bv === null || bv === undefined) return -1
      const cmp = av < bv ? -1 : av > bv ? 1 : 0
      return sortDir === 'asc' ? cmp : -cmp
    })
  }, [filtered, sortCol, sortDir])

  const toggleSort = (col: SortKey) => {
    if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortCol(col); setSortDir('desc') }
  }

  // open detail modal: find the game and derive total line from market keys
  const openDetail = (row: FlatRow) => {
    const game = games.find(g => g.game_pk === row.game_pk)
    if (!game) return
    const totalKey = Object.keys(game.simulation.totals)[0] ?? ''
    const lineNum = parseFloat(totalKey.replace(/[^0-9.]/g, '')) || 8.5
    setDetail({ game, totalLine: lineNum })
  }

  return (
    <div>
      {/* ── Controls ────────────────────────────────────────────────── */}
      <div className="flex flex-wrap gap-2 mb-4 items-end">
        <label className="flex flex-col gap-0.5 text-xs text-muted">
          Date
          <input
            type="date" value={date} onChange={e => setDate(e.target.value)}
            className="border border-border rounded px-2 py-1.5 text-sm text-gray-800 bg-white focus:outline-none focus:ring-1 focus:ring-brand"
          />
        </label>

        <label className="flex flex-col gap-0.5 text-xs text-muted">
          Market type
          <select
            value={mktType} onChange={e => setMktType(e.target.value)}
            className="border border-border rounded px-2 py-1.5 text-sm text-gray-800 bg-white focus:outline-none focus:ring-1 focus:ring-brand"
          >
            {MARKET_TYPE_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
        </label>

        {[
          { label: 'Min edge %', value: minEdge, set: setMinEdge, ph: 'e.g. 2' },
          { label: 'Min EV %',   value: minEV,   set: setMinEV,   ph: 'e.g. 1' },
          { label: 'Min stake $', value: minStake, set: setMinStake, ph: 'e.g. 10' },
        ].map(f => (
          <label key={f.label} className="flex flex-col gap-0.5 text-xs text-muted">
            {f.label}
            <input
              type="number" value={f.value} onChange={e => f.set(e.target.value)}
              placeholder={f.ph}
              className="border border-border rounded px-2 py-1.5 text-sm text-gray-800 bg-white w-24 focus:outline-none focus:ring-1 focus:ring-brand"
            />
          </label>
        ))}

        <button
          onClick={load}
          className="self-end bg-brand text-white text-sm font-semibold px-4 py-1.5 rounded hover:bg-blue-700 active:scale-95 transition-all"
        >
          {loading ? 'Loading…' : 'Refresh'}
        </button>
      </div>

      {/* ── Error ───────────────────────────────────────────────────── */}
      {error && (
        <div className="mb-3 bg-red-50 border border-red-200 text-red-700 text-sm px-3 py-2 rounded">
          ⚠ {error}
        </div>
      )}

      {/* ── Summary bar ─────────────────────────────────────────────── */}
      {!loading && sorted.length > 0 && (
        <p className="text-xs text-muted mb-2">
          {sorted.length} market{sorted.length !== 1 ? 's' : ''} across {games.length} game{games.length !== 1 ? 's' : ''}
          {(minEdge || minEV || mktType || minStake) ? ' (filtered)' : ''}
        </p>
      )}

      {/* ── Table ───────────────────────────────────────────────────── */}
      <div className="rounded-lg border border-border bg-white overflow-x-auto shadow-sm">
        <table className="w-full text-sm border-collapse">
          <thead className="bg-gray-50 border-b border-border">
            <tr>
              <SortHeader label="Game"         col="game"         sortCol={sortCol} sortDir={sortDir} onSort={toggleSort} />
              <SortHeader label="Market"       col="label"        sortCol={sortCol} sortDir={sortDir} onSort={toggleSort} />
              <SortHeader label="Model %"      col="model_prob"   sortCol={sortCol} sortDir={sortDir} onSort={toggleSort} />
              <SortHeader label="Market %"     col="market_prob"  sortCol={sortCol} sortDir={sortDir} onSort={toggleSort} />
              <SortHeader label="Odds"         col="market_odds"  sortCol={sortCol} sortDir={sortDir} onSort={toggleSort} />
              <SortHeader label="Edge"         col="edge"         sortCol={sortCol} sortDir={sortDir} onSort={toggleSort} />
              <SortHeader label="EV"           col="ev"           sortCol={sortCol} sortDir={sortDir} onSort={toggleSort} />
              <SortHeader label="Full Kelly"   col="kelly_full"   sortCol={sortCol} sortDir={sortDir} onSort={toggleSort} />
              <SortHeader label="¼K Stake"     col="stake_units"  sortCol={sortCol} sortDir={sortDir} onSort={toggleSort} />
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr><td colSpan={9} className="text-center py-10 text-muted text-sm">Loading…</td></tr>
            )}
            {!loading && sorted.length === 0 && (
              <tr><td colSpan={9} className="text-center py-10 text-muted text-sm">
                {games.length === 0 ? 'No games — click Refresh.' : 'No markets match the current filters.'}
              </td></tr>
            )}
            {sorted.map((r, i) => (
              <tr
                key={`${r.game_pk}-${r.market}`}
                className={[
                  'border-b border-gray-50 cursor-pointer',
                  i % 2 === 0 ? 'bg-white' : 'bg-gray-50/50',
                  'hover:bg-brand-50 transition-colors',
                ].join(' ')}
                onClick={() => openDetail(r)}
              >
                <td className="px-3 py-2 font-medium text-gray-800 whitespace-nowrap">{r.matchup}</td>
                <td className="px-3 py-2 text-muted whitespace-nowrap">{r.label}</td>
                <td className="px-3 py-2 tabular-nums">{pct(r.model_prob)}</td>
                <td className="px-3 py-2 tabular-nums">{pct(r.market_prob)}</td>
                <td className="px-3 py-2 tabular-nums">{odds(r.market_odds)}</td>
                <td className="px-3 py-2 tabular-nums">{signed(r.edge)}</td>
                <td className="px-3 py-2 tabular-nums">{signed(r.ev)}</td>
                <td className="px-3 py-2 tabular-nums">{pct(r.kelly_full, 2)}</td>
                <td className="px-3 py-2 tabular-nums font-semibold">
                  {r.stake_units !== null
                    ? <span className="text-brand">${r.stake_units.toFixed(0)}</span>
                    : <span className="text-gray-300">—</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* ── Game detail modal ────────────────────────────────────────── */}
      {detail && (
        <GameDetailModal
          game={detail.game}
          defaultTotalLine={detail.totalLine}
          onClose={() => setDetail(null)}
        />
      )}
    </div>
  )
}
