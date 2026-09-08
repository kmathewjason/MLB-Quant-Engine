/**
 * GameDetailModal — score-distribution histogram for a single game.
 *
 * Fetches /api/games/{game_pk}/simulation on open.
 * Renders home and away run distributions as overlapping bar charts
 * with the market total line marked as a dashed reference line.
 */
import { useState, useEffect } from 'react'
import {
  ComposedChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
  ReferenceLine, ResponsiveContainer, Legend,
} from 'recharts'
import { fetchGameSimulation } from '../api'
import type { DailyGame, GameSimData } from '../types'

interface Props {
  game: DailyGame
  defaultTotalLine: number
  onClose: () => void
}

// Build a merged histogram dataset aligned to shared bin centres.
// home and away have independent bin edges — we merge both into a
// unified x-axis by taking the union of all midpoints.
function mergeHistograms(
  home: { edges: number[]; counts: number[] },
  away: { edges: number[]; counts: number[] },
): { x: string; home: number; away: number }[] {
  const map = new Map<string, { home: number; away: number }>()

  const addSeries = (
    edges: number[],
    counts: number[],
    key: 'home' | 'away',
  ) => {
    counts.forEach((c, i) => {
      const mid = ((edges[i] + edges[i + 1]) / 2).toFixed(1)
      const entry = map.get(mid) ?? { home: 0, away: 0 }
      entry[key] += c
      map.set(mid, entry)
    })
  }

  addSeries(home.edges, home.counts, 'home')
  addSeries(away.edges, away.counts, 'away')

  return Array.from(map.entries())
    .map(([x, v]) => ({ x, ...v }))
    .sort((a, b) => parseFloat(a.x) - parseFloat(b.x))
}

export default function GameDetailModal({ game, defaultTotalLine, onClose }: Props) {
  const [sim,        setSim]      = useState<GameSimData | null>(null)
  const [loading,    setLoading]  = useState(true)
  const [error,      setError]    = useState<string | null>(null)
  const [totalLine,  setTotalLine] = useState(defaultTotalLine)
  const [nSims,      setNSims]    = useState(30_000)

  const load = (sims = nSims, line = totalLine) => {
    setLoading(true); setError(null); setSim(null)
    fetchGameSimulation(game.game_pk, { nSims: sims, totalLine: line, bins: 20 })
      .then(env => setSim(env.data))
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const merged = sim
    ? mergeHistograms(sim.home_runs.histogram, sim.away_runs.histogram)
    : []

  // total-runs histogram for the second chart
  const totalData = sim
    ? sim.total_runs.histogram.counts.map((c, i) => ({
        x: ((sim.total_runs.histogram.edges[i] + sim.total_runs.histogram.edges[i + 1]) / 2).toFixed(1),
        total: c,
      }))
    : []

  return (
    /* backdrop */
    <div
      className="fixed inset-0 z-50 bg-black/40 flex items-center justify-center p-4"
      onClick={onClose}
    >
      {/* panel */}
      <div
        className="bg-white rounded-xl shadow-2xl w-full max-w-3xl max-h-[90vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}
      >
        {/* title bar */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <div>
            <h2 className="font-bold text-gray-900 text-base">
              {game.away_team} @ {game.home_team}
            </h2>
            <p className="text-xs text-muted mt-0.5">
              {game.away_probable_pitcher} vs {game.home_probable_pitcher}
              &nbsp;·&nbsp; pk {game.game_pk}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-muted hover:text-gray-700 text-xl leading-none px-2"
          >
            ×
          </button>
        </div>

        {/* controls */}
        <div className="flex flex-wrap gap-3 items-end px-5 pt-4">
          <label className="flex flex-col gap-0.5 text-xs text-muted">
            Total line
            <input
              type="number" step="0.5" value={totalLine}
              onChange={e => setTotalLine(parseFloat(e.target.value) || 8.5)}
              className="border border-border rounded px-2 py-1.5 text-sm w-20 bg-white focus:outline-none focus:ring-1 focus:ring-brand"
            />
          </label>
          <label className="flex flex-col gap-0.5 text-xs text-muted">
            Simulations
            <select
              value={nSims} onChange={e => setNSims(parseInt(e.target.value))}
              className="border border-border rounded px-2 py-1.5 text-sm bg-white focus:outline-none focus:ring-1 focus:ring-brand"
            >
              {[10_000, 30_000, 50_000, 100_000].map(n => (
                <option key={n} value={n}>{n.toLocaleString()}</option>
              ))}
            </select>
          </label>
          <button
            onClick={() => load(nSims, totalLine)}
            className="self-end bg-brand text-white text-sm font-semibold px-4 py-1.5 rounded hover:bg-blue-700 active:scale-95 transition-all"
          >
            Re-run
          </button>
        </div>

        {/* body */}
        <div className="px-5 pt-4 pb-6">
          {error && (
            <div className="mb-4 bg-red-50 border border-red-200 text-red-700 text-sm px-3 py-2 rounded">
              ⚠ {error}
            </div>
          )}

          {loading && (
            <div className="flex items-center justify-center h-40 text-muted text-sm">
              Running {nSims.toLocaleString()} simulations…
            </div>
          )}

          {sim && !loading && (
            <>
              {/* win-prob strip */}
              <div className="flex gap-4 mb-5">
                {[
                  { label: game.home_team + ' win',   v: sim.home_win_prob, color: 'text-brand' },
                  { label: game.away_team + ' win',   v: sim.away_win_prob, color: 'text-gray-500' },
                  { label: `Over ${totalLine}`,        v: sim.over_prob,     color: 'text-emerald-600' },
                  { label: `Under ${totalLine}`,       v: sim.under_prob,    color: 'text-orange-500' },
                  { label: 'Spread cover (−1.5)',       v: sim.spread_cover_prob, color: 'text-violet-600' },
                ].map(item => (
                  <div key={item.label} className="text-center">
                    <div className={`text-xl font-bold ${item.color}`}>
                      {(item.v * 100).toFixed(1)}%
                    </div>
                    <div className="text-[11px] text-muted leading-tight">{item.label}</div>
                  </div>
                ))}
              </div>

              {/* Overlaid home/away run distributions */}
              <p className="text-xs font-semibold text-muted uppercase tracking-wide mb-2">
                Team run distributions (overlaid)
              </p>
              <ResponsiveContainer width="100%" height={200}>
                <ComposedChart data={merged} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
                  <CartesianGrid vertical={false} stroke="#f3f4f6" />
                  <XAxis dataKey="x" tick={{ fontSize: 10 }} tickLine={false} />
                  <YAxis tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={{ fontSize: 11 }}
                    formatter={(v: number, name: string) => [v.toLocaleString(), name === 'home' ? game.home_team : game.away_team]}
                  />
                  <Legend iconSize={10} wrapperStyle={{ fontSize: 11 }}
                    formatter={(val) => val === 'home' ? game.home_team : game.away_team}
                  />
                  <Bar dataKey="home" fill="#2563eb" fillOpacity={0.75} radius={[2, 2, 0, 0]} />
                  <Bar dataKey="away" fill="#9ca3af" fillOpacity={0.65} radius={[2, 2, 0, 0]} />
                </ComposedChart>
              </ResponsiveContainer>

              {/* Total runs with market line */}
              <p className="text-xs font-semibold text-muted uppercase tracking-wide mt-5 mb-2">
                Total runs — market line {totalLine}
              </p>
              <ResponsiveContainer width="100%" height={180}>
                <ComposedChart data={totalData} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
                  <CartesianGrid vertical={false} stroke="#f3f4f6" />
                  <XAxis dataKey="x" tick={{ fontSize: 10 }} tickLine={false} />
                  <YAxis tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={{ fontSize: 11 }}
                    formatter={(v: number) => [v.toLocaleString(), 'sims']}
                  />
                  <ReferenceLine
                    x={String(totalLine.toFixed(1))}
                    stroke="#ef4444"
                    strokeDasharray="4 3"
                    label={{ value: `${totalLine}`, position: 'top', fontSize: 10, fill: '#ef4444' }}
                  />
                  <Bar dataKey="total" fill="#059669" fillOpacity={0.75} radius={[2, 2, 0, 0]} />
                </ComposedChart>
              </ResponsiveContainer>

              {/* stats row */}
              <div className="mt-4 grid grid-cols-3 gap-3 text-center">
                {[
                  { label: 'Home μ runs',  v: sim.home_runs.mean.toFixed(2)  },
                  { label: 'Away μ runs',  v: sim.away_runs.mean.toFixed(2)  },
                  { label: 'Total μ runs', v: sim.total_runs.mean.toFixed(2) },
                  { label: 'Home p10–p90', v: `${sim.home_runs.p10.toFixed(0)}–${sim.home_runs.p90.toFixed(0)}` },
                  { label: 'Away p10–p90', v: `${sim.away_runs.p10.toFixed(0)}–${sim.away_runs.p90.toFixed(0)}` },
                  { label: 'Simulations',  v: sim.n_sims.toLocaleString()    },
                ].map(s => (
                  <div key={s.label} className="bg-gray-50 rounded-lg px-3 py-2">
                    <div className="text-sm font-bold text-gray-800">{s.v}</div>
                    <div className="text-[11px] text-muted">{s.label}</div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
