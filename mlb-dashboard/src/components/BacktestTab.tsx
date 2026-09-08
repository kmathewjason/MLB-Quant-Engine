/**
 * BacktestTab — calibration reliability diagram + CLV summary.
 *
 * Fetches GET /api/backtest/report.
 * Reliability diagram: mean_pred on x-axis, obs_freq on y-axis, per PA-outcome class.
 * Diagonal = perfect calibration reference line.
 */
import { useState, useEffect } from 'react'
import {
  ComposedChart, Line, Scatter, XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, Legend,
} from 'recharts'
import { fetchBacktestReport } from '../api'
import type { BacktestData, ReliabilityCurve } from '../types'

// ── palette for multi-class reliability curves ─────────────────────────────
const CLASS_COLORS = [
  '#2563eb', '#059669', '#d97706', '#7c3aed', '#dc2626', '#0891b2', '#65a30d',
]

// ── sub-components ─────────────────────────────────────────────────────────

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-gray-50 rounded-lg px-4 py-3 text-center">
      <div className="text-xl font-bold text-gray-800">{value}</div>
      <div className="text-xs text-muted mt-0.5">{label}</div>
      {sub && <div className="text-[10px] text-gray-400 mt-0.5">{sub}</div>}
    </div>
  )
}

/** Reliability diagram for one or more PA-outcome classes. */
function ReliabilityDiagram({ curves }: { curves: ReliabilityCurve[] }) {
  const [selected, setSelected] = useState<string | null>(null)

  const active = selected
    ? curves.filter(c => c.class_name === selected)
    : curves

  // Build scatter data per curve: { mean_pred, obs_freq, class_name }
  // Recharts Scatter needs a flat array per series, so we render one Scatter per class.
  const diagonal = [{ x: 0, y: 0 }, { x: 1, y: 1 }]

  return (
    <div>
      {/* class selector pills */}
      <div className="flex flex-wrap gap-1.5 mb-3">
        <button
          onClick={() => setSelected(null)}
          className={[
            'px-2.5 py-0.5 rounded-full text-xs font-medium transition-colors',
            selected === null ? 'bg-brand text-white' : 'bg-gray-100 text-muted hover:bg-gray-200',
          ].join(' ')}
        >
          All
        </button>
        {curves.map((c, i) => (
          <button
            key={c.class_name}
            onClick={() => setSelected(s => s === c.class_name ? null : c.class_name)}
            className={[
              'px-2.5 py-0.5 rounded-full text-xs font-medium transition-colors',
              selected === c.class_name
                ? 'text-white'
                : 'bg-gray-100 text-muted hover:bg-gray-200',
            ].join(' ')}
            style={selected === c.class_name ? { background: CLASS_COLORS[i % CLASS_COLORS.length] } : {}}
          >
            {c.class_name} (ECE {(c.ece * 100).toFixed(1)}%)
          </button>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart margin={{ top: 8, right: 8, bottom: 16, left: -4 }}>
          <CartesianGrid stroke="#f3f4f6" />
          <XAxis
            type="number" dataKey="x" domain={[0, 1]}
            label={{ value: 'Mean predicted prob', position: 'insideBottom', offset: -8, fontSize: 11 }}
            tick={{ fontSize: 10 }} tickLine={false}
          />
          <YAxis
            type="number" dataKey="y" domain={[0, 1]}
            label={{ value: 'Observed freq', angle: -90, position: 'insideLeft', offset: 14, fontSize: 11 }}
            tick={{ fontSize: 10 }} tickLine={false} axisLine={false}
          />
          <Tooltip
            contentStyle={{ fontSize: 11 }}
            formatter={(v: number, _name: string, props: { payload?: { class_name?: string } }) => [
              `${(v * 100).toFixed(1)}%`,
              props.payload?.class_name ?? 'obs',
            ]}
          />
          <Legend iconSize={8} wrapperStyle={{ fontSize: 11 }} />

          {/* Perfect calibration line */}
          <Line
            data={diagonal} dataKey="y" dot={false}
            stroke="#d1d5db" strokeDasharray="4 3" strokeWidth={1.5}
            name="Perfect calibration" legendType="plainline"
          />

          {/* One Scatter per class */}
          {active.map((c, i) => {
            const color = CLASS_COLORS[curves.findIndex(cc => cc.class_name === c.class_name) % CLASS_COLORS.length]
            const pts = c.bins.map(b => ({
              x:          b.mean_pred,
              y:          b.obs_freq,
              class_name: c.class_name,
              count:      b.count,
              ci_lo:      b.ci_lo_95,
              ci_hi:      b.ci_hi_95,
            }))
            return (
              <Scatter
                key={c.class_name}
                data={pts}
                fill={color}
                name={c.class_name}
                shape="circle"
                r={i === 0 ? 4 : 3}
              />
            )
          })}
        </ComposedChart>
      </ResponsiveContainer>
      <p className="text-[10px] text-muted text-center mt-1">
        Each point = one calibration bin. Dashed diagonal = perfect calibration.
      </p>
    </div>
  )
}

// ── main component ─────────────────────────────────────────────────────────

export default function BacktestTab() {
  const [data,    setData]    = useState<BacktestData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState<string | null>(null)

  const load = () => {
    setLoading(true); setError(null)
    fetchBacktestReport()
      .then(env => setData(env.data))
      .catch(e => {
        const msg = String(e)
        setError(msg.startsWith('Error: 404')
          ? 'No backtest data found. Run: .venv/bin/python main.py --backtest'
          : msg)
      })
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div>
      <div className="flex items-center gap-3 mb-4">
        <button
          onClick={load}
          className="bg-brand text-white text-sm font-semibold px-4 py-1.5 rounded hover:bg-blue-700 active:scale-95 transition-all"
        >
          {loading ? 'Loading…' : 'Refresh'}
        </button>
        {data && (
          <span className="text-xs text-muted">
            Last updated: {new Date().toLocaleTimeString()}
          </span>
        )}
      </div>

      {error && (
        <div className="mb-4 bg-red-50 border border-red-200 text-red-700 text-sm px-3 py-2 rounded">
          ⚠ {error}
        </div>
      )}

      {loading && (
        <div className="text-center py-10 text-muted text-sm">Loading backtest data…</div>
      )}

      {data && !loading && (
        <div className="space-y-4">

          {/* ── Walk-forward summary ─────────────────────────────────── */}
          {data.walk_forward.available && data.walk_forward.summary && (
            <section className="bg-white rounded-lg border border-border shadow-sm p-4">
              <h2 className="text-sm font-semibold text-gray-700 mb-3">Walk-forward summary</h2>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
                <StatCard label="Folds"         value={String(data.walk_forward.summary.n_folds)} />
                <StatCard label="OOS predictions" value={data.walk_forward.summary.n_oos_predictions.toLocaleString()} />
                <StatCard label="OOS log-loss"  value={data.walk_forward.summary.oos_log_loss.toFixed(4)} />
                <StatCard label="OOS accuracy"  value={`${(data.walk_forward.summary.oos_accuracy * 100).toFixed(1)}%`} />
                <StatCard label="Mean Brier"    value={data.walk_forward.summary.mean_fold_brier.toFixed(4)} />
              </div>
            </section>
          )}

          {/* ── Calibration reliability diagram ─────────────────────── */}
          {data.calibration.available && (
            <section className="bg-white rounded-lg border border-border shadow-sm p-4">
              <h2 className="text-sm font-semibold text-gray-700 mb-1">Calibration</h2>

              {/* Brier decomposition */}
              {data.calibration.brier_decomposition && (
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-4">
                  {(Object.entries(data.calibration.brier_decomposition) as [string, number][]).map(([k, v]) => (
                    <StatCard
                      key={k}
                      label={k.replace(/_/g, ' ')}
                      value={v.toFixed(4)}
                    />
                  ))}
                </div>
              )}

              {/* Reliability diagram */}
              {data.calibration.reliability_curves && data.calibration.reliability_curves.length > 0 ? (
                <>
                  <h3 className="text-xs font-semibold text-muted uppercase tracking-wide mb-2">
                    Reliability diagram
                  </h3>
                  <ReliabilityDiagram curves={data.calibration.reliability_curves} />
                </>
              ) : (
                <p className="text-xs text-muted">
                  Reliability curves not available (need wf_predictions.parquet in data/predictions/).
                </p>
              )}

              {/* Per-class calibration table */}
              {data.calibration.report.length > 0 && (
                <div className="mt-4 overflow-x-auto">
                  <h3 className="text-xs font-semibold text-muted uppercase tracking-wide mb-2">
                    Per-class calibration
                  </h3>
                  <table className="w-full text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-border">
                        {['Class', 'ECE', 'Brier', 'Reliability', 'Resolution', 'BSS', 'Base rate', 'N+'].map(h => (
                          <th key={h} className="px-3 py-1.5 text-left text-muted font-semibold whitespace-nowrap">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {data.calibration.report.map((r, i) => (
                        <tr key={i} className="border-b border-gray-50">
                          <td className="px-3 py-1.5 font-semibold">{r.class_name}</td>
                          <td className="px-3 py-1.5 tabular-nums">{Number(r.ece).toFixed(4)}</td>
                          <td className="px-3 py-1.5 tabular-nums">{Number(r.brier_score).toFixed(4)}</td>
                          <td className="px-3 py-1.5 tabular-nums">{Number(r.reliability).toFixed(4)}</td>
                          <td className="px-3 py-1.5 tabular-nums">{Number(r.resolution).toFixed(4)}</td>
                          <td className="px-3 py-1.5 tabular-nums">{Number(r.brier_skill_score).toFixed(3)}</td>
                          <td className="px-3 py-1.5 tabular-nums">{(Number(r.base_rate) * 100).toFixed(1)}%</td>
                          <td className="px-3 py-1.5 tabular-nums">{r.n_positive}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          )}

          {/* ── CLV summary ──────────────────────────────────────────── */}
          {data.clv.available && data.clv.summary && (
            <section className="bg-white rounded-lg border border-border shadow-sm p-4">
              <h2 className="text-sm font-semibold text-gray-700 mb-3">Closing-line value</h2>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
                <StatCard label="Total bets"      value={data.clv.summary.n_bets.toLocaleString()} />
                <StatCard
                  label="% Positive CLV"
                  value={`${(data.clv.summary.pct_positive_clv * 100).toFixed(0)}%`}
                  sub="above 50% = skill"
                />
                <StatCard
                  label="Mean CLV (log-odds)"
                  value={data.clv.summary.mean_clv_log_odds.toFixed(3)}
                  sub=">0 = buying below close"
                />
                <StatCard
                  label="t-stat / p-value"
                  value={`${data.clv.summary.clv_tstat.toFixed(2)} / ${data.clv.summary.clv_pvalue.toFixed(3)}`}
                  sub="H0: CLV = 0"
                />
                <StatCard
                  label="Result correlation"
                  value={data.clv.summary.clv_result_corr.toFixed(3)}
                  sub="CLV predicts outcome"
                />
                <StatCard
                  label="ROI %"
                  value={`${data.clv.summary.roi_pct.toFixed(2)}%`}
                />
                <StatCard
                  label="Total P&L (units)"
                  value={data.clv.summary.total_pnl_units.toFixed(2)}
                />
              </div>
            </section>
          )}

          {/* no data at all */}
          {!data.walk_forward.available && !data.calibration.available && !data.clv.available && (
            <p className="text-sm text-muted py-6 text-center">
              No backtest data found. Run the pipeline first:
              <code className="ml-1 bg-gray-100 px-1.5 py-0.5 rounded font-mono text-xs">
                .venv/bin/python main.py --backtest
              </code>
            </p>
          )}
        </div>
      )}
    </div>
  )
}
