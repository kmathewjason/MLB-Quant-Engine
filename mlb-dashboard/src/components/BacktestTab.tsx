/**
 * BacktestTab — dark-theme rewrite.
 * Walk-forward summary · Calibration reliability diagram · CLV summary.
 */
import { useState, useEffect } from 'react'
import {
  ComposedChart, Line, Scatter, XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, Legend,
} from 'recharts'
import { fetchBacktestReport } from '../api'
import type { BacktestData, ReliabilityCurve } from '../types'

// ── design tokens ───────────────────────────────────────────────────────────
const C = {
  bg:        '#0f1117',
  surface:   '#16181f',
  card:      '#1c1f29',
  border:    '#22252e',
  border2:   '#2a2d38',
  text:      '#e8eaf0',
  muted:     '#8892a4',
  muted2:    '#5a6072',
  accent:    '#4faeff',
  pos:       '#34d399',
  neg:       '#f87171',
}

const CLASS_COLORS = [
  '#4faeff', '#34d399', '#fbbf24', '#a78bfa', '#f87171', '#22d3ee', '#86efac',
]

// ── helpers ─────────────────────────────────────────────────────────────────
function fmt4(v: number | string) { return Number(v).toFixed(4) }
function fmtPct(v: number | string, d = 1) { return `${(Number(v) * 100).toFixed(d)}%` }

// ── StatCard ────────────────────────────────────────────────────────────────
function StatCard({ label, value, sub, accent }: {
  label: string; value: string; sub?: string; accent?: boolean
}) {
  return (
    <div style={{
      background: C.card, border: `1px solid ${C.border}`,
      borderRadius: 10, padding: '14px 16px', textAlign: 'center',
    }}>
      <div style={{
        fontSize: 20, fontWeight: 700,
        color: accent ? C.accent : C.text,
        fontVariantNumeric: 'tabular-nums',
      }}>
        {value}
      </div>
      <div style={{ fontSize: 11, color: C.muted, marginTop: 3 }}>{label}</div>
      {sub && <div style={{ fontSize: 10, color: C.muted2, marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

// ── SectionHeader ────────────────────────────────────────────────────────────
function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      fontSize: 11, fontWeight: 700, color: C.muted,
      textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 12,
    }}>
      {children}
    </div>
  )
}

// ── ReliabilityDiagram ───────────────────────────────────────────────────────
function ReliabilityDiagram({ curves }: { curves: ReliabilityCurve[] }) {
  const [selected, setSelected] = useState<string | null>(null)

  const active = selected ? curves.filter(c => c.class_name === selected) : curves
  const diagonal = [{ x: 0, y: 0 }, { x: 1, y: 1 }]

  return (
    <div>
      {/* Class selector pills */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 14 }}>
        <button
          onClick={() => setSelected(null)}
          style={{
            padding: '3px 11px', borderRadius: 99, fontSize: 11, fontWeight: 600,
            border: 'none', cursor: 'pointer',
            background: selected === null ? C.accent : C.border2,
            color: selected === null ? '#fff' : C.muted,
            transition: 'all 0.12s',
          }}
        >
          All
        </button>
        {curves.map((c, i) => {
          const col = CLASS_COLORS[i % CLASS_COLORS.length]
          const isActive = selected === c.class_name
          return (
            <button
              key={c.class_name}
              onClick={() => setSelected(s => s === c.class_name ? null : c.class_name)}
              style={{
                padding: '3px 11px', borderRadius: 99, fontSize: 11, fontWeight: 600,
                border: 'none', cursor: 'pointer',
                background: isActive ? col : C.border2,
                color: isActive ? '#fff' : C.muted,
                transition: 'all 0.12s',
              }}
            >
              {c.class_name}&nbsp;(ECE {(c.ece * 100).toFixed(1)}%)
            </button>
          )
        })}
      </div>

      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart margin={{ top: 8, right: 8, bottom: 20, left: -4 }}>
          <CartesianGrid stroke={C.border} strokeDasharray="3 3" />
          <XAxis
            type="number" dataKey="x" domain={[0, 1]}
            label={{ value: 'Mean predicted prob', position: 'insideBottom', offset: -10, fontSize: 10, fill: C.muted }}
            tick={{ fontSize: 10, fill: C.muted }} tickLine={false}
            axisLine={{ stroke: C.border }}
          />
          <YAxis
            type="number" dataKey="y" domain={[0, 1]}
            label={{ value: 'Observed freq', angle: -90, position: 'insideLeft', offset: 14, fontSize: 10, fill: C.muted }}
            tick={{ fontSize: 10, fill: C.muted }} tickLine={false} axisLine={false}
          />
          <Tooltip
            contentStyle={{
              background: C.card, border: `1px solid ${C.border}`,
              borderRadius: 8, fontSize: 11, color: C.text,
            }}
            formatter={(v: number, _name: string, props: { payload?: { class_name?: string } }) => [
              `${(v * 100).toFixed(1)}%`,
              props.payload?.class_name ?? 'obs',
            ]}
          />
          <Legend iconSize={8} wrapperStyle={{ fontSize: 11, color: C.muted }} />

          {/* Perfect calibration diagonal */}
          <Line
            data={diagonal} dataKey="y" dot={false}
            stroke={C.muted2} strokeDasharray="5 3" strokeWidth={1.5}
            name="Perfect cal." legendType="plainline"
          />

          {/* One Scatter per class */}
          {active.map((c, i) => {
            const color = CLASS_COLORS[curves.findIndex(cc => cc.class_name === c.class_name) % CLASS_COLORS.length]
            const pts = c.bins.map(b => ({
              x: b.mean_pred, y: b.obs_freq,
              class_name: c.class_name, count: b.count,
              ci_lo: b.ci_lo_95, ci_hi: b.ci_hi_95,
            }))
            return (
              <Scatter
                key={c.class_name}
                data={pts} fill={color} name={c.class_name}
                shape="circle" r={i === 0 ? 5 : 4}
              />
            )
          })}
        </ComposedChart>
      </ResponsiveContainer>
      <p style={{ fontSize: 10, color: C.muted2, textAlign: 'center', marginTop: 4 }}>
        Each point = one calibration bin · dashed diagonal = perfect calibration
      </p>
    </div>
  )
}

// ── section card wrapper ─────────────────────────────────────────────────────
function Card({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <section style={{
      background: C.surface, border: `1px solid ${C.border}`,
      borderRadius: 12, padding: '18px 20px', ...style,
    }}>
      {children}
    </section>
  )
}

// ── main component ───────────────────────────────────────────────────────────
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
      {/* ── toolbar ──────────────────────────────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 18 }}>
        <button
          onClick={load}
          style={{
            background: C.accent, color: '#fff', border: 'none', borderRadius: 8,
            padding: '8px 18px', fontWeight: 700, fontSize: 13, cursor: 'pointer',
          }}
        >
          {loading ? 'Loading…' : 'Refresh'}
        </button>
        {data && (
          <span style={{ fontSize: 11, color: C.muted }}>
            Updated {new Date().toLocaleTimeString()}
          </span>
        )}
      </div>

      {/* ── error ────────────────────────────────────────────────────── */}
      {error && (
        <div style={{
          background: '#2a1515', border: '1px solid #6b2020', borderRadius: 8,
          color: C.neg, fontSize: 13, padding: '10px 14px', marginBottom: 16,
        }}>
          ⚠ {error}
        </div>
      )}

      {/* ── loading skeleton ─────────────────────────────────────────── */}
      {loading && (
        <div style={{ textAlign: 'center', padding: '48px 0', color: C.muted, fontSize: 13 }}>
          Loading backtest data…
        </div>
      )}

      {/* ── content ──────────────────────────────────────────────────── */}
      {data && !loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

          {/* Walk-forward summary */}
          {data.walk_forward.available && data.walk_forward.summary && (
            <Card>
              <SectionHeader>Walk-forward summary</SectionHeader>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))',
                gap: 10,
              }}>
                <StatCard label="Folds"
                  value={String(data.walk_forward.summary.n_folds)} />
                <StatCard label="OOS predictions"
                  value={data.walk_forward.summary.n_oos_predictions.toLocaleString()} />
                <StatCard label="OOS log-loss"
                  value={fmt4(data.walk_forward.summary.oos_log_loss)} />
                <StatCard label="OOS accuracy"
                  value={fmtPct(data.walk_forward.summary.oos_accuracy)} accent />
                <StatCard label="Mean Brier"
                  value={fmt4(data.walk_forward.summary.mean_fold_brier)} />
              </div>
            </Card>
          )}

          {/* Calibration */}
          {data.calibration.available && (
            <Card>
              <SectionHeader>Calibration</SectionHeader>

              {/* Brier decomposition stat strip */}
              {data.calibration.brier_decomposition && (
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))',
                  gap: 10, marginBottom: 20,
                }}>
                  {(Object.entries(data.calibration.brier_decomposition) as [string, number][]).map(([k, v]) => (
                    <StatCard
                      key={k}
                      label={k.replace(/_/g, ' ')}
                      value={fmt4(v)}
                    />
                  ))}
                </div>
              )}

              {/* Reliability diagram */}
              {data.calibration.reliability_curves && data.calibration.reliability_curves.length > 0 ? (
                <>
                  <div style={{
                    fontSize: 11, fontWeight: 600, color: C.muted,
                    textTransform: 'uppercase', letterSpacing: '0.08em',
                    marginBottom: 10,
                  }}>
                    Reliability diagram
                    <span style={{ fontWeight: 400, marginLeft: 8, textTransform: 'none', letterSpacing: 0 }}>
                      — blue = positive corr, red = negative
                    </span>
                  </div>
                  <ReliabilityDiagram curves={data.calibration.reliability_curves} />
                </>
              ) : (
                <p style={{ fontSize: 12, color: C.muted, margin: 0 }}>
                  Reliability curves not available (need wf_predictions.parquet in data/predictions/).
                </p>
              )}

              {/* Per-class calibration table */}
              {data.calibration.report.length > 0 && (
                <div style={{ marginTop: 20, overflowX: 'auto' }}>
                  <div style={{
                    fontSize: 11, fontWeight: 600, color: C.muted,
                    textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8,
                  }}>
                    Per-class calibration
                  </div>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                    <thead>
                      <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                        {['Class', 'ECE', 'Brier', 'Reliability', 'Resolution', 'BSS', 'Base rate', 'N+'].map(h => (
                          <th key={h} style={{
                            padding: '6px 12px', textAlign: 'left',
                            color: C.muted, fontWeight: 600, whiteSpace: 'nowrap',
                          }}>
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {data.calibration.report.map((r, i) => (
                        <tr key={i} style={{
                          borderBottom: `1px solid ${C.border}`,
                          background: i % 2 === 0 ? 'transparent' : '#ffffff06',
                        }}>
                          <td style={{ padding: '7px 12px', fontWeight: 600, color: CLASS_COLORS[i % CLASS_COLORS.length] }}>
                            {r.class_name}
                          </td>
                          <td style={{ padding: '7px 12px', fontVariantNumeric: 'tabular-nums', color: C.text }}>
                            {fmt4(r.ece)}
                          </td>
                          <td style={{ padding: '7px 12px', fontVariantNumeric: 'tabular-nums', color: C.text }}>
                            {fmt4(r.brier_score)}
                          </td>
                          <td style={{ padding: '7px 12px', fontVariantNumeric: 'tabular-nums', color: C.text }}>
                            {fmt4(r.reliability)}
                          </td>
                          <td style={{ padding: '7px 12px', fontVariantNumeric: 'tabular-nums', color: C.text }}>
                            {fmt4(r.resolution)}
                          </td>
                          <td style={{ padding: '7px 12px', fontVariantNumeric: 'tabular-nums', color: C.text }}>
                            {Number(r.brier_skill_score).toFixed(3)}
                          </td>
                          <td style={{ padding: '7px 12px', fontVariantNumeric: 'tabular-nums', color: C.text }}>
                            {fmtPct(r.base_rate)}
                          </td>
                          <td style={{ padding: '7px 12px', fontVariantNumeric: 'tabular-nums', color: C.muted }}>
                            {r.n_positive}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          )}

          {/* CLV summary */}
          {data.clv.available && data.clv.summary && (
            <Card>
              <SectionHeader>Closing-line value</SectionHeader>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))',
                gap: 10,
              }}>
                <StatCard label="Total bets"
                  value={data.clv.summary.n_bets.toLocaleString()} />
                <StatCard label="% Positive CLV"
                  value={`${(data.clv.summary.pct_positive_clv * 100).toFixed(0)}%`}
                  sub="above 50% = skill" accent />
                <StatCard label="Mean CLV (log-odds)"
                  value={data.clv.summary.mean_clv_log_odds.toFixed(3)}
                  sub=">0 = buying below close" />
                <StatCard label="t-stat / p-value"
                  value={`${data.clv.summary.clv_tstat.toFixed(2)} / ${data.clv.summary.clv_pvalue.toFixed(3)}`}
                  sub="H₀: CLV = 0" />
                <StatCard label="Result correlation"
                  value={data.clv.summary.clv_result_corr.toFixed(3)}
                  sub="CLV predicts outcome" />
                <StatCard label="ROI"
                  value={`${data.clv.summary.roi_pct.toFixed(2)}%`}
                  accent={data.clv.summary.roi_pct > 0} />
                <StatCard label="Total P&L (units)"
                  value={data.clv.summary.total_pnl_units.toFixed(2)} />
              </div>
            </Card>
          )}

          {/* nothing at all */}
          {!data.walk_forward.available && !data.calibration.available && !data.clv.available && (
            <div style={{ textAlign: 'center', padding: '48px 0', color: C.muted, fontSize: 13 }}>
              No backtest data found. Run the pipeline first:&nbsp;
              <code style={{
                background: C.card, border: `1px solid ${C.border}`,
                borderRadius: 5, padding: '2px 8px', fontFamily: 'monospace',
                fontSize: 12, color: C.accent,
              }}>
                .venv/bin/python main.py --backtest
              </code>
            </div>
          )}

        </div>
      )}
    </div>
  )
}
