/**
 * SGPTab — Same-Game Parlay builder.
 *
 * Build up to 6 legs, call POST /api/predictions/sgp, then display:
 *  - Correlation-adjusted joint prob + Kelly stake
 *  - Naive (independent) joint prob + Kelly stake
 *  - % difference between the two stakes (highlighted)
 *  - Correlation matrix heat map
 *  - Per-leg marginal sim_probs
 */
import { useState } from 'react'
import { fetchSGP } from '../api'
import type { SGPLeg, SGPData, KellyVariant } from '../types'

// ── types ──────────────────────────────────────────────────────────────────

type Outcome = SGPLeg['outcome']
type Side    = SGPLeg['side']

interface LegDraft {
  side: Side
  outcome: Outcome
  line: string
  odds: string
  label: string
}

const SIDES:    Side[]    = ['home', 'away']
const OUTCOMES: Outcome[] = ['moneyline', 'over', 'under', 'spread']
const emptyLeg = (): LegDraft => ({ side: 'home', outcome: 'moneyline', line: '', odds: '-110', label: '' })

// ── small helpers ──────────────────────────────────────────────────────────

function pct(v: number, d = 2) { return `${(v * 100).toFixed(d)}%` }

function CorrCell({ v, isdiag }: { v: number; isdiag: boolean }) {
  if (isdiag) return <td className="px-3 py-1.5 text-center text-xs bg-gray-100 font-semibold">1.00</td>
  const abs = Math.abs(v)
  const bg  = v >= 0
    ? `rgba(37,99,235,${(abs * 0.45).toFixed(2)})`
    : `rgba(220,38,38,${(abs * 0.45).toFixed(2)})`
  return (
    <td className="px-3 py-1.5 text-center text-xs font-mono" style={{ background: bg }}>
      {v.toFixed(3)}
    </td>
  )
}

// ── Kelly comparison card ──────────────────────────────────────────────────

function KellyComparison({
  corrVariant, naiveVariant, bankroll,
}: {
  corrVariant: KellyVariant
  naiveVariant: KellyVariant
  bankroll: number
}) {
  const diff = corrVariant.stake_units - naiveVariant.stake_units
  const diffPct = naiveVariant.stake_units > 0
    ? (diff / naiveVariant.stake_units) * 100
    : null

  return (
    <div className="grid grid-cols-2 gap-4 mb-4">
      {/* Corr-adjusted */}
      <div className="rounded-lg border-2 border-brand bg-brand-50 p-4">
        <div className="text-xs font-semibold text-brand uppercase tracking-wide mb-2">
          Corr-adjusted (model)
        </div>
        <div className="text-3xl font-bold text-brand">
          ${corrVariant.stake_units.toFixed(0)}
        </div>
        <div className="text-xs text-muted mt-1">
          p = {pct(corrVariant.f_star)} full Kelly → ¼K
        </div>
        <div className={`mt-2 text-xs font-medium ${corrVariant.ev >= 0 ? 'text-pos' : 'text-neg'}`}>
          EV {corrVariant.ev >= 0 ? '+' : ''}{(corrVariant.ev * 100).toFixed(2)}%
        </div>
      </div>

      {/* Naive */}
      <div className="rounded-lg border border-border bg-white p-4">
        <div className="text-xs font-semibold text-muted uppercase tracking-wide mb-2">
          Naive (independent legs)
        </div>
        <div className="text-3xl font-bold text-gray-600">
          ${naiveVariant.stake_units.toFixed(0)}
        </div>
        <div className="text-xs text-muted mt-1">
          p = {pct(naiveVariant.f_star)} full Kelly → ¼K
        </div>
        <div className={`mt-2 text-xs font-medium ${naiveVariant.ev >= 0 ? 'text-pos' : 'text-neg'}`}>
          EV {naiveVariant.ev >= 0 ? '+' : ''}{(naiveVariant.ev * 100).toFixed(2)}%
        </div>
      </div>

      {/* Correlation value callout — spans full width */}
      {diffPct !== null && (
        <div className={[
          'col-span-2 rounded-lg px-4 py-3 flex items-center justify-between text-sm',
          Math.abs(diffPct) > 5
            ? 'bg-amber-50 border border-amber-200'
            : 'bg-gray-50 border border-border',
        ].join(' ')}>
          <span className="font-medium text-gray-700">
            Correlation model value
          </span>
          <span className={[
            'font-bold text-base',
            diff > 0 ? 'text-pos' : diff < 0 ? 'text-neg' : 'text-gray-500',
          ].join(' ')}>
            {diff >= 0 ? '+' : ''}${diff.toFixed(0)}&nbsp;
            ({diffPct >= 0 ? '+' : ''}{diffPct.toFixed(1)}% vs naive)
          </span>
          <span className="text-xs text-muted">
            {bankroll} unit bankroll
          </span>
        </div>
      )}
    </div>
  )
}

// ── main component ─────────────────────────────────────────────────────────

export default function SGPTab() {
  const [gameId,   setGameId]   = useState('')
  const [bankroll, setBankroll] = useState('1000')
  const [nSims,    setNSims]    = useState('20000')
  const [legs,     setLegs]     = useState<LegDraft[]>([emptyLeg(), emptyLeg()])
  const [result,   setResult]   = useState<SGPData | null>(null)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState<string | null>(null)

  const addLeg    = () => { if (legs.length < 6) setLegs(l => [...l, emptyLeg()]) }
  const removeLeg = (i: number) => { if (legs.length > 2) setLegs(l => l.filter((_, idx) => idx !== i)) }
  const updateLeg = (i: number, field: keyof LegDraft, value: string) =>
    setLegs(l => l.map((leg, idx) => idx === i ? { ...leg, [field]: value } : leg))

  const submit = () => {
    const gid = parseInt(gameId)
    if (!gid) { setError('Enter a game PK'); return }

    const parsed: SGPLeg[] = legs.map(l => {
      const leg: SGPLeg = { side: l.side, outcome: l.outcome, ...(l.label ? { label: l.label } : {}) }
      const line = parseFloat(l.line);  if (!isNaN(line)) leg.line = line
      const od   = parseFloat(l.odds);  if (!isNaN(od))   leg.odds = od
      return leg
    })

    setLoading(true); setError(null)
    fetchSGP(gid, parsed, parseFloat(bankroll) || 1000, parseInt(nSims) || 20_000)
      .then(env => setResult(env.data))
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }

  return (
    <div className="max-w-2xl">
      {/* ── Input panel ─────────────────────────────────────────────── */}
      <div className="bg-white rounded-lg border border-border shadow-sm p-4 mb-4">
        <h2 className="text-sm font-semibold text-gray-700 mb-3">Parlay parameters</h2>

        <div className="flex flex-wrap gap-3 mb-4 items-end">
          {[
            { label: 'Game PK',    v: gameId,   set: setGameId,   ph: 'e.g. 718976', w: 'w-28' },
            { label: 'Bankroll $', v: bankroll, set: setBankroll, ph: '1000',         w: 'w-24' },
            { label: '# Sims',     v: nSims,    set: setNSims,    ph: '20000',        w: 'w-24' },
          ].map(f => (
            <label key={f.label} className="flex flex-col gap-0.5 text-xs text-muted">
              {f.label}
              <input value={f.v} onChange={e => f.set(e.target.value)} placeholder={f.ph}
                className={`border border-border rounded px-2 py-1.5 text-sm bg-white ${f.w} focus:outline-none focus:ring-1 focus:ring-brand`}
              />
            </label>
          ))}
        </div>

        <h3 className="text-xs font-semibold text-muted uppercase tracking-wide mb-2">Legs</h3>
        <div className="space-y-2 mb-3">
          {legs.map((leg, i) => (
            <div key={i} className="flex flex-wrap gap-2 items-center bg-gray-50 rounded-lg px-3 py-2">
              <span className="text-xs font-mono text-muted w-5">#{i + 1}</span>

              <select value={leg.side} onChange={e => updateLeg(i, 'side', e.target.value)}
                className="border border-border rounded px-2 py-1 text-xs bg-white focus:outline-none focus:ring-1 focus:ring-brand">
                {SIDES.map(s => <option key={s} value={s}>{s}</option>)}
              </select>

              <select value={leg.outcome} onChange={e => updateLeg(i, 'outcome', e.target.value)}
                className="border border-border rounded px-2 py-1 text-xs bg-white focus:outline-none focus:ring-1 focus:ring-brand">
                {OUTCOMES.map(o => <option key={o} value={o}>{o}</option>)}
              </select>

              {['over', 'under', 'spread'].includes(leg.outcome) && (
                <input placeholder="Line" value={leg.line} onChange={e => updateLeg(i, 'line', e.target.value)}
                  className="border border-border rounded px-2 py-1 text-xs bg-white w-16 focus:outline-none focus:ring-1 focus:ring-brand" />
              )}

              <input placeholder="Odds" value={leg.odds} onChange={e => updateLeg(i, 'odds', e.target.value)}
                className="border border-border rounded px-2 py-1 text-xs bg-white w-16 focus:outline-none focus:ring-1 focus:ring-brand" />

              <input placeholder="Label" value={leg.label} onChange={e => updateLeg(i, 'label', e.target.value)}
                className="border border-border rounded px-2 py-1 text-xs bg-white w-24 focus:outline-none focus:ring-1 focus:ring-brand" />

              {legs.length > 2 && (
                <button onClick={() => removeLeg(i)}
                  className="text-red-400 hover:text-red-600 text-base leading-none ml-1">×</button>
              )}
            </div>
          ))}
        </div>

        <div className="flex gap-2">
          {legs.length < 6 && (
            <button onClick={addLeg}
              className="border border-border rounded px-3 py-1.5 text-xs text-muted hover:bg-gray-50 transition-colors">
              + Add leg
            </button>
          )}
          <button onClick={submit}
            className="bg-brand text-white text-sm font-semibold px-5 py-1.5 rounded hover:bg-blue-700 active:scale-95 transition-all">
            {loading ? 'Calculating…' : 'Calculate'}
          </button>
        </div>
      </div>

      {/* ── Error ───────────────────────────────────────────────────── */}
      {error && (
        <div className="mb-4 bg-red-50 border border-red-200 text-red-700 text-sm px-3 py-2 rounded">
          ⚠ {error}
        </div>
      )}

      {/* ── Results ─────────────────────────────────────────────────── */}
      {result && !loading && (
        <>
          {/* Joint probs summary */}
          <div className="bg-white rounded-lg border border-border shadow-sm px-4 py-3 mb-4 flex flex-wrap gap-5">
            {[
              { label: 'Corr-adjusted prob', v: pct(result.joint_prob_corr_adjusted), color: 'text-brand' },
              { label: 'Naive (independent)', v: pct(result.joint_prob_naive),        color: 'text-gray-500' },
              { label: 'Net payout (b)',      v: `${result.parlay_net_payout.toFixed(2)}×`, color: 'text-emerald-600' },
            ].map(item => (
              <div key={item.label}>
                <div className={`text-2xl font-bold ${item.color}`}>{item.v}</div>
                <div className="text-xs text-muted">{item.label}</div>
              </div>
            ))}
          </div>

          {/* Kelly comparison — the main value-add UI */}
          <KellyComparison
            corrVariant={result.kelly.corr_adjusted}
            naiveVariant={result.kelly.naive}
            bankroll={parseFloat(bankroll) || 1000}
          />

          {/* Per-leg marginals */}
          <div className="bg-white rounded-lg border border-border shadow-sm p-4 mb-4">
            <h3 className="text-xs font-semibold text-muted uppercase tracking-wide mb-3">Per-leg marginal probabilities</h3>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  {['Label', 'Side', 'Outcome', 'Line', 'Odds', 'Sim prob'].map(h => (
                    <th key={h} className="px-3 py-1.5 text-left text-xs text-muted font-semibold">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.legs.map((l, i) => (
                  <tr key={i} className="border-b border-gray-50">
                    <td className="px-3 py-1.5 font-medium">{l.label ?? `Leg ${i+1}`}</td>
                    <td className="px-3 py-1.5 text-muted">{l.side}</td>
                    <td className="px-3 py-1.5 text-muted">{l.outcome}</td>
                    <td className="px-3 py-1.5 tabular-nums">{l.line ?? '—'}</td>
                    <td className="px-3 py-1.5 tabular-nums">{l.odds ?? '—'}</td>
                    <td className="px-3 py-1.5 tabular-nums font-semibold text-brand">{pct(l.sim_prob)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Correlation matrix */}
          <div className="bg-white rounded-lg border border-border shadow-sm p-4">
            <h3 className="text-xs font-semibold text-muted uppercase tracking-wide mb-3">
              Correlation matrix
              <span className="ml-2 font-normal normal-case text-muted">blue = positive, red = negative</span>
            </h3>
            <div className="overflow-x-auto">
              <table className="border-collapse text-xs">
                <thead>
                  <tr>
                    <th className="px-3 py-1.5 text-left text-muted font-semibold" />
                    {result.legs.map((l, i) => (
                      <th key={i} className="px-3 py-1.5 text-muted font-semibold text-center">
                        {l.label ?? `L${i+1}`}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.correlation_matrix.map((row, ri) => (
                    <tr key={ri}>
                      <td className="px-3 py-1.5 text-muted font-semibold">
                        {result.legs[ri]?.label ?? `L${ri+1}`}
                      </td>
                      {row.map((v, ci) => (
                        <CorrCell key={ci} v={v} isdiag={ri === ci} />
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
