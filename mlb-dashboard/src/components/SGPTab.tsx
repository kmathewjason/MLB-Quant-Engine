/**
 * SGPTab — dark-theme rewrite.
 * Same-Game Parlay builder: up to 6 legs, correlation-adjusted Kelly vs naive,
 * correlation matrix heat map, per-leg marginal sim_probs.
 */
import { useState } from 'react'
import { fetchSGP } from '../api'
import type { SGPLeg, SGPData, KellyVariant } from '../types'

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

// ── types ────────────────────────────────────────────────────────────────────
type Outcome = SGPLeg['outcome']
type Side    = SGPLeg['side']

interface LegDraft {
  side:    Side
  outcome: Outcome
  line:    string
  odds:    string
  label:   string
}

const SIDES:    Side[]    = ['home', 'away']
const OUTCOMES: Outcome[] = ['moneyline', 'over', 'under', 'spread']
const emptyLeg = (): LegDraft => ({ side: 'home', outcome: 'moneyline', line: '', odds: '-110', label: '' })

// ── helpers ──────────────────────────────────────────────────────────────────
function pct(v: number, d = 2) { return `${(v * 100).toFixed(d)}%` }

// ── SectionHeader ─────────────────────────────────────────────────────────────
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

// ── small input ───────────────────────────────────────────────────────────────
function Field({ label, value, onChange, placeholder, width }: {
  label: string; value: string; onChange: (v: string) => void;
  placeholder?: string; width?: number | string
}) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <span style={{ fontSize: 10, fontWeight: 600, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.07em' }}>
        {label}
      </span>
      <input
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        style={{
          width, background: C.card, border: `1px solid ${C.border2}`,
          borderRadius: 6, padding: '6px 10px', fontSize: 13, color: C.text,
          outline: 'none',
        }}
      />
    </label>
  )
}

// ── select ────────────────────────────────────────────────────────────────────
function Select({ value, onChange, options }: {
  value: string; onChange: (v: string) => void; options: string[]
}) {
  return (
    <select
      value={value}
      onChange={e => onChange(e.target.value)}
      style={{
        background: C.card, border: `1px solid ${C.border2}`, borderRadius: 6,
        padding: '5px 8px', fontSize: 12, color: C.text, cursor: 'pointer',
      }}
    >
      {options.map(o => <option key={o} value={o}>{o}</option>)}
    </select>
  )
}

// ── Kelly comparison card ─────────────────────────────────────────────────────
function KellyComparison({
  corrVariant, naiveVariant, bankroll,
}: {
  corrVariant: KellyVariant
  naiveVariant: KellyVariant
  bankroll: number
}) {
  const diff    = corrVariant.stake_units - naiveVariant.stake_units
  const diffPct = naiveVariant.stake_units > 0
    ? (diff / naiveVariant.stake_units) * 100
    : null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 20 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>

        {/* Corr-adjusted — primary */}
        <div style={{
          background: C.card,
          border: `2px solid ${C.accent}`,
          borderRadius: 12, padding: '16px 18px',
        }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: C.accent, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>
            Corr-adjusted (model)
          </div>
          <div style={{ fontSize: 28, fontWeight: 800, color: C.accent, fontVariantNumeric: 'tabular-nums' }}>
            ${corrVariant.stake_units.toFixed(0)}
          </div>
          <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>
            p = {pct(corrVariant.f_star)} full Kelly → ¼K
          </div>
          <div style={{ marginTop: 6, fontSize: 12, fontWeight: 600, color: corrVariant.ev >= 0 ? C.pos : C.neg }}>
            EV {corrVariant.ev >= 0 ? '+' : ''}{(corrVariant.ev * 100).toFixed(2)}%
          </div>
        </div>

        {/* Naive — secondary */}
        <div style={{
          background: C.card, border: `1px solid ${C.border}`,
          borderRadius: 12, padding: '16px 18px',
        }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>
            Naive (independent)
          </div>
          <div style={{ fontSize: 28, fontWeight: 800, color: C.muted, fontVariantNumeric: 'tabular-nums' }}>
            ${naiveVariant.stake_units.toFixed(0)}
          </div>
          <div style={{ fontSize: 11, color: C.muted2, marginTop: 4 }}>
            p = {pct(naiveVariant.f_star)} full Kelly → ¼K
          </div>
          <div style={{ marginTop: 6, fontSize: 12, fontWeight: 600, color: naiveVariant.ev >= 0 ? C.pos : C.neg }}>
            EV {naiveVariant.ev >= 0 ? '+' : ''}{(naiveVariant.ev * 100).toFixed(2)}%
          </div>
        </div>
      </div>

      {/* Correlation value callout */}
      {diffPct !== null && (
        <div style={{
          background: Math.abs(diffPct) > 5 ? '#26200e' : C.card,
          border: `1px solid ${Math.abs(diffPct) > 5 ? C.amber : C.border}`,
          borderRadius: 10, padding: '10px 16px',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: C.text }}>
            Correlation model value
          </span>
          <span style={{ fontSize: 15, fontWeight: 800, color: diff > 0 ? C.pos : diff < 0 ? C.neg : C.muted }}>
            {diff >= 0 ? '+' : ''}${diff.toFixed(0)}&nbsp;
            ({diffPct >= 0 ? '+' : ''}{diffPct.toFixed(1)}% vs naive)
          </span>
          <span style={{ fontSize: 11, color: C.muted2 }}>{bankroll} unit bankroll</span>
        </div>
      )}
    </div>
  )
}

// ── correlation cell ──────────────────────────────────────────────────────────
function CorrCell({ v, isdiag }: { v: number; isdiag: boolean }) {
  if (isdiag) {
    return (
      <td style={{ padding: '6px 12px', textAlign: 'center', fontSize: 11, fontWeight: 700, color: C.muted }}>
        1.00
      </td>
    )
  }
  const abs = Math.abs(v)
  const bg  = v >= 0
    ? `rgba(79,174,255,${(abs * 0.5).toFixed(2)})`
    : `rgba(248,113,113,${(abs * 0.5).toFixed(2)})`
  return (
    <td style={{
      padding: '6px 12px', textAlign: 'center', fontSize: 11,
      fontFamily: 'monospace', background: bg, color: C.text,
    }}>
      {v.toFixed(3)}
    </td>
  )
}

// ── main component ────────────────────────────────────────────────────────────
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
    <div style={{ maxWidth: 680 }}>

      {/* ── Input panel ──────────────────────────────────────────────── */}
      <div style={{
        background: C.surface, border: `1px solid ${C.border}`,
        borderRadius: 12, padding: '18px 20px', marginBottom: 16,
      }}>
        <SectionHeader>Parlay parameters</SectionHeader>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, marginBottom: 18, alignItems: 'flex-end' }}>
          <Field label="Game PK"    value={gameId}   onChange={setGameId}   placeholder="e.g. 718976" width={110} />
          <Field label="Bankroll $" value={bankroll} onChange={setBankroll} placeholder="1000"        width={90}  />
          <Field label="# Sims"     value={nSims}    onChange={setNSims}    placeholder="20000"       width={90}  />
        </div>

        <SectionHeader>Legs</SectionHeader>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 14 }}>
          {legs.map((leg, i) => (
            <div key={i} style={{
              display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center',
              background: C.card, border: `1px solid ${C.border}`,
              borderRadius: 8, padding: '8px 12px',
            }}>
              <span style={{ fontSize: 11, fontFamily: 'monospace', color: C.muted2, width: 20 }}>#{i + 1}</span>

              <Select value={leg.side}    onChange={v => updateLeg(i, 'side', v)}    options={SIDES} />
              <Select value={leg.outcome} onChange={v => updateLeg(i, 'outcome', v)} options={OUTCOMES} />

              {['over', 'under', 'spread'].includes(leg.outcome) && (
                <input
                  placeholder="Line" value={leg.line}
                  onChange={e => updateLeg(i, 'line', e.target.value)}
                  style={{
                    width: 60, background: C.bg, border: `1px solid ${C.border2}`,
                    borderRadius: 6, padding: '5px 8px', fontSize: 12, color: C.text,
                  }}
                />
              )}

              <input
                placeholder="Odds" value={leg.odds}
                onChange={e => updateLeg(i, 'odds', e.target.value)}
                style={{
                  width: 60, background: C.bg, border: `1px solid ${C.border2}`,
                  borderRadius: 6, padding: '5px 8px', fontSize: 12, color: C.text,
                }}
              />

              <input
                placeholder="Label" value={leg.label}
                onChange={e => updateLeg(i, 'label', e.target.value)}
                style={{
                  width: 100, background: C.bg, border: `1px solid ${C.border2}`,
                  borderRadius: 6, padding: '5px 8px', fontSize: 12, color: C.text,
                }}
              />

              {legs.length > 2 && (
                <button
                  onClick={() => removeLeg(i)}
                  style={{
                    background: 'none', border: 'none', cursor: 'pointer',
                    color: C.neg, fontSize: 16, lineHeight: 1, padding: '0 4px',
                  }}
                >×</button>
              )}
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', gap: 10 }}>
          {legs.length < 6 && (
            <button
              onClick={addLeg}
              style={{
                background: 'none', border: `1px solid ${C.border2}`, borderRadius: 7,
                padding: '7px 14px', fontSize: 12, color: C.muted, cursor: 'pointer',
              }}
            >
              + Add leg
            </button>
          )}
          <button
            onClick={submit}
            style={{
              background: C.accent, color: '#fff', border: 'none', borderRadius: 7,
              padding: '7px 22px', fontWeight: 700, fontSize: 13, cursor: 'pointer',
            }}
          >
            {loading ? 'Calculating…' : 'Calculate'}
          </button>
        </div>
      </div>

      {/* ── Error ────────────────────────────────────────────────────── */}
      {error && (
        <div style={{
          background: '#2a1515', border: '1px solid #6b2020', borderRadius: 8,
          color: C.neg, fontSize: 13, padding: '10px 14px', marginBottom: 16,
        }}>
          ⚠ {error}
        </div>
      )}

      {/* ── Results ──────────────────────────────────────────────────── */}
      {result && !loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

          {/* Joint prob summary strip */}
          <div style={{
            background: C.surface, border: `1px solid ${C.border}`,
            borderRadius: 12, padding: '14px 20px',
            display: 'flex', flexWrap: 'wrap', gap: 24,
          }}>
            {[
              { label: 'Corr-adjusted prob', v: pct(result.joint_prob_corr_adjusted), color: C.accent },
              { label: 'Naive (independent)', v: pct(result.joint_prob_naive),         color: C.muted  },
              { label: 'Net payout (b)',      v: `${result.parlay_net_payout.toFixed(2)}×`, color: C.pos },
            ].map(item => (
              <div key={item.label}>
                <div style={{ fontSize: 24, fontWeight: 800, color: item.color, fontVariantNumeric: 'tabular-nums' }}>
                  {item.v}
                </div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>{item.label}</div>
              </div>
            ))}
          </div>

          {/* Kelly comparison */}
          <KellyComparison
            corrVariant={result.kelly.corr_adjusted}
            naiveVariant={result.kelly.naive}
            bankroll={parseFloat(bankroll) || 1000}
          />

          {/* Per-leg marginals */}
          <div style={{
            background: C.surface, border: `1px solid ${C.border}`,
            borderRadius: 12, padding: '14px 20px',
          }}>
            <SectionHeader>Per-leg marginal probabilities</SectionHeader>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                  {['Label', 'Side', 'Outcome', 'Line', 'Odds', 'Sim prob'].map(h => (
                    <th key={h} style={{ padding: '5px 10px', textAlign: 'left', color: C.muted, fontWeight: 600, whiteSpace: 'nowrap' }}>
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
                    <td style={{ padding: '6px 10px', fontWeight: 600, color: C.text }}>{l.label ?? `Leg ${i + 1}`}</td>
                    <td style={{ padding: '6px 10px', color: C.muted }}>{l.side}</td>
                    <td style={{ padding: '6px 10px', color: C.muted }}>{l.outcome}</td>
                    <td style={{ padding: '6px 10px', fontVariantNumeric: 'tabular-nums', color: C.text }}>{l.line ?? '—'}</td>
                    <td style={{ padding: '6px 10px', fontVariantNumeric: 'tabular-nums', color: C.text }}>{l.odds ?? '—'}</td>
                    <td style={{ padding: '6px 10px', fontVariantNumeric: 'tabular-nums', fontWeight: 700, color: C.accent }}>{pct(l.sim_prob)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Correlation matrix */}
          <div style={{
            background: C.surface, border: `1px solid ${C.border}`,
            borderRadius: 12, padding: '14px 20px',
          }}>
            <SectionHeader>
              Correlation matrix
              <span style={{ fontWeight: 400, marginLeft: 8, textTransform: 'none', letterSpacing: 0 }}>
                — blue = positive, red = negative
              </span>
            </SectionHeader>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr>
                    <th style={{ padding: '5px 12px' }} />
                    {result.legs.map((l, i) => (
                      <th key={i} style={{ padding: '5px 12px', color: C.muted, fontWeight: 600, textAlign: 'center' }}>
                        {l.label ?? `L${i + 1}`}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.correlation_matrix.map((row, ri) => (
                    <tr key={ri}>
                      <td style={{ padding: '5px 12px', color: C.muted, fontWeight: 600, whiteSpace: 'nowrap' }}>
                        {result.legs[ri]?.label ?? `L${ri + 1}`}
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

        </div>
      )}
    </div>
  )
}
