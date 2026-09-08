import { useState } from 'react';
import { fetchSGP } from '../api';
import type { SGPLeg, SGPData } from '../types';
import { ErrorBanner, Spinner, Card, Pct, Badge, Table } from './ui';

const OUTCOMES = ['moneyline', 'over', 'under', 'spread'] as const;
const SIDES = ['home', 'away'] as const;

function CorrelationMatrix({ matrix, legs }: { matrix: number[][]; legs: { label?: string }[] }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr>
            <th style={{ padding: '4px 8px' }} />
            {legs.map((l, i) => (
              <th key={i} style={{ padding: '4px 8px', color: '#6b7280', fontWeight: 600 }}>
                {l.label ?? `L${i + 1}`}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, ri) => (
            <tr key={ri}>
              <td style={{ padding: '4px 8px', color: '#6b7280', fontWeight: 600 }}>
                {legs[ri]?.label ?? `L${ri + 1}`}
              </td>
              {row.map((v, ci) => {
                const abs = Math.abs(v);
                const bg = ri === ci
                  ? '#f3f4f6'
                  : v > 0
                    ? `rgba(59,130,212,${abs * 0.4})`
                    : `rgba(239,68,68,${abs * 0.4})`;
                return (
                  <td key={ci} style={{
                    padding: '4px 8px',
                    background: bg,
                    textAlign: 'center',
                    borderRadius: 3,
                  }}>
                    {v.toFixed(3)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

type LegDraft = {
  side: 'home' | 'away';
  outcome: typeof OUTCOMES[number];
  line: string;
  odds: string;
  label: string;
};

const emptyLeg = (): LegDraft => ({
  side: 'home', outcome: 'moneyline', line: '', odds: '-110', label: '',
});

export function SGPTab() {
  const [gameId, setGameId] = useState('');
  const [bankroll, setBankroll] = useState('1000');
  const [nSims, setNSims] = useState('20000');
  const [legs, setLegs] = useState<LegDraft[]>([emptyLeg(), emptyLeg()]);
  const [result, setResult] = useState<SGPData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const addLeg = () => {
    if (legs.length < 6) setLegs(l => [...l, emptyLeg()]);
  };

  const removeLeg = (i: number) => {
    if (legs.length > 2) setLegs(l => l.filter((_, idx) => idx !== i));
  };

  const updateLeg = (i: number, field: keyof LegDraft, value: string) =>
    setLegs(l => l.map((leg, idx) => idx === i ? { ...leg, [field]: value } : leg));

  const submit = () => {
    const gid = parseInt(gameId);
    if (!gid) { setError('Enter a game PK'); return; }

    const parsed: SGPLeg[] = legs.map(l => {
      const leg: SGPLeg = { side: l.side, outcome: l.outcome, label: l.label || undefined };
      const line = parseFloat(l.line);
      const odds = parseFloat(l.odds);
      if (!isNaN(line)) leg.line = line;
      if (!isNaN(odds)) leg.odds = odds;
      return leg;
    });

    setLoading(true);
    setError(null);
    fetchSGP(gid, parsed, parseFloat(bankroll) || 1000, parseInt(nSims) || 20_000)
      .then(env => setResult(env.data))
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false));
  };

  return (
    <div>
      <Card title="Same-Game Parlay Builder">
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 12, alignItems: 'flex-end' }}>
          {[
            { label: 'Game PK', v: gameId, set: setGameId, ph: 'e.g. 718976' },
            { label: 'Bankroll ($)', v: bankroll, set: setBankroll, ph: '1000' },
            { label: '# Sims', v: nSims, set: setNSims, ph: '20000' },
          ].map(f => (
            <label key={f.label} style={{ display: 'flex', flexDirection: 'column', gap: 3, fontSize: 12 }}>
              {f.label}
              <input value={f.v} onChange={e => f.set(e.target.value)} placeholder={f.ph}
                style={{ border: '1px solid #d1d5db', borderRadius: 5, padding: '5px 9px', fontSize: 13, width: 120 }} />
            </label>
          ))}
        </div>

        {/* Legs */}
        {legs.map((leg, i) => (
          <div key={i} style={{
            display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6, flexWrap: 'wrap',
            background: '#f9fafb', padding: '8px 10px', borderRadius: 6,
          }}>
            <span style={{ fontSize: 12, color: '#6b7280', width: 20 }}>#{i + 1}</span>
            <select value={leg.side} onChange={e => updateLeg(i, 'side', e.target.value)}
              style={{ border: '1px solid #d1d5db', borderRadius: 5, padding: '4px 6px', fontSize: 12 }}>
              {SIDES.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
            <select value={leg.outcome} onChange={e => updateLeg(i, 'outcome', e.target.value)}
              style={{ border: '1px solid #d1d5db', borderRadius: 5, padding: '4px 6px', fontSize: 12 }}>
              {OUTCOMES.map(o => <option key={o} value={o}>{o}</option>)}
            </select>
            <input placeholder="Line" value={leg.line} onChange={e => updateLeg(i, 'line', e.target.value)}
              style={{ border: '1px solid #d1d5db', borderRadius: 5, padding: '4px 6px', fontSize: 12, width: 65 }} />
            <input placeholder="Odds" value={leg.odds} onChange={e => updateLeg(i, 'odds', e.target.value)}
              style={{ border: '1px solid #d1d5db', borderRadius: 5, padding: '4px 6px', fontSize: 12, width: 65 }} />
            <input placeholder="Label" value={leg.label} onChange={e => updateLeg(i, 'label', e.target.value)}
              style={{ border: '1px solid #d1d5db', borderRadius: 5, padding: '4px 6px', fontSize: 12, width: 90 }} />
            {legs.length > 2 && (
              <button onClick={() => removeLeg(i)}
                style={{ border: 'none', background: 'transparent', cursor: 'pointer', color: '#ef4444', fontSize: 16 }}>
                ×
              </button>
            )}
          </div>
        ))}

        <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
          {legs.length < 6 && (
            <button onClick={addLeg}
              style={{ border: '1px solid #d1d5db', borderRadius: 5, padding: '5px 12px', cursor: 'pointer', fontSize: 12 }}>
              + Add Leg
            </button>
          )}
          <button onClick={submit} style={{
            background: '#3b82d4', color: '#fff', border: 'none', borderRadius: 5,
            padding: '6px 16px', cursor: 'pointer', fontSize: 13, fontWeight: 600,
          }}>
            Calculate
          </button>
        </div>
      </Card>

      {error && <ErrorBanner message={error} />}
      {loading && <Spinner />}

      {result && !loading && (
        <>
          <Card title="Parlay Summary">
            <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
              {[
                { label: 'Corr-adjusted prob', v: result.joint_prob_corr_adjusted, color: '#3b82d4' },
                { label: 'Naive (independent)', v: result.joint_prob_naive, color: '#9ca3af' },
                { label: 'Net payout (b)', v: result.parlay_net_payout, color: '#059669', raw: true },
              ].map(item => (
                <div key={item.label} style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 24, fontWeight: 700, color: item.color }}>
                    {item.raw ? item.v.toFixed(2) + 'x' : <Pct v={item.v as number} />}
                  </div>
                  <div style={{ fontSize: 11, color: '#9ca3af' }}>{item.label}</div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Kelly Sizing">
            <Table
              headers={['Variant', 'EV', 'Full Kelly', '¼ Kelly', 'Stake']}
              rows={[
                ['Corr-adjusted',
                  <Badge value={result.kelly.corr_adjusted.ev} />,
                  <Pct v={result.kelly.corr_adjusted.f_star} />,
                  <Pct v={result.kelly.corr_adjusted.f_quarter} />,
                  `$${result.kelly.corr_adjusted.stake_units.toFixed(0)}`,
                ],
                ['Naive',
                  <Badge value={result.kelly.naive.ev} />,
                  <Pct v={result.kelly.naive.f_star} />,
                  <Pct v={result.kelly.naive.f_quarter} />,
                  `$${result.kelly.naive.stake_units.toFixed(0)}`,
                ],
              ]}
            />
          </Card>

          <Card title="Per-Leg Marginal Probabilities">
            <Table
              headers={['Leg', 'Side', 'Outcome', 'Line', 'Odds', 'Sim Prob']}
              rows={result.legs.map(l => [
                l.label ?? '—',
                l.side,
                l.outcome,
                l.line != null ? String(l.line) : '—',
                l.odds != null ? String(l.odds) : '—',
                <strong><Pct v={l.sim_prob} /></strong>,
              ])}
            />
          </Card>

          <Card title="Correlation Matrix">
            <CorrelationMatrix matrix={result.correlation_matrix} legs={result.legs} />
            <p style={{ fontSize: 11, color: '#9ca3af', marginTop: 8 }}>
              Blue = positive correlation, red = negative. Diagonal = 1.
            </p>
          </Card>
        </>
      )}
    </div>
  );
}
