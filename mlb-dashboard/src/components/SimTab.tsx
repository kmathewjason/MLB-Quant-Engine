import { useState, useCallback } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, CartesianGrid,
} from 'recharts';
import { fetchGameSimulation } from '../api';
import type { GameSimData, DistWithHist } from '../types';
import { ErrorBanner, Spinner, Card, Pct } from './ui';

function HistChart({
  dist,
  color,
  title,
  refLine,
}: {
  dist: DistWithHist;
  color: string;
  title: string;
  refLine?: number;
}) {
  const { histogram } = dist;
  const data = histogram.counts.map((c, i) => ({
    x: ((histogram.edges[i] + histogram.edges[i + 1]) / 2).toFixed(1),
    count: c,
  }));
  return (
    <div>
      <div style={{ fontSize: 12, fontWeight: 600, color: '#374151', marginBottom: 4 }}>{title}</div>
      <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 6 }}>
        μ={dist.mean.toFixed(2)} σ={dist.std.toFixed(2)}
        &nbsp;·&nbsp;p10–p90: {dist.p10.toFixed(0)}–{dist.p90.toFixed(0)}
      </div>
      <ResponsiveContainer width="100%" height={140}>
        <BarChart data={data} margin={{ top: 0, right: 4, bottom: 0, left: -10 }}>
          <CartesianGrid vertical={false} stroke="#f3f4f6" />
          <XAxis dataKey="x" tick={{ fontSize: 10 }} tickLine={false} />
          <YAxis tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
          <Tooltip
            contentStyle={{ fontSize: 11, padding: '4px 8px' }}
            formatter={(v: number) => [v.toLocaleString(), 'sims']}
          />
          {refLine !== undefined && (
            <ReferenceLine x={String(refLine.toFixed(1))} stroke="#ef4444" strokeDasharray="3 3" />
          )}
          <Bar dataKey="count" fill={color} radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function SimTab() {
  const [gameId, setGameId] = useState('');
  const [totalLine, setTotalLine] = useState('8.5');
  const [runLine, setRunLine] = useState('-1.5');
  const [nSims, setNSims] = useState('50000');
  const [data, setData] = useState<GameSimData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(() => {
    const gid = parseInt(gameId);
    if (!gid) { setError('Enter a valid game ID'); return; }
    setLoading(true);
    setError(null);
    fetchGameSimulation(gid, {
      nSims: parseInt(nSims) || 50_000,
      totalLine: parseFloat(totalLine) || 8.5,
      runLine: parseFloat(runLine) || -1.5,
      bins: 20,
    })
      .then(env => setData(env.data))
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false));
  }, [gameId, totalLine, runLine, nSims]);

  return (
    <div>
      <Card title="Parameters">
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'flex-end' }}>
          {[
            { label: 'Game PK', value: gameId, set: setGameId, ph: 'e.g. 718976' },
            { label: 'Total Line', value: totalLine, set: setTotalLine, ph: '8.5' },
            { label: 'Run Line', value: runLine, set: setRunLine, ph: '-1.5' },
            { label: '# Sims', value: nSims, set: setNSims, ph: '50000' },
          ].map(f => (
            <label key={f.label} style={{ display: 'flex', flexDirection: 'column', gap: 3, fontSize: 12, color: '#374151' }}>
              {f.label}
              <input
                value={f.value}
                onChange={e => f.set(e.target.value)}
                placeholder={f.ph}
                style={{ border: '1px solid #d1d5db', borderRadius: 5, padding: '5px 9px', fontSize: 13, width: 110 }}
              />
            </label>
          ))}
          <button onClick={run} style={{
            background: '#3b82d4', color: '#fff', border: 'none', borderRadius: 5,
            padding: '6px 16px', cursor: 'pointer', fontSize: 13, fontWeight: 600,
          }}>
            Simulate
          </button>
        </div>
      </Card>

      {error && <ErrorBanner message={error} />}
      {loading && <Spinner />}

      {data && !loading && (
        <>
          <Card title="Win Probabilities">
            <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
              {[
                { label: 'Home win', v: data.home_win_prob, color: '#3b82d4' },
                { label: 'Away win', v: data.away_win_prob, color: '#6b7280' },
                { label: `Over ${totalLine}`, v: data.over_prob, color: '#059669' },
                { label: `Under ${totalLine}`, v: data.under_prob, color: '#9ca3af' },
                { label: `Spread cover (${runLine})`, v: data.spread_cover_prob, color: '#7c3aed' },
              ].map(item => (
                <div key={item.label} style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 22, fontWeight: 700, color: item.color }}>
                    <Pct v={item.v} />
                  </div>
                  <div style={{ fontSize: 11, color: '#9ca3af' }}>{item.label}</div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Score Distributions">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
              <HistChart dist={data.home_runs} color="#3b82d4" title="Home Runs" />
              <HistChart dist={data.away_runs} color="#6b7280" title="Away Runs" />
            </div>
            <div style={{ marginTop: 16 }}>
              <HistChart
                dist={data.total_runs}
                color="#059669"
                title="Total Runs"
                refLine={parseFloat(totalLine)}
              />
            </div>
          </Card>

          <Card title="Margin Distribution (Home − Away)">
            <HistChart
              dist={{ ...data.margin_dist, histogram: data.margin_dist.histogram, p10: 0, p25: 0, p50: 0, p75: 0, p90: 0, std: data.margin_dist.std }}
              color="#7c3aed"
              title=""
              refLine={0}
            />
          </Card>
        </>
      )}
    </div>
  );
}
