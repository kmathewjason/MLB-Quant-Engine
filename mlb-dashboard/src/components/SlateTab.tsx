import { useState, useEffect } from 'react';
import { fetchDailyPredictions } from '../api';
import type { DailyGame, MarketRow } from '../types';
import { ErrorBanner, Spinner, Badge, Pct, Card, Table } from './ui';

function WinBar({ home, away, homeTeam, awayTeam }: {
  home: number; away: number; homeTeam: string; awayTeam: string;
}) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 2 }}>
        <span style={{ color: '#374151', fontWeight: 600 }}>{homeTeam}</span>
        <span style={{ color: '#6b7280' }}>{awayTeam}</span>
      </div>
      <div style={{ display: 'flex', height: 18, borderRadius: 3, overflow: 'hidden' }}>
        <div style={{
          width: `${home * 100}%`, background: '#3b82d4',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 11, color: '#fff', fontWeight: 600,
        }}>
          {(home * 100).toFixed(0)}%
        </div>
        <div style={{
          flex: 1, background: '#e5e7eb',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 11, color: '#374151', fontWeight: 600,
        }}>
          {(away * 100).toFixed(0)}%
        </div>
      </div>
    </div>
  );
}

function RunBoxes({ label, dist }: { label: string; dist: { mean: number; p10: number; p50: number; p90: number } }) {
  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 4 }}>
      <span style={{ fontSize: 12, color: '#6b7280', width: 80, flexShrink: 0 }}>{label}</span>
      <span style={{ fontSize: 12, color: '#374151' }}>
        μ <strong>{dist.mean.toFixed(1)}</strong>  ·
        p10-p90: {dist.p10.toFixed(0)}–{dist.p90.toFixed(0)}  ·
        median: {dist.p50.toFixed(0)}
      </span>
    </div>
  );
}

function MarketsTable({ markets }: { markets: MarketRow[] }) {
  if (markets.length === 0) return <p style={{ color: '#9ca3af', fontSize: 12 }}>No markets available</p>;
  return (
    <Table
      headers={['Market', 'Model %', 'Market %', 'Edge', 'EV', '¼K Stake']}
      rows={markets.map(m => [
        m.label,
        <Pct v={m.model_prob} />,
        m.market_prob != null ? <Pct v={m.market_prob} /> : <span style={{ color: '#9ca3af' }}>—</span>,
        m.edge != null ? <Badge value={m.edge} /> : <span style={{ color: '#9ca3af' }}>—</span>,
        m.ev != null ? <Badge value={m.ev} /> : <span style={{ color: '#9ca3af' }}>—</span>,
        m.stake_units != null ? `$${m.stake_units.toFixed(0)}` : <span style={{ color: '#9ca3af' }}>—</span>,
      ])}
    />
  );
}

function GameCard({ game }: { game: DailyGame }) {
  const [expanded, setExpanded] = useState(false);
  const sim = game.simulation;

  return (
    <Card>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <div>
          <span style={{ fontWeight: 700, fontSize: 15, color: '#111827' }}>
            {game.away_team} @ {game.home_team}
          </span>
          <span style={{ marginLeft: 10, fontSize: 12, color: '#9ca3af' }}>{game.status}</span>
        </div>
        <button
          onClick={() => setExpanded(e => !e)}
          style={{
            border: '1px solid #e5e7eb', borderRadius: 5, background: '#f9fafb',
            padding: '3px 10px', cursor: 'pointer', fontSize: 12, color: '#374151',
          }}
        >
          {expanded ? 'Less' : 'Details'}
        </button>
      </div>

      <div style={{ fontSize: 11, color: '#9ca3af', marginBottom: 8 }}>
        SP: {game.away_probable_pitcher} vs {game.home_probable_pitcher}
      </div>

      {/* Win probability bar */}
      <WinBar
        home={sim.home_win_prob}
        away={sim.away_win_prob}
        homeTeam={game.home_team}
        awayTeam={game.away_team}
      />

      {/* Run distributions */}
      <div style={{ marginTop: 10 }}>
        <RunBoxes label="Home runs" dist={sim.home_runs} />
        <RunBoxes label="Away runs" dist={sim.away_runs} />
        <RunBoxes label="Total"     dist={sim.total_runs} />
      </div>

      {/* Totals quick view */}
      {Object.keys(sim.totals).length > 0 && (
        <div style={{ marginTop: 8, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          {Object.entries(sim.totals).map(([k, v]) => (
            <span key={k} style={{ fontSize: 12, background: '#eff6ff', padding: '2px 8px', borderRadius: 4, color: '#1d4ed8' }}>
              O{k.split('_')[1].replace(/_/g, '.')}: <strong>{(v * 100).toFixed(0)}%</strong>
            </span>
          ))}
        </div>
      )}

      {/* Markets (collapsed by default) */}
      {expanded && (
        <div style={{ marginTop: 14 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#374151', marginBottom: 6 }}>Markets</div>
          <MarketsTable markets={game.markets} />
        </div>
      )}

      <div style={{ fontSize: 11, color: '#d1d5db', marginTop: 8 }}>
        {sim.n_sims.toLocaleString()} simulations · pk {game.game_pk}
      </div>
    </Card>
  );
}

export function SlateTab() {
  const [games, setGames] = useState<DailyGame[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [date, setDate] = useState('');

  const load = () => {
    setLoading(true);
    setError(null);
    fetchDailyPredictions(date || undefined, 10_000)
      .then(env => setGames(env.data))
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 16, alignItems: 'center' }}>
        <input
          type="date"
          value={date}
          onChange={e => setDate(e.target.value)}
          style={{ border: '1px solid #d1d5db', borderRadius: 5, padding: '5px 9px', fontSize: 13 }}
        />
        <button onClick={load} style={{
          background: '#3b82d4', color: '#fff', border: 'none', borderRadius: 5,
          padding: '6px 14px', cursor: 'pointer', fontSize: 13, fontWeight: 600,
        }}>
          Refresh
        </button>
        {loading && <span style={{ fontSize: 12, color: '#9ca3af' }}>Loading…</span>}
      </div>

      {error && <ErrorBanner message={error} />}
      {!loading && games.length === 0 && !error && (
        <p style={{ color: '#9ca3af', fontSize: 13 }}>No games found for this date.</p>
      )}
      {loading ? <Spinner /> : games.map(g => <GameCard key={g.game_pk} game={g} />)}
    </div>
  );
}
