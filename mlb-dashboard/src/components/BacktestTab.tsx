import { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid,
} from 'recharts';
import { fetchBacktestReport } from '../api';
import type { BacktestData, FoldMetric } from '../types';
import { ErrorBanner, Spinner, Card, Table } from './ui';

function FoldsChart({ folds }: { folds: FoldMetric[] }) {
  const data = folds.map(f => ({
    fold: `F${f.fold}`,
    log_loss: parseFloat(f.log_loss.toFixed(3)),
    brier: parseFloat(f.brier.toFixed(3)),
    accuracy: parseFloat((f.accuracy * 100).toFixed(1)),
  }));
  return (
    <ResponsiveContainer width="100%" height={180}>
      <LineChart data={data} margin={{ top: 4, right: 10, bottom: 0, left: -10 }}>
        <CartesianGrid stroke="#f3f4f6" />
        <XAxis dataKey="fold" tick={{ fontSize: 11 }} />
        <YAxis tick={{ fontSize: 11 }} />
        <Tooltip contentStyle={{ fontSize: 11 }} />
        <Line type="monotone" dataKey="log_loss" stroke="#3b82d4" dot={false} name="Log-loss" />
        <Line type="monotone" dataKey="brier"    stroke="#059669" dot={false} name="Brier" />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function BacktestTab() {
  const [data, setData] = useState<BacktestData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    setError(null);
    fetchBacktestReport()
      .then(env => setData(env.data))
      .catch(e => setError(
        e instanceof Error && e.message.startsWith('404')
          ? 'No backtest data found. Run the pipeline first (python main.py --backtest).'
          : String(e),
      ))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div>
      <div style={{ marginBottom: 12 }}>
        <button onClick={load} style={{
          background: '#3b82d4', color: '#fff', border: 'none', borderRadius: 5,
          padding: '6px 14px', cursor: 'pointer', fontSize: 13, fontWeight: 600,
        }}>
          Refresh
        </button>
      </div>

      {error && <ErrorBanner message={error} />}
      {loading && <Spinner />}

      {data && !loading && (
        <>
          {/* Walk-forward summary */}
          {data.walk_forward.available && data.walk_forward.summary && (
            <Card title="Walk-Forward Summary">
              <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', marginBottom: 12 }}>
                {[
                  { label: 'Folds', v: data.walk_forward.summary.n_folds },
                  { label: 'OOS preds', v: data.walk_forward.summary.n_oos_predictions.toLocaleString() },
                  { label: 'OOS Log-loss', v: data.walk_forward.summary.oos_log_loss.toFixed(3) },
                  { label: 'OOS Accuracy', v: (data.walk_forward.summary.oos_accuracy * 100).toFixed(1) + '%' },
                  { label: 'Mean Brier', v: data.walk_forward.summary.mean_fold_brier.toFixed(3) },
                ].map(item => (
                  <div key={item.label} style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: 20, fontWeight: 700, color: '#1f2937' }}>{item.v}</div>
                    <div style={{ fontSize: 11, color: '#9ca3af' }}>{item.label}</div>
                  </div>
                ))}
              </div>
              {data.walk_forward.fold_metrics && data.walk_forward.fold_metrics.length > 0 && (
                <>
                  <div style={{ fontSize: 12, fontWeight: 600, color: '#374151', marginBottom: 6 }}>
                    Per-fold log-loss &amp; Brier
                  </div>
                  <FoldsChart folds={data.walk_forward.fold_metrics} />
                </>
              )}
            </Card>
          )}

          {/* Calibration */}
          {data.calibration.available && (
            <Card title="Calibration">
              {data.calibration.brier_decomposition && (
                <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 12 }}>
                  {Object.entries(data.calibration.brier_decomposition).map(([k, v]) => (
                    <div key={k} style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 16, fontWeight: 700, color: '#1f2937' }}>
                        {(v as number).toFixed(4)}
                      </div>
                      <div style={{ fontSize: 11, color: '#9ca3af' }}>
                        {k.replace(/_/g, ' ')}
                      </div>
                    </div>
                  ))}
                </div>
              )}
              {data.calibration.report.length > 0 && (
                <Table
                  headers={['Class', 'ECE', 'Brier', 'BSS', 'Base Rate', 'N+']}
                  rows={data.calibration.report.map(r => [
                    String(r.class_name),
                    Number(r.ece).toFixed(4),
                    Number(r.brier_score).toFixed(4),
                    Number(r.brier_skill_score).toFixed(3),
                    (Number(r.base_rate) * 100).toFixed(1) + '%',
                    String(r.n_positive),
                  ])}
                />
              )}
            </Card>
          )}

          {/* CLV */}
          {data.clv.available && data.clv.summary && (
            <Card title="Closing-Line Value">
              <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
                {[
                  { label: 'Bets', v: data.clv.summary.n_bets.toLocaleString() },
                  { label: '% Positive CLV', v: (data.clv.summary.pct_positive_clv * 100).toFixed(0) + '%' },
                  { label: 'Mean CLV (log-odds)', v: data.clv.summary.mean_clv_log_odds.toFixed(3) },
                  { label: 'CLV t-stat', v: data.clv.summary.clv_tstat.toFixed(2) },
                  { label: 'p-value', v: data.clv.summary.clv_pvalue.toFixed(3) },
                  { label: 'ROI%', v: data.clv.summary.roi_pct.toFixed(2) + '%' },
                ].map(item => (
                  <div key={item.label} style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: 18, fontWeight: 700, color: '#1f2937' }}>{item.v}</div>
                    <div style={{ fontSize: 11, color: '#9ca3af' }}>{item.label}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {!data.walk_forward.available && !data.calibration.available && !data.clv.available && (
            <p style={{ color: '#9ca3af', fontSize: 13 }}>
              No backtest data found. Run the pipeline first.
            </p>
          )}
        </>
      )}
    </div>
  );
}
