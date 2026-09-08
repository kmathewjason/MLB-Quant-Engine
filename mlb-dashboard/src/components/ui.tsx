import React from 'react';

interface Props {
  message: string;
}

export function ErrorBanner({ message }: Props) {
  return (
    <div style={{
      background: '#fef2f2',
      border: '1px solid #fca5a5',
      borderRadius: 6,
      padding: '10px 14px',
      color: '#991b1b',
      fontSize: 13,
      margin: '12px 0',
    }}>
      ⚠ {message}
    </div>
  );
}

export function Spinner() {
  return (
    <div style={{ textAlign: 'center', padding: 40, color: '#6b7280', fontSize: 14 }}>
      Loading…
    </div>
  );
}

export function Badge({ value, threshold = 0 }: { value: number; threshold?: number }) {
  const positive = value > threshold;
  return (
    <span style={{
      display: 'inline-block',
      padding: '1px 7px',
      borderRadius: 10,
      fontSize: 11,
      fontWeight: 600,
      background: positive ? '#dcfce7' : '#fee2e2',
      color: positive ? '#15803d' : '#b91c1c',
    }}>
      {value >= 0 ? '+' : ''}{(value * 100).toFixed(1)}%
    </span>
  );
}

export function Pct({ v, decimals = 1 }: { v: number; decimals?: number }) {
  return <span>{(v * 100).toFixed(decimals)}%</span>;
}

export function Card({ title, children }: { title?: string; children: React.ReactNode }) {
  return (
    <div style={{
      background: '#fff',
      border: '1px solid #e5e7eb',
      borderRadius: 8,
      padding: '16px 20px',
      marginBottom: 16,
    }}>
      {title && (
        <h3 style={{ margin: '0 0 12px', fontSize: 14, fontWeight: 600, color: '#374151' }}>
          {title}
        </h3>
      )}
      {children}
    </div>
  );
}

export function Table({
  headers,
  rows,
}: {
  headers: string[];
  rows: (string | number | React.ReactNode)[][];
}) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr>
            {headers.map((h) => (
              <th key={h} style={{
                textAlign: 'left',
                padding: '6px 10px',
                borderBottom: '1px solid #e5e7eb',
                color: '#6b7280',
                fontWeight: 600,
                whiteSpace: 'nowrap',
              }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri} style={{ borderBottom: '1px solid #f3f4f6' }}>
              {row.map((cell, ci) => (
                <td key={ci} style={{ padding: '6px 10px', color: '#1f2937' }}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
