import { useState } from 'react'
import SlateTab from './components/SlateTab'
import SGPTab from './components/SGPTab'
import BacktestTab from './components/BacktestTab'

const NAV = [
  { id: 'slate',    label: "Today's Slate",  icon: '⚾' },
  { id: 'sgp',      label: 'SGP Builder',    icon: '🔗' },
  { id: 'backtest', label: 'Model Performance', icon: '📊' },
]

export default function App() {
  const [tab, setTab] = useState('slate')

  return (
    <div className="min-h-screen flex" style={{ background: '#0f1117', color: '#e8eaf0' }}>

      {/* ── Sidebar ─────────────────────────────────────────────────── */}
      <aside style={{
        width: 220, minHeight: '100vh', background: '#16181f',
        borderRight: '1px solid #22252e', display: 'flex', flexDirection: 'column',
        position: 'sticky', top: 0, flexShrink: 0,
      }}>
        {/* Logo */}
        <div style={{ padding: '22px 20px 18px', borderBottom: '1px solid #22252e' }}>
          <div style={{ fontWeight: 800, fontSize: 15, letterSpacing: '-0.3px', color: '#fff' }}>
            ⚾ MLB Quant
          </div>
          <div style={{ fontSize: 10, color: '#5a6072', marginTop: 2, fontFamily: 'monospace' }}>
            Engine v0.2 · Monte Carlo
          </div>
        </div>

        {/* Nav */}
        <nav style={{ padding: '12px 10px', flex: 1 }}>
          {NAV.map(n => {
            const active = tab === n.id
            return (
              <button key={n.id} onClick={() => setTab(n.id)} style={{
                display: 'flex', alignItems: 'center', gap: 10,
                width: '100%', padding: '9px 12px', borderRadius: 8, border: 'none',
                background: active ? '#1e8aff18' : 'transparent',
                color: active ? '#4faeff' : '#8892a4',
                fontWeight: active ? 600 : 400, fontSize: 13.5,
                cursor: 'pointer', marginBottom: 2, textAlign: 'left',
                borderLeft: active ? '2px solid #4faeff' : '2px solid transparent',
                transition: 'all 0.12s',
              }}>
                <span style={{ fontSize: 15 }}>{n.icon}</span>
                {n.label}
              </button>
            )
          })}
        </nav>

        {/* Footer */}
        <div style={{ padding: '14px 16px', borderTop: '1px solid #22252e', fontSize: 10, color: '#3a4050' }}>
          Powered by Retrosheet · OpenWeather · The Odds API
        </div>
      </aside>

      {/* ── Main content ─────────────────────────────────────────────── */}
      <main style={{ flex: 1, minWidth: 0, padding: '28px 32px', overflowX: 'hidden' }}>
        {tab === 'slate'    && <SlateTab />}
        {tab === 'sgp'      && <SGPTab />}
        {tab === 'backtest' && <BacktestTab />}
      </main>
    </div>
  )
}
