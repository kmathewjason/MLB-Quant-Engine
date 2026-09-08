import { useState } from 'react';
import { TabBar } from './components/TabBar';
import { SlateTab } from './components/SlateTab';
import { SimTab } from './components/SimTab';
import { SGPTab } from './components/SGPTab';
import { BacktestTab } from './components/BacktestTab';

const TABS = [
  { id: 'slate',    label: "Today's Slate" },
  { id: 'sim',      label: 'Game Simulation' },
  { id: 'sgp',      label: 'SGP Builder' },
  { id: 'backtest', label: 'Backtest' },
];

export default function App() {
  const [tab, setTab] = useState('slate');

  return (
    <div style={{
      fontFamily: '-apple-system, "Segoe UI", system-ui, sans-serif',
      fontSize: 14,
      lineHeight: 1.6,
      color: '#1f2328',
      background: '#f7f8fa',
      minHeight: '100vh',
    }}>
      {/* Header */}
      <header style={{
        background: '#fff',
        borderBottom: '1px solid #e5e7eb',
        padding: '0 24px',
        display: 'flex',
        alignItems: 'center',
        height: 52,
        position: 'sticky',
        top: 0,
        zIndex: 10,
      }}>
        <span style={{ fontWeight: 700, fontSize: 16, color: '#111827', letterSpacing: '-0.3px' }}>
          ⚾ MLB Quant Engine
        </span>
      </header>

      {/* Main content */}
      <main style={{ maxWidth: 960, margin: '0 auto', padding: '20px 16px 60px' }}>
        <TabBar tabs={TABS} active={tab} onChange={setTab} />

        {tab === 'slate'    && <SlateTab />}
        {tab === 'sim'      && <SimTab />}
        {tab === 'sgp'      && <SGPTab />}
        {tab === 'backtest' && <BacktestTab />}
      </main>
    </div>
  );
}
