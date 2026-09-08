import { useState } from 'react'
import TabBar from './components/TabBar'
import SlateTab from './components/SlateTab'
import SGPTab from './components/SGPTab'
import BacktestTab from './components/BacktestTab'

const TABS = [
  { id: 'slate',    label: "Today's Slate" },
  { id: 'sgp',      label: 'SGP Builder' },
  { id: 'backtest', label: 'Backtest' },
]

export default function App() {
  const [tab, setTab] = useState('slate')

  return (
    <div className="min-h-screen bg-surface">
      {/* Header */}
      <header className="sticky top-0 z-20 bg-white border-b border-border px-4 h-13 flex items-center gap-3 shadow-sm">
        <span className="text-lg">⚾</span>
        <span className="font-bold text-gray-900 tracking-tight">MLB Quant Engine</span>
        <span className="ml-auto text-xs text-muted font-mono">v0.1</span>
      </header>

      <main className="max-w-screen-xl mx-auto px-4 pt-4 pb-16">
        <TabBar tabs={TABS} active={tab} onChange={setTab} />
        {tab === 'slate'    && <SlateTab />}
        {tab === 'sgp'      && <SGPTab />}
        {tab === 'backtest' && <BacktestTab />}
      </main>
    </div>
  )
}
