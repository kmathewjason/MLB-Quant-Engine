interface Tab { id: string; label: string }
interface Props { tabs: Tab[]; active: string; onChange: (id: string) => void }

export default function TabBar({ tabs, active, onChange }: Props) {
  return (
    <div className="flex border-b border-border mb-5">
      {tabs.map(t => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          className={[
            'px-4 py-2.5 text-sm font-medium border-b-2 -mb-px transition-colors',
            active === t.id
              ? 'border-brand text-brand'
              : 'border-transparent text-muted hover:text-gray-700',
          ].join(' ')}
        >
          {t.label}
        </button>
      ))}
    </div>
  )
}
