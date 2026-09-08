interface Tab {
  id: string;
  label: string;
}

interface Props {
  tabs: Tab[];
  active: string;
  onChange: (id: string) => void;
}

export function TabBar({ tabs, active, onChange }: Props) {
  return (
    <div style={{
      display: 'flex',
      borderBottom: '2px solid #e5e7eb',
      marginBottom: 20,
      gap: 0,
    }}>
      {tabs.map(t => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          style={{
            border: 'none',
            borderBottom: active === t.id ? '2px solid #3b82d4' : '2px solid transparent',
            background: 'transparent',
            padding: '9px 18px',
            cursor: 'pointer',
            fontSize: 13,
            fontWeight: active === t.id ? 600 : 400,
            color: active === t.id ? '#3b82d4' : '#6b7280',
            marginBottom: -2,
            transition: 'color 0.15s',
          }}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}
