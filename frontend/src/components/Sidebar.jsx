import {
  BarChart3,
  CircleAlert,
  Database,
  Info,
  MessageSquare,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'

const NAV_ITEMS = [
  { id: 'overview', label: 'Overview', icon: BarChart3 },
  { id: 'dataset', label: 'Dataset', icon: Database },
  { id: 'exceptions', label: 'Exceptions', icon: CircleAlert },
  { id: 'assistant', label: 'Ask assistant', icon: MessageSquare },
  { id: 'about', label: 'About', icon: Info },
]

export default function Sidebar({ view, setView, exceptionCount }) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-icon"><Sparkles size={18} /></div>
        <div>
          <b>Settlement AI</b>
          <small>Reconciliation workspace</small>
        </div>
      </div>

      <nav>
        {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            className={view === id ? 'active' : ''}
            onClick={() => setView(id)}
          >
            <Icon size={18} />
            {label}
            {id === 'exceptions' && exceptionCount > 0 && (
              <em>{exceptionCount}</em>
            )}
          </button>
        ))}
      </nav>

      <div className="side-note">
        <ShieldCheck size={18} />
        <span>Grounded answers<br />from your ledger data</span>
      </div>
    </aside>
  )
}
