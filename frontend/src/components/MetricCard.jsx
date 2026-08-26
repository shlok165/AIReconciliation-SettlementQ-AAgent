export default function MetricCard({ label, value, note, tone = 'blue' }) {
  return (
    <article className={`metric-card ${tone}`}>
      <span className="metric-label">{label}</span>
      <strong className="metric-value">{value}</strong>
      <small className="metric-note">{note}</small>
    </article>
  )
}
