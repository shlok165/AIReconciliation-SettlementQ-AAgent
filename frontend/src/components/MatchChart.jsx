import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

const STAGE_COLORS = {
  Deterministic: '#3b82f6',
  Fuzzy: '#8b5cf6',
  LLM: '#06b6d4',
  Review: '#f59e0b',
  Unresolved: '#ef4444',
  Incorrect: '#dc2626',
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <span className="chart-tooltip-label">{label}</span>
      <span className="chart-tooltip-value">{payload[0].value} transactions</span>
    </div>
  )
}

export default function MatchChart({ stages }) {
  const data = stages.map(s => ({
    ...s,
    fill: STAGE_COLORS[s.stage] || '#6b7280',
  }))

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={data} barCategoryGap="20%">
          <CartesianGrid stroke="#edf2fb" vertical={false} strokeDasharray="3 3" />
          <XAxis
            dataKey="stage"
            axisLine={false}
            tickLine={false}
            tick={{ fontSize: 12, fill: '#6b7280' }}
          />
          <YAxis
            axisLine={false}
            tickLine={false}
            tick={{ fontSize: 12, fill: '#6b7280' }}
            allowDecimals={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(59,130,246,0.06)' }} />
          <Bar dataKey="count" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
