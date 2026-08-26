export default function ExceptionTable({ rows }) {
  if (!rows.length) return <p className="empty">No exceptions in this view.</p>
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Record</th>
            <th>Type</th>
            <th>Reason</th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${row.record_id}-${index}`}>
              <td><code>{row.record_id}</code></td>
              <td>{row.record_type.replace('_', ' ')}</td>
              <td><span className="reason">{row.exception_type.replaceAll('_', ' ')}</span></td>
              <td>{row.description}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
