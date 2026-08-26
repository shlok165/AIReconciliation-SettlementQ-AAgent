import { useState } from 'react'

export default function DataTable({ title, subtitle, rows, columns }) {
  const [page, setPage] = useState(0)
  const pageSize = 25
  const totalPages = Math.ceil(rows.length / pageSize)
  const pageRows = rows.slice(page * pageSize, (page + 1) * pageSize)

  return (
    <div className="panel data-table-panel">
      <div className="panel-head">
        <h2>{title} ({rows.length})</h2>
        <p>{subtitle}</p>
      </div>
      {rows.length === 0 ? (
        <p className="empty">No data available.</p>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                {columns.map(col => <th key={col.key}>{col.label}</th>)}
              </tr>
            </thead>
            <tbody>
              {pageRows.map((row, idx) => (
                <tr key={idx}>
                  {columns.map(col => (
                    <td key={col.key}>
                      {col.render ? col.render(row[col.key], row) : (row[col.key] ?? '—')}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {totalPages > 1 && (
            <div className="pagination">
              <button disabled={page === 0} onClick={() => setPage(p => p - 1)}>Prev</button>
              <span className="pagination-info">{page + 1} / {totalPages}</span>
              <button disabled={page >= totalPages - 1} onClick={() => setPage(p => p + 1)}>Next</button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
