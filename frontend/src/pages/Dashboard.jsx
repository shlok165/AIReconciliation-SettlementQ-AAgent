import { useMemo } from 'react'
import { LoaderCircle } from 'lucide-react'
import MetricCard from '../components/MetricCard'
import MatchChart from '../components/MatchChart'

const pct = (value) => `${Number(value || 0).toFixed(2)}%`
const num = (value) => Number(value || 0)

export default function Dashboard({ metrics, unresolved, datasetStatus, datasetSize, setDatasetSize, working, onGenerate }) {
  const stages = useMemo(
    () => metrics ? [
      { stage: 'Deterministic', count: metrics.evaluation.transaction_resolution_stage_breakdown?.deterministic_resolved_transactions || 0 },
      { stage: 'Fuzzy', count: metrics.evaluation.transaction_resolution_stage_breakdown?.fuzzy_resolved_transactions || 0 },
      { stage: 'LLM', count: metrics.evaluation.transaction_resolution_stage_breakdown?.llm_resolved_transactions || 0 },
      { stage: 'Exception', count: metrics.evaluation.transaction_resolution_stage_breakdown?.exception_resolved_transactions || 0 },
      { stage: 'Review', count: metrics.evaluation.transaction_resolution_stage_breakdown?.review_transactions || 0 },
      { stage: 'Unresolved', count: metrics.evaluation.transaction_resolution_stage_breakdown?.unresolved_transactions || 0 },
      { stage: 'Incorrect', count: metrics.evaluation.transaction_resolution_stage_breakdown?.incorrect_transactions || 0 },
    ] : [],
    [metrics],
  )

  const reviewCases = useMemo(() => {
    if (!unresolved) return []
    return unresolved.llm_cases.filter(c => {
      if (c.exception_type !== 'MANUAL_REVIEW_REQUIRED') return false
      const llm = c.llm_decision
      if (!llm) return true
      if (llm.llm_resolution === 'MATCH' && llm.llm_confidence >= 90) return false
      if (llm.llm_resolution === 'EXCEPTION' && llm.llm_confidence >= 90) return false
      return true
    })
  }, [unresolved])

  if (!metrics) {
    return <div className="empty-panel">Unable to load dataset metrics.</div>
  }

  return (
    <>
      <div className="metrics">
        <MetricCard
          label="Match accuracy"
          value={pct(metrics.evaluation.transaction_resolution_accuracy)}
          note="transactions correctly handled against ground truth"
          tone="green"
        />
        <MetricCard
          label="Transactions resolved"
          value={`${num(metrics.evaluation.correctly_resolved_transactions)} / ${num(metrics.evaluation.total_transactions)}`}
          note="fully resolved correctly"
        />
        <MetricCard
          label="Needs attention"
          value={num(metrics.evaluation.needs_attention_transactions).toLocaleString()}
          note="review + unresolved transactions"
          tone="orange"
        />
        <MetricCard
          label="False resolutions"
          value={num(metrics.evaluation.incorrectly_resolved_transactions).toLocaleString()}
          note="transactions resolved incorrectly"
          tone="purple"
        />
      </div>

      <div className="panel-grid">
        <section className="panel">
          <div className="panel-head">
            <h2>Resolution pipeline</h2>
            <p>Transaction-level stage outcomes across the full dataset</p>
          </div>
          <MatchChart stages={stages} />
        </section>

        <section className="panel">
          <div className="panel-head">
            <h2>Run controls</h2>
            <p>Generate data and run the full pipeline automatically</p>
          </div>
          <div className="generator-box">
            <label htmlFor="dataset-size">Dataset size</label>
            <div className="input-row">
              <input
                id="dataset-size"
                type="number"
                min="10"
                max="5000"
                value={datasetSize}
                onChange={(e) => setDatasetSize(e.target.value)}
              />
              <button className="primary" onClick={onGenerate} disabled={working}>
                {working ? 'Running…' : 'Generate data'}
              </button>
            </div>
            {datasetStatus && <p className="status-text">{datasetStatus}</p>}
          </div>
        </section>
      </div>

      {unresolved && unresolved.llm_cases.length > 0 && (
        <div className="panel">
          <div className="panel-head">
            <h2>LLM verdict on unresolved transactions ({unresolved.llm_cases.length} cases)</h2>
            <p>Each unmatched record evaluated by the LLM with resolution decision</p>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Case</th>
                  <th>Primary</th>
                  <th>Related Records</th>
                  <th>Engine Type</th>
                  <th>GT Expected</th>
                  <th>GT Category</th>
                  <th>LLM Verdict</th>
                  <th>LLM Matched</th>
                  <th>Confidence</th>
                  <th>LLM Reason</th>
                </tr>
              </thead>
              <tbody>
                {unresolved.llm_cases.map((c) => {
                  const llm = c.llm_decision
                  return (
                    <tr key={c.case_id}>
                      <td><code>{c.case_id}</code></td>
                      <td><code>{c.record_id}</code> <small>({c.record_type})</small></td>
                      <td>{c.related_ids.length ? c.related_ids.map(id => <code key={id} className="inline-id">{id}</code>) : <span className="muted">none</span>}</td>
                      <td><span className="reason">{c.exception_type.replaceAll('_', ' ')}</span></td>
                      <td>{c.ground_truth ? <strong className={c.ground_truth.expected_result === 'MATCH' ? 'text-green' : 'text-red'}>{c.ground_truth.expected_result}</strong> : <span className="muted">—</span>}</td>
                      <td><small>{c.ground_truth?.category || '—'}</small></td>
                      <td>
                        {llm ? (
                          <strong className={llm.llm_resolution === 'MATCH' ? 'text-green' : 'text-blue'}>
                            {llm.llm_resolution}
                          </strong>
                        ) : <span className="muted">pending</span>}
                      </td>
                      <td>
                        {llm?.llm_matched_ids?.length ? llm.llm_matched_ids.map(id => <code key={id} className="inline-id">{id}</code>) : <span className="muted">—</span>}
                      </td>
                      <td>{llm ? <strong>{num(llm.llm_confidence)}%</strong> : '—'}</td>
                      <td><small className="llm-reason">{llm?.llm_justification || '—'}</small></td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {reviewCases.length > 0 && (
        <div className="panel">
          <div className="panel-head">
            <h2>Cases needing review ({reviewCases.length})</h2>
            <p>Transactions flagged by the engine that require human validation</p>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Case</th>
                  <th>Primary</th>
                  <th>Related Records</th>
                  <th>GT Expected</th>
                  <th>GT Category</th>
                  <th>LLM Verdict</th>
                  <th>LLM Matched</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {reviewCases.map((c) => {
                  const llm = c.llm_decision
                  return (
                    <tr key={c.case_id}>
                      <td><code>{c.case_id}</code></td>
                      <td><code>{c.record_id}</code> <small>({c.record_type})</small></td>
                      <td>{c.related_ids.length ? c.related_ids.map(id => <code key={id} className="inline-id">{id}</code>) : <span className="muted">none</span>}</td>
                      <td>{c.ground_truth ? <strong className={c.ground_truth.expected_result === 'MATCH' ? 'text-green' : 'text-red'}>{c.ground_truth.expected_result}</strong> : '—'}</td>
                      <td><small>{c.ground_truth?.category || '—'}</small></td>
                      <td>
                        {llm ? (
                          <strong className={llm.llm_resolution === 'MATCH' ? 'text-green' : 'text-blue'}>
                            {llm.llm_resolution}
                          </strong>
                        ) : '—'}
                      </td>
                      <td>
                        {llm?.llm_matched_ids?.length ? llm.llm_matched_ids.map(id => <code key={id} className="inline-id">{id}</code>) : '—'}
                      </td>
                      <td>{llm ? <strong>{num(llm.llm_confidence)}%</strong> : '—'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {unresolved && unresolved.matched.length > 0 && (
        <div className="panel">
          <div className="panel-head">
            <h2>Matched relationships ({unresolved.matched.length})</h2>
            <p>All resolved payment-invoice and payment-bank_txn links</p>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Left</th>
                  <th>Right</th>
                  <th>Type</th>
                  <th>Stage</th>
                  <th>GT Expected</th>
                  <th>GT Category</th>
                </tr>
              </thead>
              <tbody>
                {unresolved.matched.map((m, i) => (
                  <tr key={`${m.left_id}-${m.right_id}-${i}`}>
                    <td><code>{m.left_id}</code></td>
                    <td><code>{m.right_id}</code></td>
                    <td><span className="reason">{m.type}</span></td>
                    <td>{m.stage}</td>
                    <td>{m.ground_truth ? <strong className={m.ground_truth.expected_result === 'MATCH' ? 'text-green' : 'text-red'}>{m.ground_truth.expected_result}</strong> : '—'}</td>
                    <td><small>{m.ground_truth?.category || '—'}</small></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  )
}
