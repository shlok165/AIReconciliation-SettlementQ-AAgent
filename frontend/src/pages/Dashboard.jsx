import { useMemo, useRef } from 'react'
import { LoaderCircle } from 'lucide-react'
import MetricCard from '../components/MetricCard'
import MatchChart from '../components/MatchChart'

const pct = (value) => `${Number(value || 0).toFixed(2)}%`
const num = (value) => Number(value || 0)

export default function Dashboard({ metrics, unresolved, datasetStatus, datasetSize, setDatasetSize, working, onGenerate }) {
  const attentionRef = useRef(null)

  const stages = useMemo(
    () => metrics ? [
      { stage: 'Deterministic', count: metrics.evaluation.transaction_resolution_stage_breakdown?.deterministic_resolved_transactions || 0 },
      { stage: 'Fuzzy', count: metrics.evaluation.transaction_resolution_stage_breakdown?.fuzzy_resolved_transactions || 0 },
      { stage: 'LLM', count: metrics.evaluation.transaction_resolution_stage_breakdown?.llm_resolved_transactions || 0 },
      { stage: 'Review', count: metrics.evaluation.transaction_resolution_stage_breakdown?.review_transactions || 0 },
      { stage: 'Incorrect', count: metrics.evaluation.transaction_resolution_stage_breakdown?.incorrect_transactions || 0 },
    ] : [],
    [metrics],
  )

  const attentionCases = useMemo(() => {
    if (!unresolved) return []
    const cases = []
    for (const c of unresolved.llm_cases) {
      const llm = c.llm_decision
      if (c.exception_type === 'MANUAL_REVIEW_REQUIRED') {
        let needsReview = true
        if (llm) {
          if (llm.llm_resolution === 'MATCH' && llm.llm_confidence >= 90) needsReview = false
          if (llm.llm_resolution === 'EXCEPTION' && llm.llm_confidence >= 90) needsReview = false
        }
        if (needsReview) cases.push({ ...c, _status: 'Review' })
      } else if (c.exception_type === 'NO_MATCH_FOUND') {
        cases.push({ ...c, _status: 'Unresolved' })
      }
    }
    return cases
  }, [unresolved])

  const attentionCount = attentionCases.length || num(metrics.evaluation.needs_attention_transactions)

  const scrollToAttention = () => {
    attentionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

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
          value={attentionCount.toLocaleString()}
          note="review + unresolved transactions — click to view"
          tone="orange"
          onClick={scrollToAttention}
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

      {attentionCases.length > 0 && (
        <div className="panel" ref={attentionRef} id="needs-attention">
          <div className="panel-head">
            <h2>Needs attention ({attentionCases.length})</h2>
            <p>Review cases and unresolved transactions requiring human validation</p>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Status</th>
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
                {attentionCases.map((c) => {
                  const llm = c.llm_decision
                  return (
                    <tr key={c.case_id}>
                      <td><span className={`attention-badge ${c._status.toLowerCase()}`}>{c._status}</span></td>
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
