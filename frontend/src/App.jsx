import { useEffect, useMemo, useState } from 'react'
import {
  BarChart3,
  Bot,
  CircleAlert,
  Info,
  LoaderCircle,
  MessageSquare,
  RefreshCw,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import {
  askQuestion,
  generateDataset,
  generateReport,
  getExceptions,
  getMetrics,
  runReconciliation,
} from './services/api'
import './App.css'

const pct = (value) => `${Number(value || 0).toFixed(2)}%`
const num = (value) => Number(value || 0)

function Card({ label, value, note, tone = 'blue' }) {
  return (
    <article className={`metric-card ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{note}</small>
    </article>
  )
}

function Table({ rows }) {
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

export default function App() {
  const [view, setView] = useState('overview')
  const [metrics, setMetrics] = useState(null)
  const [exceptions, setExceptions] = useState([])
  const [loading, setLoading] = useState(true)
  const [working, setWorking] = useState(false)
  const [error, setError] = useState('')
  const [question, setQuestion] = useState('')
  const [requestStatus, setRequestStatus] = useState('')
  const [datasetSize, setDatasetSize] = useState(100)
  const [datasetStatus, setDatasetStatus] = useState('')
  const [messages, setMessages] = useState([
    { role: 'assistant', text: 'Ask about a payment, invoice, bank transaction, gateway fee, exception, or reconciliation metric.' },
  ])

  const refresh = async () => {
    setLoading(true)
    setError('')
    try {
      const [m, e] = await Promise.all([getMetrics(), getExceptions()])
      setMetrics(m)
      setExceptions(e)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Could not connect to FastAPI. Start it on port 8000 and try again.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refresh()
  }, [])

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

  const submit = async (event) => {
    event.preventDefault()
    const text = question.trim()
    if (!text || working) return

    setMessages((items) => [...items, { role: 'user', text }])
    setQuestion('')
    setWorking(true)
    setRequestStatus('Sending request to the Settlement Q&A agent...')

    try {
      const response = await askQuestion(text)
      setMessages((items) => [...items, { role: 'assistant', text: response.answer, trace: response.tool_trace }])
      const toolCount = response.tool_trace?.length || 0
      setRequestStatus(`Request completed${toolCount ? ` with ${toolCount} tool call${toolCount === 1 ? '' : 's'}` : ''}.`)
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || 'The assistant could not complete that request.'
      setMessages((items) => [...items, { role: 'assistant', text: detail, failed: true }])
      setRequestStatus('Request failed. See the assistant response for details.')
    } finally {
      setWorking(false)
    }
  }

  const generateData = async () => {
    setWorking(true)
    setError('')
    setDatasetStatus('Generating dataset, running reconciliation, and exporting report...')
    try {
      const result = await generateDataset(Number(datasetSize))
      await runReconciliation()
      await generateReport()
      setDatasetStatus(`Generated ${result.total_records} records from a requested size of ${result.size_requested}. Reconciliation and report completed.`)
      await refresh()
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Could not generate the dataset.')
      setDatasetStatus('')
    } finally {
      setWorking(false)
    }
  }

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <i><Sparkles size={18} /></i>
          <div>
            <b>Settlement AI</b>
            <small>Reconciliation workspace</small>
          </div>
        </div>

        <nav>
          <button className={view === 'overview' ? 'active' : ''} onClick={() => setView('overview')}><BarChart3 size={18} />Overview</button>
          <button className={view === 'exceptions' ? 'active' : ''} onClick={() => setView('exceptions')}><CircleAlert size={18} />Exceptions <em>{exceptions.length}</em></button>
          <button className={view === 'assistant' ? 'active' : ''} onClick={() => setView('assistant')}><MessageSquare size={18} />Ask assistant</button>
          <button className={view === 'about' ? 'active' : ''} onClick={() => setView('about')}><Info size={18} />About</button>
        </nav>

        <div className="side-note">
          <ShieldCheck size={18} />
          <span>Grounded answers<br />from your ledger data</span>
        </div>
      </aside>

      <section className="content">
        <header>
          <div>
            <p className="eyebrow">AI RECONCILIATION ENGINE</p>
            <h1>{view === 'overview' ? 'Settlement overview' : view === 'exceptions' ? 'Exception review' : view === 'about' ? 'About the generator' : 'Settlement Q&A'}</h1>
            <p className="subtitle">Current local dataset · deterministic-first workflow</p>
          </div>
          <button className="refresh" onClick={refresh}><RefreshCw size={18} className={loading ? 'spin' : ''} /></button>
        </header>

        {error && <div className="error"><CircleAlert size={18} />{error}</div>}

        {loading ? (
          <div className="loading"><LoaderCircle className="spin" size={26} /> Loading current reconciliation results…</div>
        ) : (
          <>
            {view === 'overview' && metrics && (
              <>
                <div className="metrics">
                  <Card label="Match accuracy" value={pct(metrics.evaluation.transaction_resolution_accuracy)} note="transactions correctly handled against ground truth" tone="green" />
                  <Card label="Transactions resolved" value={`${num(metrics.evaluation.correctly_resolved_transactions)} / ${num(metrics.evaluation.total_transactions)}`} note="fully resolved correctly" />
                  <Card label="Needs attention" value={num(metrics.evaluation.needs_attention_transactions).toLocaleString()} note="review + unresolved transactions" tone="orange" />
                  <Card label="False resolutions" value={num(metrics.evaluation.incorrectly_resolved_transactions).toLocaleString()} note="transactions resolved incorrectly" tone="purple" />
                </div>

                <div className="panel-grid">
                  <section className="panel">
                    <div className="panel-head">
                      <h2>Resolution pipeline</h2>
                      <p>Transaction-level stage outcomes: deterministic, fuzzy, LLM, exception-handled, review, unresolved, and incorrect</p>
                    </div>

                    <div className="chart-wrap">
                      <ResponsiveContainer width="100%" height={220}>
                        <BarChart data={stages}>
                          <CartesianGrid stroke="#edf2fb" vertical={false} />
                          <XAxis dataKey="stage" />
                          <YAxis />
                          <Tooltip />
                          <Bar dataKey="count" radius={[6, 6, 0, 0]} fill="#4477ef" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </section>

                  <section className="panel">
                    <div className="panel-head">
                      <h2>Run controls</h2>
                      <p>Generate data and run the full pipeline automatically</p>
                    </div>

                    <div className="generator-box">
                      <label htmlFor="dataset-size">Dataset size</label>
                      <div className="input-row">
                        <input id="dataset-size" type="number" min="10" max="5000" value={datasetSize} onChange={(e) => setDatasetSize(e.target.value)} />
                        <button className="primary" onClick={generateData} disabled={working}>Generate data</button>
                      </div>
                      {datasetStatus && <p className="status-text">{datasetStatus}</p>}
                    </div>

                    <div className="generator-box secondary-metrics">
                      <label>Relationship-level diagnostics</label>
                      <p>
                        Precision {pct(metrics.evaluation.precision)} · Coverage {pct(metrics.evaluation.coverage)} ·
                        Resolved relationships {num(metrics.evaluation.resolved_match_relationships).toLocaleString()} /
                        {num(metrics.evaluation.expected_match_relationships).toLocaleString()}
                      </p>
                      <p>
                        Relationship pipeline: Deterministic {num(metrics.reconciliation.deterministic_confirmed_matches).toLocaleString()} ·
                        Fuzzy {num(metrics.reconciliation.fuzzy_auto_matches).toLocaleString()} ·
                        Review {num(metrics.reconciliation.manual_review_candidates).toLocaleString()} ·
                        Rejected {num(metrics.reconciliation.rejected_fuzzy_candidates).toLocaleString()}
                      </p>
                      <p>
                        Deterministic correct {num(metrics.evaluation.identification_matrix?.deterministic?.correct_relationships).toLocaleString()} ·
                        Fuzzy correct {num(metrics.evaluation.identification_matrix?.fuzzy?.correct_relationships).toLocaleString()} ·
                        LLM correct {num(metrics.evaluation.identification_matrix?.llm?.correct_relationships).toLocaleString()}
                      </p>
                    </div>
                  </section>
                </div>
              </>
            )}

            {view === 'exceptions' && (
              <div className="panel">
                <div className="panel-head">
                  <h2>Exception log</h2>
                  <p>Records that need validation or follow-up</p>
                </div>
                <Table rows={exceptions} />
              </div>
            )}

            {view === 'assistant' && (
              <div className="chat-panel">
                <div className="chat-title">
                  <i><Bot size={16} /></i>
                  <div>
                    <h2>Settlement QA</h2>
                    <p>Grounded answers from the current reconciliation state</p>
                  </div>
                </div>

                <div className="chat-log">
                  {messages.map((msg, idx) => (
                    <div key={`${msg.role}-${idx}`} className={`message ${msg.role}`}>
                      <div className="bubble">
                        {msg.text}
                        {msg.trace && <pre>{JSON.stringify(msg.trace, null, 2)}</pre>}
                      </div>
                    </div>
                  ))}
                </div>

                <form className="chat-input" onSubmit={submit}>
                  <input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask a ledger question…" />
                  <button type="submit" disabled={working}>{working ? 'Sending…' : 'Send'}</button>
                </form>
                {requestStatus && <p className="request-status" aria-live="polite">{requestStatus}</p>}
              </div>
            )}

            {view === 'about' && (
              <div className="panel about-panel">
                <div className="panel-head">
                  <h2>How transaction resolution works</h2>
                  <p>What each pipeline outcome means</p>
                </div>

                <div className="resolution-guide">
                  <article className="resolution-item">
                    <h3>Deterministic</h3>
                    <p>An exact, rule-based match passes all required checks, such as IDs, amount, date window, and settlement invariants.</p>
                    <span>Example: payment <code>PAY-1042</code> explicitly links to invoice <code>INV-1042</code>, and the amount and date are valid.</span>
                  </article>
                  <article className="resolution-item">
                    <h3>Fuzzy</h3>
                    <p>A likely match is selected from imperfect data using text similarity, amount proximity, and date proximity.</p>
                    <span>Example: memo “inv 1042” and a one-day date difference identify invoice <code>INV-1042</code> despite a missing link ID.</span>
                  </article>
                  <article className="resolution-item">
                    <h3>LLM</h3>
                    <p>The language model is used only to break a genuine tie between candidates that already passed the matching safeguards.</p>
                    <span>Example: two invoices have the same amount and date, so the memo context selects the candidate mentioning the correct order.</span>
                  </article>
                  <article className="resolution-item">
                    <h3>Exception resolved</h3>
                    <p>The transaction is correctly recognized as a genuine exception instead of being forced into a match.</p>
                    <span>Example: an orphan bank transaction has no corresponding payment, so it is correctly recorded as an exception.</span>
                  </article>
                  <article className="resolution-item">
                    <h3>Review</h3>
                    <p>The evidence is plausible but ambiguous or conflicting, so a person must validate it before it is accepted.</p>
                    <span>Example: one payment scores highly against two invoices with nearly identical details.</span>
                  </article>
                  <article className="resolution-item">
                    <h3>Unresolved</h3>
                    <p>No acceptable match was found, and the record was not classified as a known exception.</p>
                    <span>Example: a payment is missing both a usable reference and a bank transaction within the allowed date and amount window.</span>
                  </article>
                  <article className="resolution-item">
                    <h3>Incorrect</h3>
                    <p>A match was accepted, but it disagrees with the expected relationship in the ground-truth data.</p>
                    <span>Example: payment <code>PAY-1042</code> is linked to <code>INV-1043</code> when the expected invoice is <code>INV-1042</code>.</span>
                  </article>
                </div>

                <div className="about-divider" />
                <div className="panel-head">
                  <h2>Data generation rules</h2>
                  <p>How the synthetic ledger is built</p>
                </div>

                <ul className="rules-list">
                  <li><strong>Clean matches:</strong> exact invoice, payment, and bank references with matching amounts and dates.</li>
                  <li><strong>Unstructured memos:</strong> missing linked invoice IDs and text references embedded in descriptions or notes.</li>
                  <li><strong>OCR noise:</strong> typos, swapped characters, abbreviations, and reference corruption.</li>
                  <li><strong>Gateway fees:</strong> net settled amounts differ from gross values due to interchange or card-processing fees.</li>
                  <li><strong>Banking delay:</strong> settlement dates and bank posting dates vary across business days and weekends.</li>
                  <li><strong>Partial payments:</strong> one invoice may be settled in multiple installment payments.</li>
                  <li><strong>AI ambiguity:</strong> close semantic matches with similar amounts and dates require tie-breaking.</li>
                  <li><strong>Exceptions:</strong> orphan records, severe date lag, and short-paid amounts remain unresolved by design.</li>
                </ul>
              </div>
            )}

            {!metrics && !loading && <div className="empty-panel">Unable to load dataset metrics.</div>}
          </>
        )}
      </section>
    </main>
  )
}
