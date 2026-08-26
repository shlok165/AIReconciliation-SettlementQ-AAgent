import { Bot } from 'lucide-react'

export default function ChatBox({ messages, question, setQuestion, working, requestStatus, dataset, onSubmit }) {
  return (
    <div className="chat-panel">
      <div className="chat-title">
        <div className="chat-title-icon"><Bot size={16} /></div>
        <div>
          <h2>Settlement QA</h2>
          <p>Grounded answers from the current reconciliation state</p>
        </div>
      </div>

      {dataset && (
        <div className="chat-dataset-badge">
          Dataset loaded: {dataset.invoice_count} invoices · {dataset.payment_count} payments · {dataset.bank_transaction_count} bank transactions
        </div>
      )}

      <div className="chat-log">
        {messages.map((msg, idx) => (
          <div key={`${msg.role}-${idx}`} className={`message ${msg.role}`}>
            <div className={`bubble ${msg.failed ? 'failed' : ''}`}>
              {msg.text}
              {msg.trace && (
                <details className="trace-details">
                  <summary>Tool calls ({msg.trace.length})</summary>
                  <pre>{JSON.stringify(msg.trace, null, 2)}</pre>
                </details>
              )}
            </div>
          </div>
        ))}
      </div>

      <form className="chat-input" onSubmit={onSubmit}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a ledger question…"
          disabled={working}
        />
        <button type="submit" disabled={working || !question.trim()}>
          {working ? 'Sending…' : 'Send'}
        </button>
      </form>
      {requestStatus && <p className="request-status" aria-live="polite">{requestStatus}</p>}
    </div>
  )
}
