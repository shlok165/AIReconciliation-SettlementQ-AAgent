export default function About() {
  return (
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
          <span>Example: memo "inv 1042" and a one-day date difference identify invoice <code>INV-1042</code> despite a missing link ID.</span>
        </article>
        <article className="resolution-item">
          <h3>LLM</h3>
          <p>The language model evaluates unmatched transactions and breaks genuine ties between candidates.</p>
          <span>Example: OCR noise or unstructured memos confuse deterministic rules, but the LLM correctly identifies the match.</span>
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
  )
}
