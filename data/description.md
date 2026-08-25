# Extreme Financial Reconciliation Dataset

This dataset contains synthetic financial records for rigorously evaluating a multi-source finance reconciliation engine and AI tie-breaker agent across real-world enterprise ledger challenges.

## Raw Data

The `raw/` directory contains the input data available to the agent:

- **`invoices.csv`** — Expected customer payments, invoice dates, customer references, and line-item descriptions.
- **`bank_transactions.csv`** — Bank settlement transactions, cleared dates, amounts, reference numbers, and unparsed deposit memos.
- **`payments.csv`** — Payment gateway records, gross amounts, processing fees, net settled amounts, and settlement dates.

The raw data does **not** contain reconciliation outcomes or labels. The engine must determine whether records should be successfully reconciled, held for fuzzy scoring, or flagged as exceptions.

---

## Dataset Breakdown & Complexity Categories

| Category | Description | Primary Reconciliation Mechanism | Cases | Invoices | Payments | Bank Txns |
| :--- | :--- | :--- | :---:| :---:| :---:| :---:|
| **Clean Exact Matches** | Fully populated `linked_invoice_id`, same-day clearance, exact amounts. | Deterministic Matching (Rule-based) | 25 | 25 | 25 | 25 |
| **Unstructured Reference Memos** | `linked_invoice_id` is omitted; references embedded in free-text memos. | Fuzzy / Lexical Entity Extraction | 20 | 20 | 20 | 20 |
| **OCR Noise & Name Typos** | Typos, character confusions (`0` vs `O`, `1` vs `I`), abbreviation variations. | Multi-Signal RapidFuzz Scoring | 15 | 15 | 15 | 15 |
| **Gateway Fees & Net Settlement** | Interchange deductions (2.9% + $0.30) where bank amount equals net. | Fee Invariant Reconciliation | 10 | 10 | 10 | 10 |
| **Banking Delays & Cutoffs** | 2–6 day settlement delays across weekends and holiday cutoff windows. | Date Window Tolerances | 10 | 10 | 10 | 10 |
| **Partial Payment Installments** | Single invoice settled across multiple separate payment installments. | Cumulative Amount Aggregation | 5 | 5 | 10 | 10 |
| **AI Tie-Breaker Ambiguity** | Competing invoices with identical amounts/dates to the same vendor. | LLM / Semantic Context Tie-Breaker | 5 | 10 | 5 | 5 |
| **Complex Genuine Exceptions** | Orphans, severe date lags (>90 days), severe underpayments/shortfalls. | Exception & Anomaly Detection | 10 | 6 | 6 | 6 |
| **Total** | **Extreme Benchmark Suite** | | **100** | **101** | **101** | **101** |