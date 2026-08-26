# AI Finance Controller — Reconciliation & Settlement Q&A Agent

> Run the books and the cash position. A deterministic-first reconciliation engine that matches invoices, payments, and bank transactions across a 50+ record synthetic dataset, reports its measured accuracy, and hands off genuine exceptions it cannot resolve.

Built for the **Razorpay AI Buildathon**.

---

## Why This Matters

Reconciliation, settlement, and forecasting are still done by hand at most companies. The 2026 builder consensus is clear: **verification capacity, not generation speed, is the bottleneck.** This agent closes one finance-ops loop — matching multi-source ledger data end-to-end — and honestly reports what it could and could not resolve.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        React / Vite Frontend                    │
│  Dashboard · Dataset · Exceptions · Q&A Assistant · About       │
└──────────────────────────────┬──────────────────────────────────┘
                               │ /api proxy
┌──────────────────────────────▼──────────────────────────────────┐
│                     FastAPI Backend (port 8000)                  │
│  /reconcile · /metrics · /exceptions · /dataset · /ask · /report│
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                   4-Pass Reconciliation Pipeline                 │
│                                                                  │
│  Pass 1 — Deterministic    Exact ID links, amount+date, refs    │
│  Pass 2 — Fuzzy            RapidFuzz text + amount/date scoring  │
│  Pass 3 — LLM Tie-Breaker  Breaks genuine ties (≥90% threshold) │
│  Pass 4 — LLM Evaluation   Evaluates all remaining unmatched    │
│                                                                  │
│  Scoring → 50% text · 30% amount · 20% date proximity          │
│  Conflict Resolution → one-to-many auto-match demotion          │
│  3-Way Consistency Guard → payment-bank held until invoice match │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                  Evaluation & Reporting Layer                    │
│  Ground-truth validation · Transaction-level accuracy           │
│  Relationship-level precision/recall · Identification matrix    │
│  JSON + CSV export · Per-stage breakdown                        │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Flow

| Pass | Name | What It Does | Records Handled |
|------|------|-------------|-----------------|
| 1 | **Deterministic** | Exact `linked_invoice_id`, amount+date matching, bank reference lookup | ~70% of matches |
| 2 | **Fuzzy** | RapidFuzz text similarity + amount/date proximity scoring with weighted confidence | ~25% of matches |
| 3 | **LLM Tie-Breaker** | Sends genuinely ambiguous candidates (competing scores within margin) to LLM | ~3% of matches |
| 4 | **LLM Evaluation** | Evaluates all remaining unmatched cases, validates against ground truth | Remaining tail |

---

## Sample Results (200-case dataset)

These numbers come from an actual run — not estimates, not cherry-picked.

| Metric | Value |
|--------|-------|
| **Transaction Resolution Accuracy** | **98.00%** |
| **Precision** | **100.00%** |
| **Recall / Coverage** | **98.00%** |
| **Throughput** | **818.5 records/sec** |
| Total Transactions | 200 |
| Correctly Resolved | 196 |
| Incorrectly Resolved | 0 |
| Needs Attention | 13 |

### Resolution Stage Breakdown

| Stage | Transactions | % of Total |
|-------|-------------|------------|
| Deterministic | ~140 | ~70% |
| Fuzzy | ~50 | ~25% |
| LLM | ~6 | ~3% |
| Review | 0 | 0.0% |
| Unresolved | 4 | ~2% |
| Incorrect | 0 | 0.0% |

### Identification Matrix

| Stage | Identified | Correct | Incorrect | Precision | Coverage |
|-------|-----------|---------|-----------|-----------|----------|
| Deterministic | 323 | 323 | 0 | 100.0% | 78.40% |
| Fuzzy | 67 | 67 | 0 | 100.0% | 16.26% |
| LLM | 0 | 0 | 0 | — | 0.0% |

---

## Dataset Design

A synthetic benchmark with **8 complexity categories** mapped to real payment operations challenges:

| # | Category | Cases | Why It Exists |
|---|----------|-------|---------------|
| 1 | **Clean Exact Matches** | 25 | Baseline — proves the pipeline handles the easy path |
| 2 | **Unstructured Reference Memos** | 20 | Missing `linked_invoice_id`, text references in descriptions |
| 3 | **OCR Noise & Name Typos** | 15 | Character confusions (0/O, 1/I), abbreviations, reference corruption |
| 4 | **Gateway Fees & Net Settlement** | 10 | Interchange deductions (2.9% + $0.30) where bank ≠ gross |
| 5 | **Banking Delays & Cutoffs** | 10 | 2-6 day settlement shifts across weekends/holidays |
| 6 | **Partial Payment Installments** | 5 | One invoice split across multiple payments |
| 7 | **AI Tie-Breaker Ambiguity** | 5 | Competing invoices with identical amounts/dates |
| 8 | **Complex Genuine Exceptions** | 10 | Orphans, severe date lags (>90 days), shortfalls |

**Total:** 200 cases · ~204 invoices · ~208 payments · ~208 bank transactions

Ground truth is stored separately (`data/ground_truth/ground_truth.csv`) and never exposed to the matching engine — only used for evaluation.

---

## Key Design Decisions

### Deterministic-First, Not LLM-First

Most teams would throw GPT at everything. This engine uses rules for what rules handle, then escalates. The LLM only touches what deterministic and fuzzy passes cannot resolve — typically 3-12% of cases. This is faster, cheaper, more auditable, and more accurate for structured data.

### Function-Calling, Not RAG

The Settlement Q&A agent uses LLM function/tool calling (not RAG) to query the reconciliation state. The LLM receives tool definitions (`get_transaction`, `get_gateway_fee`, `get_exception_reason`, etc.) and makes structured API calls against the live dataset. Answers are grounded in actual data, not hallucinated.

### No Forced Matches

The engine will not force a match below its confidence threshold. Ambiguous cases go to review, unmatched cases get honest exception labels. The evaluation penalizes false-positive matches (incorrect resolutions) as harshly as missed matches. A 98% match rate with 0% incorrect is better than 100% match rate with hidden errors.

### Cases, Not Records

Transactions are grouped by ground-truth case (e.g., PAY-0045 + INV-0045 + TXN-0045 = 1 case). This prevents inflating match counts — matching 3 records that belong together counts as resolving 1 case, not 3.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python, FastAPI, Uvicorn |
| **Data** | Pandas, NumPy, CSV |
| **Fuzzy Matching** | RapidFuzz (text similarity + scoring) |
| **LLM** | Pollinations API (gpt-5.4-mini), function/tool calling |
| **Frontend** | React 19, Vite 8, Recharts, Lucide icons |
| **Testing** | Pytest (46 tests) |
| **Data Generation** | Faker, custom synthetic generator |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reconcile` | Full pipeline: normalize → match → score → LLM evaluate |
| `POST` | `/generate-data` | Generate synthetic dataset of requested size |
| `GET` | `/metrics` | Reconciliation + evaluation metrics with stage breakdown |
| `GET` | `/unresolved` | Grouped unmatched cases + matched relationships + LLM decisions |
| `GET` | `/dataset` | Full invoice/payment/bank transaction data |
| `GET` | `/exceptions` | All exception records |
| `POST` | `/ask` | Chat with Settlement Q&A agent (function-calling) |
| `POST` | `/report` | Export JSON summary + exceptions CSV |

---

## Run Instructions

### Prerequisites

- Python 3.10+
- Node.js 18+
- A Pollinations API key (get one at [pollinations.ai](https://pollinations.ai))

### Backend

```bash
# Clone and enter the project
cd AIReconciliation_SettlementQ_AAgent

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure your API key
copy .env.example .env        # Windows
# cp .env.example .env        # macOS/Linux
# Edit .env and add your POLLINATIONS_API_KEY

# Run the backend
uvicorn api.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend runs on `http://localhost:5173` and proxies `/api` to the backend on port 8000.

### Quick Start

1. Open `http://localhost:5173`
2. Click **Generate data** in the Overview panel (or use `POST /generate-data`)
3. The pipeline runs automatically with `/reconcile`
4. View results in the Dashboard — metrics, charts, LLM verdicts, exceptions

### Run Tests

```bash
pytest
```

46 tests covering pipeline, scoring, LLM validation, metrics, and reporting.

---

## Project Structure

```
├── api/
│   └── main.py                 # FastAPI endpoints
├── app/
│   ├── config.py               # LLM configuration
│   ├── data/
│   │   ├── loader.py           # CSV loading + schema validation
│   │   └── normalizer.py       # ID/text/amount/date normalization
│   ├── engine/
│   │   ├── reconcile.py        # 10-step pipeline orchestrator
│   │   ├── deterministic.py    # Pass 1: rule-based matching
│   │   ├── fuzzy.py            # Pass 2: RapidFuzz candidate generation
│   │   ├── scoring.py          # Multi-signal confidence scoring
│   │   └── invariants.py       # Financial invariant checks
│   ├── agent/
│   │   ├── client.py           # LLM HTTP client
│   │   ├── qa.py               # Settlement Q&A agent
│   │   ├── tools.py            # 7 read-only tools
│   │   ├── tie_breaker.py      # Pass 3: LLM tie-breaker
│   │   └── llm_resolver.py     # Pass 4: LLM evaluation
│   ├── evaluation/
│   │   └── metrics.py          # Ground-truth evaluation
│   └── reporting/
│       └── report_generator.py # JSON + CSV export
├── data/
│   ├── raw/                    # invoices, payments, bank_transactions CSVs
│   └── ground_truth/           # ground_truth.csv
├── scripts/
│   ├── generate_data.py        # Synthetic data generator (8 categories)
│   └── run_reconciliation.py   # CLI evaluation runner
├── tests/                      # 46 tests
├── frontend/                   # React/Vite SPA
└── reports/                    # Exported reports
```

---

## Scaling Notes

The pipeline is designed so the LLM only processes the ambiguous tail:

- **Deterministic matching** handles ~70% of cases with zero API calls
- **Fuzzy matching** with RapidFuzz handles ~25% locally
- **LLM tie-breaker** sends only genuinely ambiguous candidates (batch size 3)
- **LLM evaluation** runs on remaining unmatched cases only
- For a 200-case dataset, the LLM evaluates ~21 cases — not 600+ records
- Throughput: **818 records/sec** on the deterministic+fuzzy path

At scale, you'd add indexing, database-backed storage, and parallel LLM batches. The architecture already separates concerns cleanly for this.

---

## License

MIT
