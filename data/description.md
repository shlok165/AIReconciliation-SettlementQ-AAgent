# Finance Reconciliation Dataset

This dataset contains synthetic financial records for evaluating a multi-source finance reconciliation agent.

## Raw Data

The `raw/` directory contains the input data available to the agent:

- **`invoices.csv`** — Expected customer payments and invoice details.
- **`bank_transactions.csv`** — Transactions recorded by the bank.
- **`payments.csv`** — Payment processor and settlement records.

The raw data does **not** contain reconciliation outcomes or labels. The agent must determine whether records should be successfully reconciled or flagged as exceptions.

---

## Dataset Size

The dataset contains **60 underlying reconciliation cases**.

Because some cases intentionally contain:

- multiple invoice candidates,
- missing bank transactions,
- missing payment records,
- or records existing in only one source,

the number of records in each CSV is different.

The generated dataset contains:

| Dataset | Records |
|---|---:|
| Invoices | 63 |
| Bank transactions | 57 |
| Payments | 57 |
| Ground truth cases | 60 |

The difference is intentional:

- **63 invoices** because the 5 ambiguous cases each contain **2 invoice candidates**, adding 5 extra invoices.
- **57 bank transactions** because only 2 of the 5 genuine exception cases generate a bank transaction.
- **57 payments** because only 2 of the 5 genuine exception cases generate a payment.
- **60 ground truth records** because every underlying reconciliation scenario has exactly one ground truth case.

---

## Record Count Breakdown

| Category | Cases | Invoices | Bank Transactions | Payments |
|---|---:|---:|---:|---:|
| Clean exact matches | 30 | 30 | 30 | 30 |
| Fuzzy/date tolerance matches | 10 | 10 | 10 | 10 |
| Gateway fee cases | 5 | 5 | 5 | 5 |
| Ambiguous matches | 5 | 10 | 5 | 5 |
| Genuine exception cases | 5 | 3 | 2 | 2 |
| Amount mismatch cases | 5 | 5 | 5 | 5 |
| **Total** | **60** | **63** | **57** | **57** |

---

## Why are there 63 invoices?

Most reconciliation cases generate one invoice.

However, every ambiguous case generates **two equally plausible invoice candidates**:

```text
5 ambiguous cases × 2 invoices = 10 invoices