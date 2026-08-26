from pathlib import Path

import pandas as pd

from app.engine.reconcile import reconcile
from app.evaluation.metrics import calculate_evaluation_metrics


def test_transaction_metrics_count_match_and_exception_correct(tmp_path):
    invoices = pd.DataFrame(
        [
            {
                "invoice_id": "INV-1",
                "expected_amount": 10.0,
                "status": "posted",
                "invoice_date": "2026-01-01",
                "customer_ref": "C1",
                "description": "Matched invoice",
            },
            {
                "invoice_id": "INV-2",
                "expected_amount": 20.0,
                "status": "posted",
                "invoice_date": "2026-01-01",
                "customer_ref": "C2",
                "description": "Orphan invoice",
            },
        ]
    )
    payments = pd.DataFrame(
        [
            {
                "payment_id": "PAY-1",
                "gross_amount": 10.0,
                "fee": 0.0,
                "net_settled_amount": 10.0,
                "settlement_date": "2026-01-02",
                "linked_invoice_id": "INV-1",
            }
        ]
    )
    bank = pd.DataFrame(
        [
            {
                "transaction_id": "TXN-1",
                "amount": 10.0,
                "date": "2026-01-02",
                "description": "PAY-1",
                "reference_no": "PAY-1",
            }
        ]
    )

    result = reconcile(invoices, payments, bank)

    gt_path = tmp_path / "ground_truth.csv"
    pd.DataFrame(
        [
            {
                "case_id": "CASE-1",
                "invoice_id": "INV-1",
                "payment_id": "PAY-1",
                "transaction_id": "TXN-1",
                "expected_result": "MATCH",
                "exception_reason": "",
            },
            {
                "case_id": "CASE-2",
                "invoice_id": "INV-2",
                "payment_id": "",
                "transaction_id": "",
                "expected_result": "EXCEPTION",
                "exception_reason": "ORPHAN_INVOICE",
            },
        ]
    ).to_csv(gt_path, index=False)

    evaluation = calculate_evaluation_metrics(result, ground_truth_path=gt_path, elapsed_seconds=1.0)

    assert evaluation.total_transactions == 2
    assert evaluation.correctly_resolved_transactions == 2
    assert evaluation.transactions_requiring_review == 0
    assert evaluation.unresolved_transactions == 0
    assert evaluation.incorrectly_resolved_transactions == 0
    assert evaluation.needs_attention_transactions == 0
    assert evaluation.transaction_resolution_accuracy == 100.0


def test_transaction_metrics_detect_false_resolution_for_expected_exception(tmp_path):
    invoices = pd.DataFrame(
        [
            {
                "invoice_id": "INV-1",
                "expected_amount": 10.0,
                "status": "posted",
                "invoice_date": "2026-01-01",
                "customer_ref": "C1",
                "description": "Should not be matched in truth",
            }
        ]
    )
    payments = pd.DataFrame(
        [
            {
                "payment_id": "PAY-1",
                "gross_amount": 10.0,
                "fee": 0.0,
                "net_settled_amount": 10.0,
                "settlement_date": "2026-01-02",
                "linked_invoice_id": "INV-1",
            }
        ]
    )
    bank = pd.DataFrame(
        [
            {
                "transaction_id": "TXN-1",
                "amount": 10.0,
                "date": "2026-01-02",
                "description": "PAY-1",
                "reference_no": "PAY-1",
            }
        ]
    )

    result = reconcile(invoices, payments, bank)

    gt_path = tmp_path / "ground_truth.csv"
    pd.DataFrame(
        [
            {
                "case_id": "CASE-X",
                "invoice_id": "INV-1",
                "payment_id": "PAY-1",
                "transaction_id": "TXN-1",
                "expected_result": "EXCEPTION",
                "exception_reason": "ORPHAN_PAYMENT",
            }
        ]
    ).to_csv(gt_path, index=False)

    evaluation = calculate_evaluation_metrics(result, ground_truth_path=gt_path, elapsed_seconds=1.0)

    assert evaluation.total_transactions == 1
    assert evaluation.correctly_resolved_transactions == 0
    assert evaluation.incorrectly_resolved_transactions == 1
    assert evaluation.transaction_resolution_accuracy == 0.0
