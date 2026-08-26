from pathlib import Path

import pandas as pd

from app.engine.reconcile import reconcile
from app.evaluation.metrics import calculate_evaluation_metrics
from app.reporting.report_generator import generate_final_report


def _datasets():
    invoices = pd.DataFrame([{"invoice_id": "INV-1", "expected_amount": 10.0, "status": "posted", "invoice_date": "2026-01-01", "customer_ref": "C1", "description": "Test"}])
    payments = pd.DataFrame([{"payment_id": "PAY-1", "gross_amount": 10.0, "fee": 0.0, "net_settled_amount": 10.0, "settlement_date": "2026-01-02", "linked_invoice_id": "INV-1"}])
    bank = pd.DataFrame([{"transaction_id": "TXN-1", "amount": 10.0, "date": "2026-01-02", "description": "PAY-1", "reference_no": "PAY-1"}])
    return invoices, payments, bank


def test_metrics_and_report_export(tmp_path):
    invoices, payments, bank = _datasets()
    result = reconcile(invoices, payments, bank)
    ground_truth = tmp_path / "ground_truth.csv"
    pd.DataFrame([{"invoice_id": "INV-1", "payment_id": "PAY-1", "transaction_id": "TXN-1", "expected_result": "MATCH"}]).to_csv(ground_truth, index=False)
    evaluation = calculate_evaluation_metrics(result, ground_truth_path=ground_truth, elapsed_seconds=1.0)
    paths = generate_final_report(result, evaluation, output_dir=tmp_path / "reports")
    assert evaluation.precision == 100.0
    assert evaluation.coverage == 100.0
    assert Path(paths["summary"]).exists()
    assert Path(paths["exceptions"]).exists()
