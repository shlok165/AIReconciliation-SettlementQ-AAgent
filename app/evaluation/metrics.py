"""Evaluation metrics based on reconciliation output and optional ground truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import pandas as pd

from app.engine.reconcile import ReconciliationResult

MatchKey = Tuple[str, str, str]


@dataclass(frozen=True)
class EvaluationMetrics:
    precision: float
    recall: float
    accuracy: float
    coverage: float
    exception_rate: float
    throughput_records_per_second: float
    true_positives: int
    false_positives: int
    false_negatives: int
    expected_match_relationships: int
    resolved_match_relationships: int

    def as_dict(self) -> Dict[str, float | int]:
        return asdict(self)


def actual_match_relationships(result: ReconciliationResult) -> Set[MatchKey]:
    """Return canonical invoice-payment and payment-bank links from all resolved stages."""
    matches: Set[MatchKey] = set()
    for match in result.confirmed_invoice_payment_matches:
        matches.add(("INVOICE_PAYMENT", str(match.right_id), str(match.left_id)))
    for match in result.confirmed_payment_bank_matches:
        matches.add(("PAYMENT_BANK", str(match.left_id), str(match.right_id)))
    for scored in result.auto_matches:
        candidate = scored.candidate
        if "PAYMENT_BANK" in candidate.match_type:
            matches.add(("PAYMENT_BANK", str(candidate.left_id), str(candidate.right_id)))
        else:
            matches.add(("INVOICE_PAYMENT", str(candidate.right_id), str(candidate.left_id)))
    return matches


def expected_match_relationships(ground_truth: pd.DataFrame) -> Set[MatchKey]:
    """Read the repository ground-truth schema without treating exceptions as matches."""
    expected: Set[MatchKey] = set()
    for _, row in ground_truth.iterrows():
        if str(row.get("expected_result", "")).strip().upper() != "MATCH":
            continue
        invoice_id, payment_id, transaction_id = (str(row.get(name, "")).strip() for name in ("invoice_id", "payment_id", "transaction_id"))
        if invoice_id and payment_id and invoice_id.lower() != "nan" and payment_id.lower() != "nan":
            expected.add(("INVOICE_PAYMENT", invoice_id, payment_id))
        if payment_id and transaction_id and payment_id.lower() != "nan" and transaction_id.lower() != "nan":
            expected.add(("PAYMENT_BANK", payment_id, transaction_id))
    return expected


def calculate_evaluation_metrics(
    result: ReconciliationResult, *, ground_truth_path: Optional[Path] = None,
    elapsed_seconds: float = 0.0,
) -> EvaluationMetrics:
    actual = actual_match_relationships(result)
    expected: Set[MatchKey] = set()
    if ground_truth_path and ground_truth_path.exists():
        expected = expected_match_relationships(pd.read_csv(ground_truth_path))

    true_positives = len(actual & expected) if expected else 0
    false_positives = len(actual - expected) if expected else 0
    false_negatives = len(expected - actual) if expected else 0
    precision = true_positives / len(actual) * 100 if actual and expected else 0.0
    recall = true_positives / len(expected) * 100 if expected else 0.0
    total_records = result.metrics.total_invoices + result.metrics.total_payments + result.metrics.total_bank_transactions
    unresolved = result.metrics.unresolved_invoices + result.metrics.unresolved_payments + result.metrics.unresolved_bank_transactions
    return EvaluationMetrics(
        precision=round(precision, 2),
        recall=round(recall, 2),
        # In a link-prediction reconciliation task, accuracy is precision over
        # resolved relationships; coverage is recall over ground truth links.
        accuracy=round(precision, 2),
        coverage=round(recall, 2),
        exception_rate=round(unresolved / total_records * 100, 2) if total_records else 0.0,
        throughput_records_per_second=round(total_records / elapsed_seconds, 2) if elapsed_seconds > 0 else 0.0,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        expected_match_relationships=len(expected),
        resolved_match_relationships=len(actual),
    )
