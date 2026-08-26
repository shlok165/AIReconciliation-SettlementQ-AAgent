"""Evaluation metrics based on reconciliation output and optional ground truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from app.engine.reconcile import ReconciliationResult

MatchKey = Tuple[str, str, str]


@dataclass(frozen=True)
class EvaluationMetrics:
    # Primary transaction-level metrics.
    total_transactions: int
    correctly_resolved_transactions: int
    transactions_requiring_review: int
    unresolved_transactions: int
    incorrectly_resolved_transactions: int
    needs_attention_transactions: int
    transaction_resolution_accuracy: float
    transaction_resolution_stage_breakdown: Dict[str, int]

    # Secondary relationship-level metrics.
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
    identification_matrix: Dict[str, Dict[str, float | int]]

    # LLM evaluation metrics (optional, populated when LLM pass is enabled).
    llm_cases_evaluated: int = 0
    llm_resolved_transactions: int = 0
    llm_incorrect_resolutions: int = 0
    llm_correct_exception_determinations: int = 0
    llm_resolution_accuracy: float = 0.0
    llm_resolution_details: List[Dict[str, Any]] = field(default_factory=list)

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


def _stage_match_relationships(result: ReconciliationResult) -> Dict[str, Set[MatchKey]]:
    """Return relationship sets split by deterministic, fuzzy, and LLM resolution stages."""
    deterministic: Set[MatchKey] = set()
    for match in result.confirmed_invoice_payment_matches:
        deterministic.add(("INVOICE_PAYMENT", str(match.right_id), str(match.left_id)))
    for match in result.confirmed_payment_bank_matches:
        deterministic.add(("PAYMENT_BANK", str(match.left_id), str(match.right_id)))

    fuzzy: Set[MatchKey] = set()
    llm: Set[MatchKey] = set()

    for scored in result.auto_matches:
        candidate = scored.candidate
        if "PAYMENT_BANK" in candidate.match_type:
            key = ("PAYMENT_BANK", str(candidate.left_id), str(candidate.right_id))
        else:
            key = ("INVOICE_PAYMENT", str(candidate.right_id), str(candidate.left_id))

        used_llm = any("LLM tie-breaker selected" in reason for reason in scored.reasons)
        if used_llm:
            llm.add(key)
        else:
            fuzzy.add(key)

    return {
        "deterministic": deterministic,
        "fuzzy": fuzzy,
        "llm": llm,
    }


def _normalize_id(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def _case_relationship_subset(
    case_invoice_ids: Set[str], case_payment_ids: Set[str], case_transaction_ids: Set[str], actual_matches: Set[MatchKey],
) -> Set[MatchKey]:
    case_matches: Set[MatchKey] = set()
    for match_type, left, right in actual_matches:
        if match_type == "INVOICE_PAYMENT":
            if left in case_invoice_ids or right in case_payment_ids:
                case_matches.add((match_type, left, right))
        elif match_type == "PAYMENT_BANK":
            if left in case_payment_ids or right in case_transaction_ids:
                case_matches.add((match_type, left, right))
    return case_matches


def _transaction_level_metrics(
    result: ReconciliationResult,
    *,
    ground_truth: pd.DataFrame,
    actual_matches: Set[MatchKey],
    stage_sets: Dict[str, Set[MatchKey]],
) -> Dict[str, float | int]:
    cases: Dict[str, Dict[str, Any]] = {}
    for row_idx, row in ground_truth.iterrows():
        case_id = _normalize_id(row.get("case_id")) or f"ROW-{row_idx}"
        expected_result = _normalize_id(row.get("expected_result")).upper() or "UNKNOWN"
        invoice_id = _normalize_id(row.get("invoice_id"))
        payment_id = _normalize_id(row.get("payment_id"))
        transaction_id = _normalize_id(row.get("transaction_id"))

        case = cases.setdefault(
            case_id,
            {
                "expected_result": expected_result,
                "invoice_ids": set(),
                "payment_ids": set(),
                "transaction_ids": set(),
                "expected_relationships": set(),
            },
        )

        if expected_result and case["expected_result"] == "UNKNOWN":
            case["expected_result"] = expected_result

        if invoice_id:
            case["invoice_ids"].add(invoice_id)
        if payment_id:
            case["payment_ids"].add(payment_id)
        if transaction_id:
            case["transaction_ids"].add(transaction_id)

        if expected_result == "MATCH":
            if invoice_id and payment_id:
                case["expected_relationships"].add(("INVOICE_PAYMENT", invoice_id, payment_id))
            if payment_id and transaction_id:
                case["expected_relationships"].add(("PAYMENT_BANK", payment_id, transaction_id))

    unresolved_invoice_ids = set(result.unresolved_invoice_ids)
    unresolved_payment_ids = set(result.unresolved_payment_ids)
    unresolved_bank_ids = set(result.unresolved_bank_transaction_ids)
    review_payment_ids = {item.candidate.left_id for item in result.review_candidates}

    exception_entity_ids: Set[str] = set()
    for exception in result.exceptions:
        rec_id = _normalize_id(exception.record_id)
        if rec_id:
            exception_entity_ids.add(rec_id)
        for rel_id in getattr(exception, "related_ids", []) or []:
            norm_rel_id = _normalize_id(rel_id)
            if norm_rel_id:
                exception_entity_ids.add(norm_rel_id)

    total_transactions = len(cases)
    correctly_resolved_transactions = 0
    transactions_requiring_review = 0
    unresolved_transactions = 0
    incorrectly_resolved_transactions = 0
    transaction_stage_breakdown: Dict[str, int] = {
        "deterministic_resolved_transactions": 0,
        "fuzzy_resolved_transactions": 0,
        "llm_resolved_transactions": 0,
        "exception_resolved_transactions": 0,
        "review_transactions": 0,
        "unresolved_transactions": 0,
        "incorrect_transactions": 0,
    }

    for case in cases.values():
        case_invoice_ids = case["invoice_ids"]
        case_payment_ids = case["payment_ids"]
        case_transaction_ids = case["transaction_ids"]
        case_expected = case["expected_relationships"]
        expected_result = case["expected_result"]

        case_actual = _case_relationship_subset(
            case_invoice_ids,
            case_payment_ids,
            case_transaction_ids,
            actual_matches,
        )
        wrong_links = case_actual - case_expected
        expected_links_found = case_expected.issubset(case_actual)

        has_review = any(payment_id in review_payment_ids for payment_id in case_payment_ids)
        has_unresolved = (
            any(invoice_id in unresolved_invoice_ids for invoice_id in case_invoice_ids)
            or any(payment_id in unresolved_payment_ids for payment_id in case_payment_ids)
            or any(transaction_id in unresolved_bank_ids for transaction_id in case_transaction_ids)
        )
        has_exception_signal = any(
            entity_id in exception_entity_ids
            for entity_id in (case_invoice_ids | case_payment_ids | case_transaction_ids)
        )

        if expected_result == "MATCH":
            if wrong_links:
                incorrectly_resolved_transactions += 1
                transaction_stage_breakdown["incorrect_transactions"] += 1
            elif expected_links_found and not has_review and not has_unresolved:
                correctly_resolved_transactions += 1
                has_llm = any(rel in stage_sets["llm"] for rel in case_expected)
                has_fuzzy = any(rel in stage_sets["fuzzy"] for rel in case_expected)
                if has_llm:
                    transaction_stage_breakdown["llm_resolved_transactions"] += 1
                elif has_fuzzy:
                    transaction_stage_breakdown["fuzzy_resolved_transactions"] += 1
                else:
                    transaction_stage_breakdown["deterministic_resolved_transactions"] += 1
            elif has_review:
                transactions_requiring_review += 1
                transaction_stage_breakdown["review_transactions"] += 1
            else:
                unresolved_transactions += 1
                transaction_stage_breakdown["unresolved_transactions"] += 1
        else:
            if case_actual:
                incorrectly_resolved_transactions += 1
                transaction_stage_breakdown["incorrect_transactions"] += 1
            elif has_review:
                transactions_requiring_review += 1
                transaction_stage_breakdown["review_transactions"] += 1
            elif has_unresolved or has_exception_signal:
                correctly_resolved_transactions += 1
                transaction_stage_breakdown["exception_resolved_transactions"] += 1
            else:
                unresolved_transactions += 1
                transaction_stage_breakdown["unresolved_transactions"] += 1

    needs_attention_transactions = transactions_requiring_review + unresolved_transactions
    transaction_resolution_accuracy = (
        correctly_resolved_transactions / total_transactions * 100.0
        if total_transactions > 0
        else 0.0
    )

    return {
        "total_transactions": total_transactions,
        "correctly_resolved_transactions": correctly_resolved_transactions,
        "transactions_requiring_review": transactions_requiring_review,
        "unresolved_transactions": unresolved_transactions,
        "incorrectly_resolved_transactions": incorrectly_resolved_transactions,
        "needs_attention_transactions": needs_attention_transactions,
        "transaction_resolution_accuracy": round(transaction_resolution_accuracy, 2),
        "transaction_resolution_stage_breakdown": transaction_stage_breakdown,
    }


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
    llm_resolution_result: Optional[Any] = None,
) -> EvaluationMetrics:
    actual = actual_match_relationships(result)
    stage_sets = _stage_match_relationships(result)
    transaction_metrics = {
        "total_transactions": 0,
        "correctly_resolved_transactions": 0,
        "transactions_requiring_review": 0,
        "unresolved_transactions": 0,
        "incorrectly_resolved_transactions": 0,
        "needs_attention_transactions": 0,
        "transaction_resolution_accuracy": 0.0,
        "transaction_resolution_stage_breakdown": {
            "deterministic_resolved_transactions": 0,
            "fuzzy_resolved_transactions": 0,
            "llm_resolved_transactions": 0,
            "exception_resolved_transactions": 0,
            "review_transactions": 0,
            "unresolved_transactions": 0,
            "incorrect_transactions": 0,
        },
    }
    expected: Set[MatchKey] = set()
    ground_truth_df: Optional[pd.DataFrame] = None
    if ground_truth_path and ground_truth_path.exists():
        ground_truth_df = pd.read_csv(ground_truth_path)
        expected = expected_match_relationships(ground_truth_df)
        transaction_metrics = _transaction_level_metrics(
            result,
            ground_truth=ground_truth_df,
            actual_matches=actual,
            stage_sets=stage_sets,
        )

    true_positives = len(actual & expected) if expected else 0
    false_positives = len(actual - expected) if expected else 0
    false_negatives = len(expected - actual) if expected else 0
    precision = true_positives / len(actual) * 100 if actual and expected else 0.0
    recall = true_positives / len(expected) * 100 if expected else 0.0
    total_records = result.metrics.total_invoices + result.metrics.total_payments + result.metrics.total_bank_transactions
    unresolved = result.metrics.unresolved_invoices + result.metrics.unresolved_payments + result.metrics.unresolved_bank_transactions

    identification_matrix: Dict[str, Dict[str, float | int]] = {}
    expected_count = len(expected)
    for stage_name, stage_matches in stage_sets.items():
        correct = len(stage_matches & expected) if expected else 0
        incorrect = len(stage_matches - expected) if expected else 0
        identified = len(stage_matches)
        stage_precision = (correct / identified * 100) if identified and expected else 0.0
        stage_coverage = (correct / expected_count * 100) if expected_count else 0.0
        identification_matrix[stage_name] = {
            "identified_relationships": identified,
            "correct_relationships": correct,
            "incorrect_relationships": incorrect,
            "precision": round(stage_precision, 2),
            "coverage_contribution": round(stage_coverage, 2),
        }

    llm_evaluated = 0
    llm_resolved = 0
    llm_incorrect = 0
    llm_correct_exception = 0
    llm_accuracy = 0.0
    llm_details: List[Dict[str, Any]] = []

    if llm_resolution_result is not None:
        llm_evaluated = getattr(llm_resolution_result, "total_cases_evaluated", 0)
        llm_resolved = getattr(llm_resolution_result, "llm_resolved_count", 0)
        llm_incorrect = getattr(llm_resolution_result, "llm_incorrect_count", 0)
        llm_correct_exception = getattr(llm_resolution_result, "llm_correct_exception_count", 0)
        llm_accuracy = getattr(llm_resolution_result, "llm_resolution_accuracy", 0.0)
        llm_details = getattr(llm_resolution_result, "details", [])

    # Add LLM eval resolved count into the stage breakdown so the bar chart
    # reflects transactions resolved by the LLM evaluation pass.
    if llm_resolved > 0:
        transaction_metrics["transaction_resolution_stage_breakdown"]["llm_resolved_transactions"] += llm_resolved
        transaction_metrics["correctly_resolved_transactions"] += llm_resolved
        transaction_metrics["review_transactions"] = max(0, transaction_metrics["transactions_requiring_review"] - llm_resolved)
        transaction_metrics["needs_attention_transactions"] = (
            transaction_metrics["transactions_requiring_review"] + transaction_metrics["unresolved_transactions"]
        )
        total = transaction_metrics["total_transactions"]
        if total > 0:
            transaction_metrics["transaction_resolution_accuracy"] = round(
                transaction_metrics["correctly_resolved_transactions"] / total * 100.0, 2
            )

    return EvaluationMetrics(
        total_transactions=transaction_metrics["total_transactions"],
        correctly_resolved_transactions=transaction_metrics["correctly_resolved_transactions"],
        transactions_requiring_review=transaction_metrics["transactions_requiring_review"],
        unresolved_transactions=transaction_metrics["unresolved_transactions"],
        incorrectly_resolved_transactions=transaction_metrics["incorrectly_resolved_transactions"],
        needs_attention_transactions=transaction_metrics["needs_attention_transactions"],
        transaction_resolution_accuracy=transaction_metrics["transaction_resolution_accuracy"],
        transaction_resolution_stage_breakdown=transaction_metrics["transaction_resolution_stage_breakdown"],
        precision=round(precision, 2),
        recall=round(recall, 2),
        accuracy=round(precision, 2),
        coverage=round(recall, 2),
        exception_rate=round(unresolved / total_records * 100, 2) if total_records else 0.0,
        throughput_records_per_second=round(total_records / elapsed_seconds, 2) if elapsed_seconds > 0 else 0.0,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        expected_match_relationships=len(expected),
        resolved_match_relationships=len(actual),
        identification_matrix=identification_matrix,
        llm_cases_evaluated=llm_evaluated,
        llm_resolved_transactions=llm_resolved,
        llm_incorrect_resolutions=llm_incorrect,
        llm_correct_exception_determinations=llm_correct_exception,
        llm_resolution_accuracy=llm_accuracy,
        llm_resolution_details=llm_details,
    )
