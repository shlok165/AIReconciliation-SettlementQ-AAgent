"""Tests for the LLM resolver module."""

from pathlib import Path

import pandas as pd

from app.agent.llm_resolver import (
    LLMResolutionDecision,
    UnmatchedCase,
    build_unmatched_cases,
    validate_llm_decisions,
    _build_ground_truth_map,
    _expected_partner_id,
)
from app.engine.reconcile import reconcile, ExceptionRecord


def _make_test_data():
    invoices = pd.DataFrame([
        {
            "invoice_id": "INV-001",
            "expected_amount": 500.0,
            "status": "PAID",
            "invoice_date": "2026-01-01",
            "customer_ref": "CUST-1001",
            "description": "Cloud hosting services",
        },
        {
            "invoice_id": "INV-002",
            "expected_amount": 750.0,
            "status": "PAID",
            "invoice_date": "2026-01-02",
            "customer_ref": "CUST-1002",
            "description": "Database consulting",
        },
        {
            "invoice_id": "INV-003",
            "expected_amount": 1200.0,
            "status": "UNPAID",
            "invoice_date": "2026-01-03",
            "customer_ref": "CUST-1003",
            "description": "Orphan invoice no payment",
        },
    ])
    payments = pd.DataFrame([
        {
            "payment_id": "PAY-001",
            "gross_amount": 500.0,
            "fee": 0.0,
            "net_settled_amount": 500.0,
            "settlement_date": "2026-01-02",
            "linked_invoice_id": "INV-001",
        },
        {
            "payment_id": "PAY-002",
            "gross_amount": 750.0,
            "fee": 0.0,
            "net_settled_amount": 750.0,
            "settlement_date": "2026-01-03",
            "linked_invoice_id": "",
        },
    ])
    bank = pd.DataFrame([
        {
            "transaction_id": "TXN-001",
            "amount": 500.0,
            "date": "2026-01-02",
            "description": "ACH credit INV-001",
            "reference_no": "REF-001",
        },
        {
            "transaction_id": "TXN-002",
            "amount": 750.0,
            "date": "2026-01-03",
            "description": "Wire transfer",
            "reference_no": "REF-002",
        },
        {
            "transaction_id": "TXN-003",
            "amount": 200.0,
            "date": "2026-01-05",
            "description": "Mystery deposit",
            "reference_no": "UNKNOWN",
        },
    ])
    return invoices, payments, bank


def test_build_unmatched_cases_identifies_review_candidates():
    invoices, payments, bank = _make_test_data()
    result = reconcile(invoices, payments, bank)

    cases = build_unmatched_cases(result, invoices, payments, bank)

    assert isinstance(cases, list)
    for case in cases:
        assert isinstance(case, UnmatchedCase)
        assert case.case_id.startswith("LLM-")
        assert case.record_id
        assert case.record_type in ("PAYMENT", "BANK_TRANSACTION", "INVOICE")
        assert isinstance(case.source_record, dict)
        assert isinstance(case.potential_candidates, list)


def test_build_unmatched_cases_finds_nearby_candidates():
    invoices, payments, bank = _make_test_data()
    result = reconcile(invoices, payments, bank)

    cases = build_unmatched_cases(result, invoices, payments, bank, max_candidates_per_case=3)

    for case in cases:
        for cand in case.potential_candidates:
            assert "type" in cand
            assert "id" in cand
            assert cand["type"] in ("INVOICE", "PAYMENT", "BANK_TXN")


def test_validate_llm_decisions_correct_match():
    gt = pd.DataFrame([
        {
            "case_id": "CASE-1",
            "invoice_id": "INV-001",
            "payment_id": "PAY-001",
            "transaction_id": "TXN-001",
            "expected_result": "MATCH",
            "exception_reason": "",
            "category": "CLEAN",
        },
    ])

    cases = [
        UnmatchedCase(
            case_id="LLM-0001",
            exception_type="MANUAL_REVIEW_REQUIRED",
            record_id="PAY-001",
            record_type="PAYMENT",
            description="Review candidate",
            related_ids=["INV-001"],
            source_record={"payment_id": "PAY-001", "gross_amount": 500.0},
            potential_candidates=[],
        ),
    ]

    decisions = [
        LLMResolutionDecision(
            case_id="LLM-0001",
            record_id="PAY-001",
            resolution="MATCH",
            matched_ids=["INV-001"],
            confidence=95.0,
            justification="Amount and date match.",
        ),
    ]

    result = validate_llm_decisions(decisions, cases, gt)
    assert result.llm_resolved_count == 1
    assert result.llm_incorrect_count == 0
    assert result.details[0]["verdict"] == "CORRECTLY_RESOLVED"


def test_validate_llm_decisions_incorrect_match():
    gt = pd.DataFrame([
        {
            "case_id": "CASE-1",
            "invoice_id": "INV-001",
            "payment_id": "PAY-001",
            "transaction_id": "TXN-001",
            "expected_result": "MATCH",
            "exception_reason": "",
            "category": "CLEAN",
        },
    ])

    cases = [
        UnmatchedCase(
            case_id="LLM-0001",
            exception_type="MANUAL_REVIEW_REQUIRED",
            record_id="PAY-001",
            record_type="PAYMENT",
            description="Review candidate",
            related_ids=["INV-002"],
            source_record={"payment_id": "PAY-001", "gross_amount": 500.0},
            potential_candidates=[],
        ),
    ]

    decisions = [
        LLMResolutionDecision(
            case_id="LLM-0001",
            record_id="PAY-001",
            resolution="MATCH",
            matched_ids=["INV-002"],
            confidence=85.0,
            justification="Wrong match.",
        ),
    ]

    result = validate_llm_decisions(decisions, cases, gt)
    assert result.llm_resolved_count == 0
    assert result.llm_incorrect_count == 1
    assert result.details[0]["verdict"] == "INCORRECTLY_RESOLVED"


def test_validate_llm_decisions_correct_exception():
    gt = pd.DataFrame([
        {
            "case_id": "CASE-1",
            "invoice_id": "INV-003",
            "payment_id": "",
            "transaction_id": "",
            "expected_result": "EXCEPTION",
            "exception_reason": "ORPHAN_INVOICE",
            "category": "GENUINE_EXCEPTION",
        },
    ])

    cases = [
        UnmatchedCase(
            case_id="LLM-0001",
            exception_type="NO_MATCH_FOUND",
            record_id="INV-003",
            record_type="INVOICE",
            description="No match found.",
            related_ids=[],
            source_record={"invoice_id": "INV-003", "expected_amount": 1200.0},
            potential_candidates=[],
        ),
    ]

    decisions = [
        LLMResolutionDecision(
            case_id="LLM-0001",
            record_id="INV-003",
            resolution="EXCEPTION",
            matched_ids=[],
            confidence=90.0,
            justification="No matching payment found.",
        ),
    ]

    result = validate_llm_decisions(decisions, cases, gt)
    assert result.llm_resolved_count == 0
    assert result.llm_incorrect_count == 0
    assert result.llm_correct_exception_count == 1
    assert result.details[0]["verdict"] == "CORRECTLY_EXCEPTION"


def test_validate_llm_decisions_incorrect_exception_when_should_match():
    gt = pd.DataFrame([
        {
            "case_id": "CASE-1",
            "invoice_id": "INV-001",
            "payment_id": "PAY-001",
            "transaction_id": "TXN-001",
            "expected_result": "MATCH",
            "exception_reason": "",
            "category": "CLEAN",
        },
    ])

    cases = [
        UnmatchedCase(
            case_id="LLM-0001",
            exception_type="MANUAL_REVIEW_REQUIRED",
            record_id="PAY-001",
            record_type="PAYMENT",
            description="Review candidate",
            related_ids=["INV-001"],
            source_record={"payment_id": "PAY-001", "gross_amount": 500.0},
            potential_candidates=[],
        ),
    ]

    decisions = [
        LLMResolutionDecision(
            case_id="LLM-0001",
            record_id="PAY-001",
            resolution="EXCEPTION",
            matched_ids=[],
            confidence=80.0,
            justification="Cannot determine match.",
        ),
    ]

    result = validate_llm_decisions(decisions, cases, gt)
    assert result.llm_resolved_count == 0
    assert result.llm_incorrect_count == 1
    assert result.details[0]["verdict"] == "INCORRECTLY_RESOLVED"


def test_validate_llm_decisions_accuracy_calculation():
    gt = pd.DataFrame([
        {"case_id": "C1", "invoice_id": "INV-1", "payment_id": "PAY-1", "transaction_id": "", "expected_result": "MATCH", "exception_reason": "", "category": ""},
        {"case_id": "C2", "invoice_id": "INV-2", "payment_id": "", "transaction_id": "", "expected_result": "EXCEPTION", "exception_reason": "ORPHAN", "category": ""},
        {"case_id": "C3", "invoice_id": "INV-3", "payment_id": "PAY-3", "transaction_id": "", "expected_result": "MATCH", "exception_reason": "", "category": ""},
    ])

    cases = [
        UnmatchedCase(case_id="L1", exception_type="REVIEW", record_id="PAY-1", record_type="PAYMENT", description="", related_ids=[], source_record={}, potential_candidates=[]),
        UnmatchedCase(case_id="L2", exception_type="NO_MATCH", record_id="INV-2", record_type="INVOICE", description="", related_ids=[], source_record={}, potential_candidates=[]),
        UnmatchedCase(case_id="L3", exception_type="REVIEW", record_id="PAY-3", record_type="PAYMENT", description="", related_ids=[], source_record={}, potential_candidates=[]),
    ]

    decisions = [
        LLMResolutionDecision(case_id="L1", record_id="PAY-1", resolution="MATCH", matched_ids=["INV-1"], confidence=95, justification=""),
        LLMResolutionDecision(case_id="L2", record_id="INV-2", resolution="EXCEPTION", matched_ids=[], confidence=90, justification=""),
        LLMResolutionDecision(case_id="L3", record_id="PAY-3", resolution="MATCH", matched_ids=["INV-1"], confidence=85, justification=""),
    ]

    result = validate_llm_decisions(decisions, cases, gt)
    assert result.total_cases_evaluated == 3
    assert result.llm_resolved_count == 1
    assert result.llm_correct_exception_count == 1
    assert result.llm_incorrect_count == 1
    assert result.llm_resolution_accuracy == round((1 + 1) / 3 * 100, 2)


def test_ground_truth_map_covers_all_entities():
    gt = pd.DataFrame([
        {"case_id": "C1", "invoice_id": "INV-1", "payment_id": "PAY-1", "transaction_id": "TXN-1", "expected_result": "MATCH", "exception_reason": "", "category": ""},
    ])

    gt_map = _build_ground_truth_map(gt)
    assert "INV-1" in gt_map
    assert "PAY-1" in gt_map
    assert "TXN-1" in gt_map
    assert gt_map["INV-1"]["expected_result"] == "MATCH"


def test_expected_partner_id_for_payment():
    assert _expected_partner_id("PAYMENT", "INV-1", "", "TXN-1") == "INV-1"
    assert _expected_partner_id("PAYMENT", "", "PAY-1", "TXN-1") == "TXN-1"
    assert _expected_partner_id("PAYMENT", "", "", "") == ""


def test_expected_partner_id_for_bank_transaction():
    assert _expected_partner_id("BANK_TRANSACTION", "", "PAY-1", "TXN-1") == "PAY-1"


def test_expected_partner_id_for_invoice():
    assert _expected_partner_id("INVOICE", "INV-1", "PAY-1", "") == "PAY-1"


def test_validate_with_empty_decisions():
    gt = pd.DataFrame([
        {"case_id": "C1", "invoice_id": "INV-1", "payment_id": "PAY-1", "transaction_id": "", "expected_result": "MATCH", "exception_reason": "", "category": ""},
    ])

    result = validate_llm_decisions([], [], gt)
    assert result.total_cases_evaluated == 0
    assert result.llm_resolution_accuracy == 0.0


def test_e2e_reconcile_with_llm_eval_off():
    invoices, payments, bank = _make_test_data()
    result = reconcile(invoices, payments, bank)
    assert result.llm_evaluation_result is None


def test_metrics_includes_llm_fields():
    from app.evaluation.metrics import EvaluationMetrics

    metrics = EvaluationMetrics(
        total_transactions=10,
        correctly_resolved_transactions=8,
        transactions_requiring_review=1,
        unresolved_transactions=1,
        incorrectly_resolved_transactions=0,
        needs_attention_transactions=2,
        transaction_resolution_accuracy=80.0,
        transaction_resolution_stage_breakdown={},
        precision=90.0,
        recall=85.0,
        accuracy=90.0,
        coverage=85.0,
        exception_rate=10.0,
        throughput_records_per_second=100.0,
        true_positives=9,
        false_positives=1,
        false_negatives=2,
        expected_match_relationships=11,
        resolved_match_relationships=10,
        identification_matrix={},
        llm_cases_evaluated=5,
        llm_resolved_transactions=3,
        llm_incorrect_resolutions=1,
        llm_correct_exception_determinations=1,
        llm_resolution_accuracy=80.0,
        llm_resolution_details=[],
    )

    d = metrics.as_dict()
    assert d["llm_cases_evaluated"] == 5
    assert d["llm_resolved_transactions"] == 3
    assert d["llm_incorrect_resolutions"] == 1
    assert d["llm_resolution_accuracy"] == 80.0
