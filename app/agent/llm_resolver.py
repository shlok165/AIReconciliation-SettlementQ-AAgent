"""LLM-powered resolution attempt for unmatched reconciliation transactions.

After deterministic, fuzzy, and tie-breaker passes, some transactions remain
unresolved or under review. This module sends rich context for each such case
to an LLM and evaluates whether the model can resolve them correctly.

Key design: batches multiple cases per LLM call for speed, and uses a
simplified prompt format that gives the model clear candidate comparison tables.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app.agent.client import PollinationsClient, LLMClientError
from app.engine.reconcile import ExceptionRecord, ReconciliationResult
from app.engine.scoring import ScoredCandidate

logger = logging.getLogger(__name__)

BATCH_SIZE = 3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UnmatchedCase:
    """A single transaction the engine failed to resolve, with full context."""
    case_id: str
    exception_type: str
    record_id: str
    record_type: str
    description: str
    related_ids: List[str]
    source_record: Dict[str, Any]
    potential_candidates: List[Dict[str, Any]]


@dataclass(frozen=True)
class LLMResolutionDecision:
    """LLM verdict on a single unmatched case."""
    case_id: str
    record_id: str
    resolution: str  # "MATCH" or "EXCEPTION"
    matched_ids: List[str]
    confidence: float
    justification: str


@dataclass
class LLMResolutionResult:
    """Aggregate outcome of the LLM evaluation pass."""
    decisions: List[LLMResolutionDecision]
    total_cases_evaluated: int
    llm_resolved_count: int
    llm_incorrect_count: int
    llm_correct_exception_count: int
    llm_resolution_accuracy: float
    details: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        v = float(val)
        return v if not pd.isna(v) else None
    except (ValueError, TypeError):
        return None


def _row_to_dict(df: pd.DataFrame, id_column: str, record_id: str) -> Dict[str, Any]:
    if not record_id:
        return {"id": record_id, "found": False}
    matches = df[df[id_column].astype(str).str.upper() == record_id.strip().upper()]
    if matches.empty:
        return {"id": record_id, "found": False}
    raw = matches.iloc[0].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in raw.items() if not k.endswith("_normalized")}


def _find_exception_for_record(
    exceptions: List[ExceptionRecord], record_id: str
) -> Optional[ExceptionRecord]:
    for exc in exceptions:
        if exc.record_id.upper() == record_id.upper():
            return exc
    return None


def _find_nearby_candidates(
    source_record: Dict[str, Any],
    source_type: str,
    invoices: pd.DataFrame,
    payments: pd.DataFrame,
    bank_transactions: pd.DataFrame,
    *,
    exclude_ids: Optional[set] = None,
    max_candidates: int = 5,
) -> List[Dict[str, Any]]:
    """Find records with similar amount that could be potential matches.

    Skips any record whose ID is in ``exclude_ids`` (already matched by
    deterministic / fuzzy / tie-breaker passes).
    """
    skip = exclude_ids or set()
    candidates: List[Dict[str, Any]] = []

    if source_type == "PAYMENT":
        pay_amount = _safe_float(source_record.get("net_settled_amount") or source_record.get("gross_amount"))
        if pay_amount is None:
            return candidates

        for _, inv_row in invoices.iterrows():
            inv_id = inv_row.get("invoice_id", "")
            if inv_id in skip:
                continue
            inv_amount = _safe_float(inv_row.get("expected_amount"))
            if inv_amount is not None:
                diff = abs(pay_amount - inv_amount)
                diff_pct = diff / max(inv_amount, 1) * 100
                if diff_pct <= 15:
                    candidates.append({
                        "type": "INVOICE",
                        "id": inv_id,
                        "amount": inv_amount,
                        "date": str(inv_row.get("invoice_date", "")),
                        "description": str(inv_row.get("description", ""))[:60],
                        "status": str(inv_row.get("status", "")),
                        "amount_diff": round(diff, 2),
                    })

        for _, txn_row in bank_transactions.iterrows():
            txn_id = txn_row.get("transaction_id", "")
            if txn_id in skip:
                continue
            txn_amount = _safe_float(txn_row.get("amount"))
            if txn_amount is not None:
                diff = abs(pay_amount - txn_amount)
                diff_pct = diff / max(txn_amount, 1) * 100
                if diff_pct <= 15:
                    candidates.append({
                        "type": "BANK_TXN",
                        "id": txn_id,
                        "amount": txn_amount,
                        "date": str(txn_row.get("date", "")),
                        "description": str(txn_row.get("description", ""))[:60],
                        "reference": str(txn_row.get("reference_no", "")),
                        "amount_diff": round(diff, 2),
                    })

    elif source_type == "BANK_TRANSACTION":
        txn_amount = _safe_float(source_record.get("amount"))
        if txn_amount is None:
            return candidates

        for _, pay_row in payments.iterrows():
            pay_id = pay_row.get("payment_id", "")
            if pay_id in skip:
                continue
            pay_amount = _safe_float(pay_row.get("net_settled_amount") or pay_row.get("gross_amount"))
            if pay_amount is not None:
                diff = abs(txn_amount - pay_amount)
                diff_pct = diff / max(pay_amount, 1) * 100
                if diff_pct <= 15:
                    candidates.append({
                        "type": "PAYMENT",
                        "id": pay_id,
                        "amount": pay_amount,
                        "date": str(pay_row.get("settlement_date", "")),
                        "description": f"gross={pay_row.get('gross_amount','')} fee={pay_row.get('fee','')} net={pay_row.get('net_settled_amount','')}",
                        "linked_invoice": str(pay_row.get("linked_invoice_id", "")),
                        "amount_diff": round(diff, 2),
                    })

    elif source_type == "INVOICE":
        inv_amount = _safe_float(source_record.get("expected_amount"))
        if inv_amount is None:
            return candidates

        for _, pay_row in payments.iterrows():
            pay_id = pay_row.get("payment_id", "")
            if pay_id in skip:
                continue
            pay_amount = _safe_float(pay_row.get("net_settled_amount") or pay_row.get("gross_amount"))
            if pay_amount is not None:
                diff = abs(inv_amount - pay_amount)
                diff_pct = diff / max(inv_amount, 1) * 100
                if diff_pct <= 15:
                    candidates.append({
                        "type": "PAYMENT",
                        "id": pay_id,
                        "amount": pay_amount,
                        "date": str(pay_row.get("settlement_date", "")),
                        "description": f"gross={pay_row.get('gross_amount','')} fee={pay_row.get('fee','')} net={pay_row.get('net_settled_amount','')}",
                        "linked_invoice": str(pay_row.get("linked_invoice_id", "")),
                        "amount_diff": round(diff, 2),
                    })

    candidates.sort(key=lambda c: c.get("amount_diff", 999))
    return candidates[:max_candidates]


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_unmatched_cases(
    result: ReconciliationResult,
    invoices: pd.DataFrame,
    payments: pd.DataFrame,
    bank_transactions: pd.DataFrame,
    *,
    ground_truth_path: Optional[Path] = None,
    max_candidates_per_case: int = 5,
) -> List[UnmatchedCase]:
    """Build one case per unique ground-truth case that has unmatched records.

    If a single ground truth case has an unmatched payment, bank txn, and
    invoice, they all belong together in one LLM case — not three.
    """
    already_matched: set = set()
    for m in result.confirmed_payment_bank_matches:
        already_matched.add(m.left_id)
        already_matched.add(m.right_id)
    for m in result.confirmed_invoice_payment_matches:
        already_matched.add(m.left_id)
        already_matched.add(m.right_id)
    for auto in result.auto_matches:
        already_matched.add(auto.candidate.left_id)
        already_matched.add(auto.candidate.right_id)

    unresolved_payments = set(result.unresolved_payment_ids) - already_matched
    unresolved_txns = set(result.unresolved_bank_transaction_ids) - already_matched
    unresolved_invoices = set(result.unresolved_invoice_ids) - already_matched
    review_payment_ids = {r.candidate.left_id for r in result.review_candidates if r.candidate.left_id not in already_matched}

    all_unmatched = unresolved_payments | unresolved_txns | unresolved_invoices | review_payment_ids

    if not all_unmatched:
        return []

    # Build ground-truth case lookup: record_id -> (case_id, case_row)
    gt_lookup: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    has_gt = ground_truth_path is not None and ground_truth_path.exists()

    if has_gt:
        gt_df = pd.read_csv(ground_truth_path)
        for _, row in gt_df.iterrows():
            case_id = str(row.get("case_id", ""))
            for col in ("payment_id", "invoice_id", "transaction_id"):
                rid = str(row.get(col, "")).strip()
                if rid and rid.lower() != "nan" and rid in all_unmatched:
                    gt_lookup[rid] = (case_id, row.to_dict())

    # Group unmatched records by ground truth case
    case_groups: Dict[str, List[Tuple[str, str]]] = {}  # case_id -> [(record_id, record_type)]
    for rid in all_unmatched:
        if rid in gt_lookup:
            case_id = gt_lookup[rid][0]
        else:
            case_id = f"__no_gt_{rid}"
        case_groups.setdefault(case_id, [])
        if rid.startswith("PAY"):
            case_groups[case_id].append((rid, "PAYMENT"))
        elif rid.startswith("TXN"):
            case_groups[case_id].append((rid, "BANK_TRANSACTION"))
        elif rid.startswith("INV"):
            case_groups[case_id].append((rid, "INVOICE"))
        else:
            case_groups[case_id].append((rid, "UNKNOWN"))

    review_lookup: Dict[str, Any] = {}
    for r in result.review_candidates:
        lid = r.candidate.left_id
        if lid not in review_lookup or r.confidence_score > review_lookup[lid].confidence_score:
            review_lookup[lid] = r

    # Pick primary record for each group (prefer payment > bank_txn > invoice)
    type_priority = {"PAYMENT": 0, "BANK_TRANSACTION": 1, "INVOICE": 2, "UNKNOWN": 3}

    cases: List[UnmatchedCase] = []
    case_counter = 0

    for group_id, records in case_groups.items():
        records.sort(key=lambda r: type_priority.get(r[1], 9))
        primary_id, primary_type = records[0]

        if primary_type == "PAYMENT":
            df, col = payments, "payment_id"
        elif primary_type == "BANK_TRANSACTION":
            df, col = bank_transactions, "transaction_id"
        else:
            df, col = invoices, "invoice_id"

        source_record = _row_to_dict(df, col, primary_id)
        exception = _find_exception_for_record(result.exceptions, primary_id)
        review = review_lookup.get(primary_id)

        if exception:
            exception_type = exception.exception_type
            description = exception.description
        elif review:
            exception_type = "MANUAL_REVIEW_REQUIRED"
            description = f"Review candidate (confidence: {review.confidence_score:.1f})"
        else:
            exception_type = "NO_MATCH_FOUND"
            description = f"Unmatched: {'; '.join(f'{t} {rid}' for rid, t in records)}"

        potential = _find_nearby_candidates(
            source_record, primary_type, invoices, payments, bank_transactions,
            exclude_ids=already_matched,
            max_candidates=max_candidates_per_case,
        )

        related_ids = [rid for rid, _ in records if rid != primary_id]

        case_counter += 1
        cases.append(UnmatchedCase(
            case_id=f"LLM-{case_counter:04d}",
            exception_type=exception_type,
            record_id=primary_id,
            record_type=primary_type,
            description=description,
            related_ids=sorted(related_ids),
            source_record=source_record,
            potential_candidates=potential,
        ))

    return cases


# ---------------------------------------------------------------------------
# Batch LLM prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a financial reconciliation AI. You examine unmatched transactions "
    "and decide if each should be MATCHED to a candidate or remain an EXCEPTION.\n\n"
    "For each case you will see:\n"
    "- The unmatched record details\n"
    "- Candidate records with amounts, dates, descriptions\n\n"
    "Rules:\n"
    "- MATCH only if amounts are very close (within a few cents or within 5%) AND dates are reasonable.\n"
    "- If amounts differ by more than 5%, or dates are weeks apart, choose EXCEPTION.\n"
    "- Genuine exceptions: orphan records with no close candidate, fraud, data errors.\n"
    "- Return a JSON array with one object per case: "
    '[{"case_id":"...","resolution":"MATCH"|"EXCEPTION","matched_id":"...or null","confidence":0-100,"reason":"..."}]\n'
    "- Return ONLY valid JSON array. No markdown fences, no text outside the JSON."
)


def _format_source_record_for_prompt(record: Dict[str, Any], record_type: str) -> str:
    """Compact one-line summary of the source record."""
    if record_type == "PAYMENT":
        return (
            f"ID={record.get('payment_id','?')} "
            f"gross={record.get('gross_amount','?')} "
            f"fee={record.get('fee','?')} "
            f"net={record.get('net_settled_amount','?')} "
            f"date={record.get('settlement_date','?')} "
            f"linked_inv={record.get('linked_invoice_id','')}"
        )
    elif record_type == "BANK_TRANSACTION":
        return (
            f"ID={record.get('transaction_id','?')} "
            f"amount={record.get('amount','?')} "
            f"date={record.get('date','?')} "
            f"desc={str(record.get('description',''))[:50]} "
            f"ref={record.get('reference_no','')}"
        )
    elif record_type == "INVOICE":
        return (
            f"ID={record.get('invoice_id','?')} "
            f"amount={record.get('expected_amount','?')} "
            f"date={record.get('invoice_date','?')} "
            f"status={record.get('status','?')} "
            f"desc={str(record.get('description',''))[:50]}"
        )
    return str(record)[:120]


def _build_batch_prompt(cases: List[UnmatchedCase]) -> str:
    """Build a compact prompt with all cases in a batch."""
    lines = [
        "Analyze each case below. For each, decide MATCH (with a candidate ID) or EXCEPTION.",
        "",
    ]

    for case in cases:
        lines.append(f"--- Case {case.case_id} ---")
        lines.append(f"Record: {case.record_type} {case.record_id}")
        lines.append(f"Details: {_format_source_record_for_prompt(case.source_record, case.record_type)}")
        lines.append(f"Exception: {case.exception_type} - {case.description[:80]}")

        if case.potential_candidates:
            lines.append("Candidates:")
            for i, cand in enumerate(case.potential_candidates[:4]):
                cand_id = cand.get("id", "?")
                cand_type = cand.get("type", "?")
                cand_amount = cand.get("amount", "?")
                cand_date = cand.get("date", "?")
                cand_desc = str(cand.get("description", ""))[:40]
                diff = cand.get("amount_diff", "?")
                eng_conf = cand.get("engine_confidence", "")
                eng_tag = f" [engine_conf={eng_conf:.0f}]" if eng_conf else ""
                lines.append(
                    f"  [{i}] {cand_type} {cand_id} | amt={cand_amount} | date={cand_date} | diff={diff} | {cand_desc}{eng_tag}"
                )
        else:
            lines.append("Candidates: NONE found")

        lines.append("")

    lines.append(
        'Return JSON array: [{"case_id":"...","resolution":"MATCH"|"EXCEPTION",'
        '"matched_id":"candidate_id or null","confidence":0-100,"reason":"short reason"}]'
    )
    return "\n".join(lines)


def _parse_batch_response(content: str) -> List[Dict[str, Any]]:
    """Extract JSON array from LLM response."""
    text = content.strip()
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            text = match.group()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Main resolution function (batched)
# ---------------------------------------------------------------------------

def resolve_unmatched_with_llm(
    cases: List[UnmatchedCase],
    client: PollinationsClient,
    *,
    min_confidence: float = 50.0,
    batch_size: int = BATCH_SIZE,
    log_path: Path = Path("reports/llm_unmatched_decisions.jsonl"),
) -> List[LLMResolutionDecision]:
    """Send batches of unmatched cases to the LLM and collect decisions."""
    decisions: List[LLMResolutionDecision] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for batch_start in range(0, len(cases), batch_size):
        batch = cases[batch_start:batch_start + batch_size]
        prompt = _build_batch_prompt(batch)

        try:
            message = client.chat([
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ], temperature=0.0)
            content = message.get("content", "")
            parsed_list = _parse_batch_response(content)
        except (LLMClientError, json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning("LLM batch failed at offset %d: %s", batch_start, exc)
            for case in batch:
                decisions.append(LLMResolutionDecision(
                    case_id=case.case_id,
                    record_id=case.record_id,
                    resolution="EXCEPTION",
                    matched_ids=[],
                    confidence=0.0,
                    justification=f"LLM error: {exc}",
                ))
            continue

        parsed_map = {str(p.get("case_id", "")).upper(): p for p in parsed_list}

        for case in batch:
            parsed = parsed_map.get(case.case_id.upper(), {})
            if not parsed:
                for p in parsed_list:
                    if p.get("record_id", "").upper() == case.record_id.upper():
                        parsed = p
                        break

            resolution = str(parsed.get("resolution", "EXCEPTION")).upper()
            matched_id = parsed.get("matched_id") or parsed.get("matched_record_id")
            confidence = _safe_float(parsed.get("confidence")) or 0.0
            justification = str(parsed.get("reason", "") or parsed.get("justification", ""))

            if resolution == "MATCH" and matched_id and confidence >= min_confidence:
                matched_id = str(matched_id).strip().upper()
                if matched_id == "NULL" or matched_id == "NONE":
                    matched_id = None
                    resolution = "EXCEPTION"

            if resolution == "MATCH" and matched_id:
                decisions.append(LLMResolutionDecision(
                    case_id=case.case_id,
                    record_id=case.record_id,
                    resolution="MATCH",
                    matched_ids=[matched_id],
                    confidence=confidence,
                    justification=justification,
                ))
            else:
                decisions.append(LLMResolutionDecision(
                    case_id=case.case_id,
                    record_id=case.record_id,
                    resolution="EXCEPTION",
                    matched_ids=[],
                    confidence=confidence,
                    justification=justification,
                ))

        with log_path.open("a", encoding="utf-8") as log:
            for case in batch:
                d = decisions[case_counter_offset(cases, case, decisions)]
                log.write(json.dumps({
                    "case_id": case.case_id,
                    "record_id": case.record_id,
                    "resolution": d.resolution,
                    "matched_ids": d.matched_ids,
                    "confidence": d.confidence,
                    "justification": d.justification,
                }, default=str) + "\n")

    return decisions


def case_counter_offset(all_cases: List[UnmatchedCase], target: UnmatchedCase, decisions: List[LLMResolutionDecision]) -> int:
    """Find the index in decisions list for a given case."""
    for i in range(len(decisions) - 1, -1, -1):
        if decisions[i].case_id == target.case_id:
            return i
    return len(decisions) - 1


# ---------------------------------------------------------------------------
# Ground truth validation
# ---------------------------------------------------------------------------

def validate_llm_decisions(
    decisions: List[LLMResolutionDecision],
    cases: List[UnmatchedCase],
    ground_truth: pd.DataFrame,
) -> LLMResolutionResult:
    """Compare LLM decisions against ground truth to compute resolved/incorrect counts."""
    gt_map = _build_ground_truth_map(ground_truth)
    case_map = {c.record_id.upper(): c for c in cases}

    details: List[Dict[str, Any]] = []
    resolved = 0
    incorrect = 0
    correct_exception = 0

    for decision in decisions:
        case = case_map.get(decision.record_id.upper())
        if not case:
            continue

        gt = gt_map.get(decision.record_id.upper())
        expected_result = gt.get("expected_result", "UNKNOWN") if gt else "UNKNOWN"
        expected_invoice = gt.get("invoice_id", "") if gt else ""
        expected_payment = gt.get("payment_id", "") if gt else ""
        expected_txn = gt.get("transaction_id", "") if gt else ""

        detail: Dict[str, Any] = {
            "case_id": decision.case_id,
            "record_id": decision.record_id,
            "record_type": case.record_type,
            "exception_type": case.exception_type,
            "llm_resolution": decision.resolution,
            "llm_matched_ids": decision.matched_ids,
            "llm_confidence": decision.confidence,
            "llm_justification": decision.justification,
            "expected_result": expected_result,
            "verdict": "UNKNOWN",
        }

        if expected_result == "MATCH":
            if decision.resolution == "MATCH":
                llm_matched = decision.matched_ids[0] if decision.matched_ids else ""
                expected_partner = _expected_partner_id(case.record_type, expected_invoice, expected_payment, expected_txn)
                if expected_partner and llm_matched.upper() == expected_partner.upper():
                    detail["verdict"] = "CORRECTLY_RESOLVED"
                    resolved += 1
                else:
                    detail["verdict"] = "INCORRECTLY_RESOLVED"
                    incorrect += 1
            else:
                detail["verdict"] = "INCORRECTLY_RESOLVED"
                incorrect += 1
        elif expected_result == "EXCEPTION":
            if decision.resolution == "EXCEPTION":
                detail["verdict"] = "CORRECTLY_EXCEPTION"
                correct_exception += 1
            else:
                detail["verdict"] = "INCORRECTLY_RESOLVED"
                incorrect += 1
        else:
            if decision.resolution == "EXCEPTION":
                detail["verdict"] = "CORRECTLY_EXCEPTION"
                correct_exception += 1
            else:
                detail["verdict"] = "INCORRECTLY_RESOLVED"
                incorrect += 1

        details.append(detail)

    total = len(decisions)
    accuracy = ((resolved + correct_exception) / total * 100.0) if total > 0 else 0.0

    return LLMResolutionResult(
        decisions=decisions,
        total_cases_evaluated=total,
        llm_resolved_count=resolved,
        llm_incorrect_count=incorrect,
        llm_correct_exception_count=correct_exception,
        llm_resolution_accuracy=round(accuracy, 2),
        details=details,
    )


def _build_ground_truth_map(ground_truth: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    gt_map: Dict[str, Dict[str, str]] = {}
    for _, row in ground_truth.iterrows():
        entry = {
            "case_id": str(row.get("case_id", "")),
            "expected_result": str(row.get("expected_result", "")).upper(),
            "invoice_id": str(row.get("invoice_id", "")),
            "payment_id": str(row.get("payment_id", "")),
            "transaction_id": str(row.get("transaction_id", "")),
            "category": str(row.get("category", "")),
        }
        for entity_id in [entry["invoice_id"], entry["payment_id"], entry["transaction_id"]]:
            if entity_id and entity_id.lower() != "nan":
                gt_map[entity_id.upper()] = entry
    return gt_map


def _expected_partner_id(
    record_type: str, invoice_id: str, payment_id: str, transaction_id: str
) -> str:
    if record_type == "PAYMENT":
        if invoice_id and invoice_id.lower() != "nan":
            return invoice_id
        if transaction_id and transaction_id.lower() != "nan":
            return transaction_id
    elif record_type == "BANK_TRANSACTION":
        if payment_id and payment_id.lower() != "nan":
            return payment_id
    elif record_type == "INVOICE":
        if payment_id and payment_id.lower() != "nan":
            return payment_id
    return ""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_llm_evaluation(
    result: ReconciliationResult,
    invoices: pd.DataFrame,
    payments: pd.DataFrame,
    bank_transactions: pd.DataFrame,
    client: PollinationsClient,
    *,
    ground_truth_path: Optional[Path] = None,
    min_confidence: float = 50.0,
    log_path: Path = Path("reports/llm_unmatched_decisions.jsonl"),
) -> LLMResolutionResult:
    """Full orchestrator: build cases, resolve with LLM, validate against ground truth."""
    cases = build_unmatched_cases(
        result, invoices, payments, bank_transactions,
        ground_truth_path=ground_truth_path,
    )

    if not cases:
        return LLMResolutionResult(
            decisions=[],
            total_cases_evaluated=0,
            llm_resolved_count=0,
            llm_incorrect_count=0,
            llm_correct_exception_count=0,
            llm_resolution_accuracy=100.0,
            details=[],
        )

    print(f"  LLM evaluating {len(cases)} unmatched cases (batch size {BATCH_SIZE})...")

    decisions = resolve_unmatched_with_llm(
        cases, client, min_confidence=min_confidence, log_path=log_path,
    )

    ground_truth_df: Optional[pd.DataFrame] = None
    if ground_truth_path and ground_truth_path.exists():
        ground_truth_df = pd.read_csv(ground_truth_path)

    if ground_truth_df is not None:
        return validate_llm_decisions(decisions, cases, ground_truth_df)

    total = len(decisions)
    resolved = sum(1 for d in decisions if d.resolution == "MATCH")
    return LLMResolutionResult(
        decisions=decisions,
        total_cases_evaluated=total,
        llm_resolved_count=resolved,
        llm_incorrect_count=0,
        llm_correct_exception_count=total - resolved,
        llm_resolution_accuracy=0.0,
        details=[],
    )
