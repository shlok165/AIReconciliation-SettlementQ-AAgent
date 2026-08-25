"""Reconciliation orchestration layer for multi-source financial reconciliation system.

Connects normalization, deterministic matching, fuzzy matching, multi-signal scoring,
relationship-aware conflict resolution, and safety guards into an end-to-end pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set
import pandas as pd

from app.data.normalizer import (
    normalize_datasets,
)
from app.engine.deterministic import (
    AmbiguousMatch,
    DeterministicMatch as ConfirmedMatch,
    DeterministicMatchResult,
    run_deterministic_matching,
)

from app.engine.fuzzy import (
    FuzzyMatchResult as FuzzyResult,
    run_fuzzy_matching,
)

from app.engine.scoring import ScoredCandidate, ScoringResult, score_fuzzy_candidates


@dataclass(frozen=True)
class ExceptionRecord:
    """Structured exception item detailing why a record remains unresolved or under review."""
    exception_type: str
    record_id: str
    record_type: str
    description: str
    related_ids: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReconciliationMetrics:
    """Quantitative metrics summarizing pipeline performance and reconciliation coverage."""
    total_invoices: int
    total_payments: int
    total_bank_transactions: int

    deterministic_confirmed_matches: int
    fuzzy_auto_matches: int
    manual_review_candidates: int
    rejected_fuzzy_candidates: int

    unresolved_payments: int
    unresolved_bank_transactions: int
    unresolved_invoices: int

    payment_match_rate: float
    bank_transaction_match_rate: float
    invoice_match_rate: float


@dataclass
class ReconciliationResult:
    """Complete structured output of the reconciliation pipeline."""
    deterministic_result: DeterministicMatchResult
    fuzzy_result: FuzzyResult
    scoring_result: ScoringResult

    confirmed_payment_bank_matches: List[ConfirmedMatch]
    confirmed_invoice_payment_matches: List[ConfirmedMatch]
    
    auto_matches: List[ScoredCandidate]
    review_candidates: List[ScoredCandidate]
    rejected_candidates: List[ScoredCandidate]

    unresolved_payment_ids: List[str]
    unresolved_bank_transaction_ids: List[str]
    unresolved_invoice_ids: List[str]

    exceptions: List[ExceptionRecord]
    metrics: ReconciliationMetrics


def reconcile(
    invoices: pd.DataFrame,
    payments: pd.DataFrame,
    bank_transactions: pd.DataFrame,
    *,
    amount_tolerance_minor: int = 100,
    max_date_difference_days: int = 7,
    text_weight: float = 0.50,
    amount_weight: float = 0.30,
    date_weight: float = 0.20,
    auto_match_threshold: float = 90.0,
    review_threshold: float = 70.0,
    minimum_score_margin: float = 5.0,
    include_ambiguous_as_unresolved: bool = True,
    require_three_way_consistency: bool = True,
) -> ReconciliationResult:
    """Execute end-to-end multi-source financial reconciliation pipeline."""
    # 1. Normalization (normalize_datasets returns norm_inv, norm_bank, norm_pay)
    norm_inv, norm_bank, norm_pay = normalize_datasets(
        invoices=invoices,
        bank_transactions=bank_transactions,
        payments=payments,
    )

    # 2. Deterministic Matching
    det_result = run_deterministic_matching(
        invoices=norm_inv,
        payments=norm_pay,
        bank_transactions=norm_bank,
    )

    # 3. Fuzzy Matching
    fuzzy_result = run_fuzzy_matching(
        invoices=norm_inv,
        payments=norm_pay,
        bank_transactions=norm_bank,
        deterministic_result=det_result,
        max_amount_difference_minor=amount_tolerance_minor,
        max_date_difference_days=max_date_difference_days,
        include_ambiguous_as_unresolved=include_ambiguous_as_unresolved,
    )

    # 4. Multi-Signal Scoring
    all_fuzzy_candidates = (
        fuzzy_result.payment_bank_candidates + fuzzy_result.invoice_payment_candidates
    )
    scoring_result = score_fuzzy_candidates(
        candidates=all_fuzzy_candidates,
        text_weight=text_weight,
        amount_weight=amount_weight,
        date_weight=date_weight,
        max_amount_difference_minor=amount_tolerance_minor,
        max_date_difference_days=max_date_difference_days,
        auto_match_threshold=auto_match_threshold,
        review_threshold=review_threshold,
        minimum_score_margin=minimum_score_margin,
    )

    # 5. Relationship-Aware Conflict Resolution with Unified ID Orientations
    # Convention: 
    #   - PAYMENT_BANK: left_id = payment_id, right_id = transaction_id
    #   - INVOICE_PAYMENT: left_id = payment_id, right_id = invoice_id
    pay_to_bank_matches: Dict[str, List[ScoredCandidate]] = {}
    bank_to_pay_matches: Dict[str, List[ScoredCandidate]] = {}
    pay_to_inv_matches: Dict[str, List[ScoredCandidate]] = {}
    inv_to_pay_matches: Dict[str, List[ScoredCandidate]] = {}

    for scored in scoring_result.auto_matches:
        cand = scored.candidate
        if "PAYMENT_BANK" in cand.match_type:
            pay_to_bank_matches.setdefault(cand.left_id, []).append(scored)
            bank_to_pay_matches.setdefault(cand.right_id, []).append(scored)
        elif "INVOICE" in cand.match_type:
            pay_to_inv_matches.setdefault(cand.left_id, []).append(scored)
            inv_to_pay_matches.setdefault(cand.right_id, []).append(scored)

    accepted_auto_matches: List[ScoredCandidate] = []
    conflict_review_candidates: List[ScoredCandidate] = list(scoring_result.review_candidates)
    exceptions: List[ExceptionRecord] = []

    for amb in det_result.ambiguous_matches:
        rec_type = getattr(amb, "source_type", "PAYMENT")
        exceptions.append(
            ExceptionRecord(
                exception_type="AMBIGUOUS_DETERMINISTIC_MATCH",
                record_id=amb.source_id,
                record_type=rec_type,
                description=f"Deterministic matching found multiple exact matches for source {amb.source_id}: {amb.reason}",
                related_ids=list(amb.candidate_ids),
            )
        )

    for scored in scoring_result.auto_matches:
        cand = scored.candidate
        has_conflict = False
        reason = ""

        if "PAYMENT_BANK" in cand.match_type:
            if len(pay_to_bank_matches.get(cand.left_id, [])) > 1:
                has_conflict = True
                reason = f"Payment {cand.left_id} is assigned to multiple auto-matched bank transactions."
            elif len(bank_to_pay_matches.get(cand.right_id, [])) > 1:
                has_conflict = True
                reason = f"Bank transaction {cand.right_id} is assigned to multiple auto-matched payments."

        elif "INVOICE" in cand.match_type:
            if len(pay_to_inv_matches.get(cand.left_id, [])) > 1:
                has_conflict = True
                reason = f"Payment {cand.left_id} is assigned to multiple auto-matched invoices."
            elif len(inv_to_pay_matches.get(cand.right_id, [])) > 1:
                has_conflict = True
                reason = f"Invoice {cand.right_id} is assigned to multiple auto-matched payments."

        if has_conflict:
            conflict_item = ScoredCandidate(
                candidate=cand,
                confidence_score=scored.confidence_score,
                text_score=scored.text_score,
                amount_score=scored.amount_score,
                date_score=scored.date_score,
                decision="REVIEW",
                reasons=scored.reasons + [reason],
            )
            conflict_review_candidates.append(conflict_item)

            rec_id = cand.left_id
            rec_type = "PAYMENT"

            exceptions.append(
                ExceptionRecord(
                    exception_type="CONFLICTING_AUTO_MATCH",
                    record_id=rec_id,
                    record_type=rec_type,
                    description=reason,
                    related_ids=[cand.right_id],
                )
            )
        else:
            accepted_auto_matches.append(scored)

    # 6. ID Extraction for Review Candidates
    reviewed_payment_ids: Set[str] = set()
    reviewed_bank_ids: Set[str] = set()
    reviewed_invoice_ids: Set[str] = set()

    for rev in conflict_review_candidates:
        c = rev.candidate
        rec_id = c.left_id
        rec_type = "PAYMENT"
        rel_id = c.right_id
        reviewed_payment_ids.add(c.left_id)
        if "PAYMENT_BANK" in c.match_type:
            reviewed_bank_ids.add(c.right_id)
        else:
            reviewed_invoice_ids.add(c.right_id)

        if not any(e.record_id == rec_id and e.record_type == rec_type for e in exceptions):
            exceptions.append(
                ExceptionRecord(
                    exception_type="MANUAL_REVIEW_REQUIRED",
                    record_id=rec_id,
                    record_type=rec_type,
                    description=f"Fuzzy candidate requires manual review (confidence: {rev.confidence_score:.1f}). Reasons: {'; '.join(rev.reasons)}",
                    related_ids=[rel_id],
                )
            )

    # 7. 3-Way Consistency Guard & Final Unresolved ID Calculation
    # If require_three_way_consistency is enabled and invoices are present,
    # payment-bank matches are only confirmed if the payment has a validated invoice match.
    # Otherwise, the settlement is held pending 3-way invoice resolution.
    confirmed_invoice_payment_ids: Set[str] = set()
    for m in det_result.invoice_payment_matches:
        confirmed_invoice_payment_ids.add(m.left_id)
    for auto in accepted_auto_matches:
        c = auto.candidate
        if "INVOICE" in c.match_type:
            confirmed_invoice_payment_ids.add(c.left_id)

    tot_inv_count = len(norm_inv["invoice_id_normalized"].dropna()) if "invoice_id_normalized" in norm_inv.columns else 0

    confirmed_pb_matches: List[ConfirmedMatch] = []
    held_pb_matches: List[Tuple[ConfirmedMatch, str]] = []

    for m in det_result.payment_bank_matches:
        if not require_three_way_consistency or tot_inv_count == 0 or m.left_id in confirmed_invoice_payment_ids:
            confirmed_pb_matches.append(m)
        else:
            held_pb_matches.append((
                m,
                f"Payment {m.left_id} matched bank transaction {m.right_id} but lacks a confirmed invoice match (held pending 3-way invoice resolution)."
            ))

    for m, reason in held_pb_matches:
        exceptions.append(
            ExceptionRecord(
                exception_type="UNRESOLVED_INVOICE_DEPENDENCY",
                record_id=m.left_id,
                record_type="PAYMENT",
                description=reason,
                related_ids=[m.right_id],
            )
        )

    final_accepted_auto_matches: List[ScoredCandidate] = []
    for auto in accepted_auto_matches:
        c = auto.candidate
        if "PAYMENT_BANK" in c.match_type:
            if not require_three_way_consistency or tot_inv_count == 0 or c.left_id in confirmed_invoice_payment_ids:
                final_accepted_auto_matches.append(auto)
            else:
                exceptions.append(
                    ExceptionRecord(
                        exception_type="UNRESOLVED_INVOICE_DEPENDENCY",
                        record_id=c.left_id,
                        record_type="PAYMENT",
                        description=f"Auto-matched payment {c.left_id} to bank transaction {c.right_id} held pending 3-way invoice resolution.",
                        related_ids=[c.right_id],
                    )
                )
        else:
            final_accepted_auto_matches.append(auto)

    confirmed_payment_ids: Set[str] = set()
    confirmed_bank_ids: Set[str] = set()
    confirmed_invoice_ids: Set[str] = set()

    for m in confirmed_pb_matches:
        confirmed_payment_ids.add(m.left_id)
        confirmed_bank_ids.add(m.right_id)
    for m in det_result.invoice_payment_matches:
        confirmed_payment_ids.add(m.left_id)
        confirmed_invoice_ids.add(m.right_id)

    for auto in final_accepted_auto_matches:
        c = auto.candidate
        if "PAYMENT_BANK" in c.match_type:
            confirmed_payment_ids.add(c.left_id)
            confirmed_bank_ids.add(c.right_id)
        elif "INVOICE" in c.match_type:
            confirmed_payment_ids.add(c.left_id)
            confirmed_invoice_ids.add(c.right_id)

    all_invoice_ids = set(norm_inv["invoice_id_normalized"].dropna().astype(str).unique()) if "invoice_id_normalized" in norm_inv.columns else set()
    all_payment_ids = set(norm_pay["payment_id_normalized"].dropna().astype(str).unique()) if "payment_id_normalized" in norm_pay.columns else set()
    all_bank_ids = set(norm_bank["transaction_id_normalized"].dropna().astype(str).unique()) if "transaction_id_normalized" in norm_bank.columns else set()

    final_unresolved_payment_ids = sorted(list(all_payment_ids - confirmed_payment_ids))
    final_unresolved_bank_ids = sorted(list(all_bank_ids - confirmed_bank_ids))
    final_unresolved_invoice_ids = sorted(list(all_invoice_ids - confirmed_invoice_ids))

    # 8. Type-Aware Exception Deduplication for NO_MATCH_FOUND
    already_reported_exceptions = {
        (e.record_type, e.record_id)
        for e in exceptions
    }

    held_bank_ids: Set[str] = set()
    for e in exceptions:
        if e.exception_type == "UNRESOLVED_INVOICE_DEPENDENCY":
            held_bank_ids.update(e.related_ids)

    for p_id in final_unresolved_payment_ids:
        if p_id not in reviewed_payment_ids and ("PAYMENT", p_id) not in already_reported_exceptions:
            exceptions.append(
                ExceptionRecord(
                    exception_type="NO_MATCH_FOUND",
                    record_id=p_id,
                    record_type="PAYMENT",
                    description=f"Payment {p_id} could not be matched by deterministic or fuzzy engine.",
                )
            )

    for b_id in final_unresolved_bank_ids:
        if b_id not in reviewed_bank_ids and b_id not in held_bank_ids and ("BANK_TRANSACTION", b_id) not in already_reported_exceptions:
            exceptions.append(
                ExceptionRecord(
                    exception_type="NO_MATCH_FOUND",
                    record_id=b_id,
                    record_type="BANK_TRANSACTION",
                    description=f"Bank transaction {b_id} could not be matched by deterministic or fuzzy engine.",
                )
            )

    for i_id in final_unresolved_invoice_ids:
        if i_id not in reviewed_invoice_ids and ("INVOICE", i_id) not in already_reported_exceptions:
            exceptions.append(
                ExceptionRecord(
                    exception_type="NO_MATCH_FOUND",
                    record_id=i_id,
                    record_type="INVOICE",
                    description=f"Invoice {i_id} could not be matched by deterministic or fuzzy engine.",
                )
            )

    # 9. Structured Metrics Calculation
    tot_inv = len(all_invoice_ids)
    tot_pay = len(all_payment_ids)
    tot_bank = len(all_bank_ids)

    det_matches_count = len(confirmed_pb_matches) + len(det_result.invoice_payment_matches)
    auto_matches_count = len(final_accepted_auto_matches)

    matched_payments_count = len(confirmed_payment_ids)
    matched_bank_count = len(confirmed_bank_ids)
    matched_invoice_count = len(confirmed_invoice_ids)

    payment_match_rate = (matched_payments_count / tot_pay * 100.0) if tot_pay > 0 else 0.0
    bank_match_rate = (matched_bank_count / tot_bank * 100.0) if tot_bank > 0 else 0.0
    invoice_match_rate = (matched_invoice_count / tot_inv * 100.0) if tot_inv > 0 else 0.0

    metrics = ReconciliationMetrics(
        total_invoices=tot_inv,
        total_payments=tot_pay,
        total_bank_transactions=tot_bank,
        deterministic_confirmed_matches=det_matches_count,
        fuzzy_auto_matches=auto_matches_count,
        manual_review_candidates=len(conflict_review_candidates),
        rejected_fuzzy_candidates=len(scoring_result.rejected_candidates),
        unresolved_payments=len(final_unresolved_payment_ids),
        unresolved_bank_transactions=len(final_unresolved_bank_ids),
        unresolved_invoices=len(final_unresolved_invoice_ids),
        payment_match_rate=float(round(payment_match_rate, 2)),
        bank_transaction_match_rate=float(round(bank_match_rate, 2)),
        invoice_match_rate=float(round(invoice_match_rate, 2)),
    )

    return ReconciliationResult(
        deterministic_result=det_result,
        fuzzy_result=fuzzy_result,
        scoring_result=scoring_result,
        confirmed_payment_bank_matches=confirmed_pb_matches,
        confirmed_invoice_payment_matches=det_result.invoice_payment_matches,
        auto_matches=final_accepted_auto_matches,
        review_candidates=conflict_review_candidates,
        rejected_candidates=scoring_result.rejected_candidates,
        unresolved_payment_ids=final_unresolved_payment_ids,
        unresolved_bank_transaction_ids=final_unresolved_bank_ids,
        unresolved_invoice_ids=final_unresolved_invoice_ids,
        exceptions=exceptions,
        metrics=metrics,
    )