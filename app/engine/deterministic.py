"""Deterministic matching engine for reconciliation and settlement datasets.

This module performs high-confidence, rule-based deterministic matching across normalized
invoices, payments, and bank transactions. It identifies exact explicit links and unambiguous
exact-amount/date matches while cleanly identifying ambiguous candidate sets and unmatched records.

It consumes normalized DataFrames and leverages invariant helpers from app.engine.invariants.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple
import pandas as pd

from app.engine.invariants import check_date_window


@dataclass(frozen=True)
class DeterministicMatch:
    """Represents a single confirmed deterministic match between two records."""

    match_type: str  # e.g., 'EXPLICIT_LINK', 'EXACT_AMOUNT_DATE', 'REFERENCE_EQUALS'
    left_id: str
    right_id: str
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AmbiguousMatch:
    """Represents a record that matched multiple candidate records ambiguously."""

    source_id: str
    source_type: str  # 'PAYMENT' or 'BANK_TRANSACTION'
    candidate_ids: List[str]
    reason: str


@dataclass
class DeterministicMatchResult:
    """Container for all outcomes of the deterministic matching pipeline."""

    invoice_payment_matches: List[DeterministicMatch] = field(default_factory=list)
    payment_bank_matches: List[DeterministicMatch] = field(default_factory=list)
    unmatched_invoice_ids: List[str] = field(default_factory=list)
    unmatched_payment_ids: List[str] = field(default_factory=list)
    unmatched_bank_transaction_ids: List[str] = field(default_factory=list)
    ambiguous_matches: List[AmbiguousMatch] = field(default_factory=list)


def _validate_columns(df: pd.DataFrame, required_cols: List[str], dataset_name: str) -> None:
    """Validate that required normalized columns are present in the DataFrame."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required normalized columns for {dataset_name} in deterministic matching: {missing}"
        )


def match_payments_to_invoices(
    invoices: pd.DataFrame,
    payments: pd.DataFrame,
    amount_tolerance_minor: int = 0,
    max_date_difference_days: int = 7,
) -> Tuple[List[DeterministicMatch], List[str], List[str]]:
    """Perform deterministic matching between payments and invoices using explicit links with invariant validation."""
    _validate_columns(invoices, ["invoice_id_normalized"], "Invoices")
    _validate_columns(
        payments, ["payment_id_normalized", "linked_invoice_id_normalized"], "Payments"
    )

    matches: List[DeterministicMatch] = []
    matched_invoice_ids: Set[str] = set()
    matched_payment_ids: Set[str] = set()

    all_invoice_ids = set(invoices["invoice_id_normalized"].dropna().astype(str))
    all_payment_ids = set(payments["payment_id_normalized"].dropna().astype(str))

    invoice_lookup = {
        str(row["invoice_id_normalized"]): row for _, row in invoices.iterrows()
    }

    # Group payments by linked invoice
    payments_by_inv: Dict[str, List[pd.Series]] = {}
    for _, pay_row in payments.iterrows():
        linked_inv_id = pay_row.get("linked_invoice_id_normalized")
        if pd.notna(linked_inv_id) and str(linked_inv_id).strip():
            payments_by_inv.setdefault(str(linked_inv_id).strip(), []).append(pay_row)

    for linked_inv_str, pay_rows in payments_by_inv.items():
        if linked_inv_str not in invoice_lookup:
            continue

        inv_row = invoice_lookup[linked_inv_str]
        inv_amount = inv_row.get("expected_amount_normalized")
        if pd.isna(inv_amount):
            inv_amount = inv_row.get("gross_amount_normalized")
        inv_date = inv_row.get("invoice_date_normalized")

        # Check total payments amount vs invoice amount
        total_pay_amount = 0
        has_amount_info = pd.notna(inv_amount)
        for p in pay_rows:
            p_amount = p.get("gross_amount_normalized")
            if pd.isna(p_amount):
                p_amount = p.get("net_settled_amount_normalized")
            if pd.notna(p_amount):
                total_pay_amount += int(p_amount)
            else:
                has_amount_info = False

        if has_amount_info:
            diff_minor = abs(int(total_pay_amount) - int(inv_amount))
            if diff_minor > amount_tolerance_minor:
                continue

        # Check date window for each payment
        dates_valid = True
        for p in pay_rows:
            pay_date = p.get("settlement_date_normalized")
            if pd.notna(pay_date) and pd.notna(inv_date):
                date_res = check_date_window(
                    earlier_date=inv_date,
                    later_date=pay_date,
                    max_difference_days=max_date_difference_days,
                    allow_future_dates=True,
                    rule_name="DETERMINISTIC_INVOICE_PAYMENT_DATE",
                )
                if not date_res.passed:
                    dates_valid = False
                    break

        if not dates_valid:
            continue

        # All invariants passed for this invoice and its linked payment(s)
        for p in pay_rows:
            pay_id = str(p["payment_id_normalized"])
            matches.append(
                DeterministicMatch(
                    match_type="PAYMENT_INVOICE_EXPLICIT_LINK",
                    left_id=pay_id,
                    right_id=linked_inv_str,
                    evidence={
                        "linked_invoice_id": linked_inv_str,
                        "rule": "Exact string equality on payment.linked_invoice_id_normalized with invariant checks",
                    },
                )
            )
            matched_payment_ids.add(pay_id)
        matched_invoice_ids.add(linked_inv_str)

    unmatched_payment_ids = sorted(list(all_payment_ids - matched_payment_ids))
    unmatched_invoice_ids = sorted(list(all_invoice_ids - matched_invoice_ids))

    return matches, unmatched_invoice_ids, unmatched_payment_ids


def match_payments_to_bank_transactions(
    payments: pd.DataFrame,
    bank_transactions: pd.DataFrame,
    amount_tolerance_minor: int = 0,
    max_date_difference_days: int = 7,
) -> Tuple[List[DeterministicMatch], List[AmbiguousMatch], List[str], List[str]]:
    """Match payments to bank transactions deterministically using exact amounts and date windows."""
    _validate_columns(
        payments,
        [
            "payment_id_normalized",
            "net_settled_amount_normalized",
            "settlement_date_normalized",
        ],
        "Payments",
    )
    _validate_columns(
        bank_transactions,
        [
            "transaction_id_normalized",
            "amount_normalized",
            "date_normalized",
        ],
        "Bank Transactions",
    )

    candidates_per_payment: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    candidates_per_bank: Dict[str, List[str]] = {}

    payment_lookup = {
        str(row["payment_id_normalized"]): row for _, row in payments.iterrows()
    }
    bank_lookup = {
        str(row["transaction_id_normalized"]): row
        for _, row in bank_transactions.iterrows()
    }

    all_payment_ids = set(payment_lookup.keys())
    all_bank_ids = set(bank_lookup.keys())

    for pay_id, pay_row in payment_lookup.items():
        candidates_per_payment[pay_id] = []
        pay_net = pay_row["net_settled_amount_normalized"]
        pay_date = pay_row["settlement_date_normalized"]

        if pd.isna(pay_net) or pd.isna(pay_date):
            continue

        for bank_id, bank_row in bank_lookup.items():
            bank_amt = bank_row["amount_normalized"]
            bank_date = bank_row["date_normalized"]

            if pd.isna(bank_amt) or pd.isna(bank_date):
                continue

            diff_minor = abs(int(pay_net) - int(bank_amt))
            if diff_minor > amount_tolerance_minor:
                continue

            date_res = check_date_window(
                earlier_date=pay_date,
                later_date=bank_date,
                max_difference_days=max_date_difference_days,
                allow_future_dates=True,
                rule_name="DETERMINISTIC_PAYMENT_BANK_DATE",
            )
            if not date_res.passed:
                continue

            evidence = {
                "payment_net_settled_amount": pay_net,
                "bank_amount": bank_amt,
                "amount_difference_minor": diff_minor,
                "payment_settlement_date": str(pay_date),
                "bank_transaction_date": str(bank_date),
                "day_difference": date_res.details.get("day_difference"),
            }
            candidates_per_payment[pay_id].append((bank_id, evidence))

            if bank_id not in candidates_per_bank:
                candidates_per_bank[bank_id] = []
            candidates_per_bank[bank_id].append(pay_id)

    confirmed_matches: List[DeterministicMatch] = []
    ambiguous_matches: List[AmbiguousMatch] = []

    matched_payment_ids: Set[str] = set()
    matched_bank_ids: Set[str] = set()
    ambiguous_payment_ids: Set[str] = set()
    ambiguous_bank_ids: Set[str] = set()

    for pay_id, cand_list in candidates_per_payment.items():
        if len(cand_list) > 1:
            cand_bank_ids = [c[0] for c in cand_list]
            ambiguous_matches.append(
                AmbiguousMatch(
                    source_id=pay_id,
                    source_type="PAYMENT",
                    candidate_ids=cand_bank_ids,
                    reason=f"Payment matched {len(cand_bank_ids)} bank transactions with identical amount and date constraints.",
                )
            )
            ambiguous_payment_ids.add(pay_id)
            for b_id in cand_bank_ids:
                ambiguous_bank_ids.add(b_id)

    for bank_id, pay_cand_ids in candidates_per_bank.items():
        if len(pay_cand_ids) > 1:
            ambiguous_matches.append(
                AmbiguousMatch(
                    source_id=bank_id,
                    source_type="BANK_TRANSACTION",
                    candidate_ids=pay_cand_ids,
                    reason=f"Bank transaction matched {len(pay_cand_ids)} payments with identical amount and date constraints.",
                )
            )
            ambiguous_bank_ids.add(bank_id)
            for p_id in pay_cand_ids:
                ambiguous_payment_ids.add(p_id)

    for pay_id, cand_list in candidates_per_payment.items():
        if len(cand_list) == 1:
            bank_id, evidence = cand_list[0]
            if len(candidates_per_bank.get(bank_id, [])) == 1:
                if pay_id not in ambiguous_payment_ids and bank_id not in ambiguous_bank_ids:
                    confirmed_matches.append(
                        DeterministicMatch(
                            match_type="PAYMENT_BANK_EXACT_AMOUNT_DATE",
                            left_id=pay_id,
                            right_id=bank_id,
                            evidence=evidence,
                        )
                    )
                    matched_payment_ids.add(pay_id)
                    matched_bank_ids.add(bank_id)

    unmatched_payment_ids = sorted(
        list(all_payment_ids - matched_payment_ids - ambiguous_payment_ids)
    )
    unmatched_bank_ids = sorted(
        list(all_bank_ids - matched_bank_ids - ambiguous_bank_ids)
    )

    return (
        confirmed_matches,
        ambiguous_matches,
        unmatched_payment_ids,
        unmatched_bank_ids,
    )


def match_bank_reference_to_identifiers(
    payments: pd.DataFrame,
    bank_transactions: pd.DataFrame,
    excluded_payment_ids: Set[str],
    excluded_bank_ids: Set[str],
) -> Tuple[List[DeterministicMatch], Set[str], Set[str]]:
    """Optional reference matching using exact equality between bank reference and payment IDs.

    Strictly ignores both already matched AND ambiguous IDs to avoid overriding multi-candidate ties.
    """
    _validate_columns(payments, ["payment_id_normalized"], "Payments")
    _validate_columns(
        bank_transactions,
        ["transaction_id_normalized", "reference_no_normalized"],
        "Bank Transactions",
    )

    ref_matches: List[DeterministicMatch] = []
    updated_pay_ids = set(excluded_payment_ids)
    updated_bank_ids = set(excluded_bank_ids)

    avail_payments = payments[
        ~payments["payment_id_normalized"].astype(str).isin(excluded_payment_ids)
    ]
    avail_banks = bank_transactions[
        ~bank_transactions["transaction_id_normalized"]
        .astype(str)
        .isin(excluded_bank_ids)
    ]

    pay_id_map = {
        str(row["payment_id_normalized"]): row
        for _, row in avail_payments.iterrows()
        if pd.notna(row["payment_id_normalized"])
    }

    for _, bank_row in avail_banks.iterrows():
        bank_id = str(bank_row["transaction_id_normalized"])
        bank_ref = bank_row["reference_no_normalized"]

        if pd.isna(bank_ref) or str(bank_ref).strip() == "":
            continue

        bank_ref_str = str(bank_ref).strip()

        if bank_ref_str in pay_id_map and bank_id not in updated_bank_ids:
            pay_id = bank_ref_str
            if pay_id not in updated_pay_ids:
                ref_matches.append(
                    DeterministicMatch(
                        match_type="BANK_REFERENCE_EXACT_PAYMENT_ID",
                        left_id=pay_id,
                        right_id=bank_id,
                        evidence={
                            "bank_reference_no": bank_ref_str,
                            "matched_payment_id": pay_id,
                            "rule": "Exact string equality between bank reference and payment ID",
                        },
                    )
                )
                updated_pay_ids.add(pay_id)
                updated_bank_ids.add(bank_id)

    return ref_matches, updated_pay_ids, updated_bank_ids


def run_deterministic_matching(
    invoices: pd.DataFrame,
    payments: pd.DataFrame,
    bank_transactions: pd.DataFrame,
    amount_tolerance_minor: int = 0,
    max_date_difference_days: int = 7,
    enable_reference_matching: bool = True,
) -> DeterministicMatchResult:
    """Orchestrate the full deterministic matching stage across all datasets."""
    # 1. Invoice <-> Payment Explicit Links
    inv_pay_matches, unmatched_inv_ids, _ = match_payments_to_invoices(
        invoices=invoices,
        payments=payments,
        amount_tolerance_minor=amount_tolerance_minor,
        max_date_difference_days=max_date_difference_days,
    )

    # 2. Payment <-> Bank Exact Amount/Date Matches
    pay_bank_matches, ambiguous_matches, unmatched_pay_ids, unmatched_bank_ids = (
        match_payments_to_bank_transactions(
            payments=payments,
            bank_transactions=bank_transactions,
            amount_tolerance_minor=amount_tolerance_minor,
            max_date_difference_days=max_date_difference_days,
        )
    )

    consumed_pay_ids = {m.left_id for m in pay_bank_matches}
    consumed_bank_ids = {m.right_id for m in pay_bank_matches}

    ambiguous_pay_ids = {
        a.source_id for a in ambiguous_matches if a.source_type == "PAYMENT"
    }
    ambiguous_bank_ids = {
        a.source_id
        for a in ambiguous_matches
        if a.source_type == "BANK_TRANSACTION"
    }

    # 3. Optional Reference Matching (Strictly excludes both consumed and ambiguous IDs)
    if enable_reference_matching:
        excluded_pay_ids = consumed_pay_ids | ambiguous_pay_ids
        excluded_bank_ids = consumed_bank_ids | ambiguous_bank_ids

        ref_matches, final_excluded_pay_ids, final_excluded_bank_ids = (
            match_bank_reference_to_identifiers(
                payments=payments,
                bank_transactions=bank_transactions,
                excluded_payment_ids=excluded_pay_ids,
                excluded_bank_ids=excluded_bank_ids,
            )
        )
        if ref_matches:
            pay_bank_matches.extend(ref_matches)

            all_pay_ids = set(
                payments["payment_id_normalized"].dropna().astype(str)
            )
            all_bank_ids = set(
                bank_transactions["transaction_id_normalized"].dropna().astype(str)
            )

            unmatched_pay_ids = sorted(
                list(all_pay_ids - final_excluded_pay_ids)
            )
            unmatched_bank_ids = sorted(
                list(all_bank_ids - final_excluded_bank_ids)
            )

    return DeterministicMatchResult(
        invoice_payment_matches=inv_pay_matches,
        payment_bank_matches=pay_bank_matches,
        unmatched_invoice_ids=unmatched_inv_ids,
        unmatched_payment_ids=unmatched_pay_ids,
        unmatched_bank_transaction_ids=unmatched_bank_ids,
        ambiguous_matches=ambiguous_matches,
    )