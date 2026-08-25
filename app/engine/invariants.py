"""Cross-dataset financial and business invariant validation engine.

This module validates logical, date, and financial consistency among normalized
invoices, payments, and bank transactions. It operates downstream of loader.py and
normalizer.py, focusing purely on relationship validation without mutating inputs or
duplicating single-row structural checks.

All monetary comparisons are evaluated strictly in minor units (integers).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import pandas as pd


@dataclass(frozen=True)
class InvariantResult:
    """Represents the outcome of a single invariant validation check."""

    passed: bool
    rule: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchInvariantSummary:
    """Summary of all invariant checks performed on a candidate match set."""

    all_passed: bool
    results: List[InvariantResult]

    @property
    def failures(self) -> List[InvariantResult]:
        """Return only failed invariant checks."""
        return [res for res in self.results if not res.passed]


def _extract_val(row_or_df: Union[pd.Series, dict], field_name: str) -> Any:
    """Extract a value safely from a pd.Series or dict, raising KeyError if missing."""
    if isinstance(row_or_df, pd.Series):
        if field_name not in row_or_df.index:
            raise KeyError(
                f"Required normalized column '{field_name}' not found in Series/DataFrame index."
            )
        return row_or_df[field_name]
    elif isinstance(row_or_df, dict):
        if field_name not in row_or_df:
            raise KeyError(f"Required key '{field_name}' not found in input dictionary.")
        return row_or_df[field_name]
    else:
        raise TypeError(f"Expected pd.Series or dict, got {type(row_or_df).__name__}")


def check_date_window(
    earlier_date: Any,
    later_date: Any,
    max_difference_days: int = 7,
    allow_future_dates: bool = False,
    rule_name: str = "DATE_WINDOW_CHECK",
) -> InvariantResult:
    """Validate if two dates fall within an acceptable calendar day window.

    Args:
        earlier_date: The expected earlier date (Timestamp, str, or datetime).
        later_date: The expected later date (Timestamp, str, or datetime).
        max_difference_days: Maximum allowed difference in calendar days.
        allow_future_dates: If True, later_date can occur before earlier_date within
            max_difference_days.
        rule_name: Custom rule identifier for the result payload.

    Returns:
        InvariantResult indicating whether the date window constraint was satisfied.
    """
    ts_earlier = pd.to_datetime(earlier_date, errors="coerce")
    ts_later = pd.to_datetime(later_date, errors="coerce")

    if pd.isna(ts_earlier) or pd.isna(ts_later):
        return InvariantResult(
            passed=False,
            rule=rule_name,
            message="Invalid or missing date timestamp for date window comparison.",
            details={
                "earlier_date": str(earlier_date),
                "later_date": str(later_date),
            },
        )

    # Normalize to midnight for strict calendar day calculation
    dt1 = ts_earlier.normalize()
    dt2 = ts_later.normalize()
    day_diff = (dt2 - dt1).days

    if day_diff < 0 and not allow_future_dates:
        return InvariantResult(
            passed=False,
            rule=rule_name,
            message=(
                f"Chronological anomaly: later event date ({dt2.strftime('%Y-%m-%d')}) "
                f"precedes earlier event date ({dt1.strftime('%Y-%m-%d')})."
            ),
            details={
                "earlier_date": dt1.strftime("%Y-%m-%d"),
                "later_date": dt2.strftime("%Y-%m-%d"),
                "day_difference": day_diff,
            },
        )

    abs_day_diff = abs(day_diff)
    if abs_day_diff > max_difference_days:
        return InvariantResult(
            passed=False,
            rule=rule_name,
            message=(
                f"Date gap of {abs_day_diff} day(s) exceeds maximum allowed window "
                f"of {max_difference_days} day(s)."
            ),
            details={
                "earlier_date": dt1.strftime("%Y-%m-%d"),
                "later_date": dt2.strftime("%Y-%m-%d"),
                "day_difference": day_diff,
                "abs_day_difference": abs_day_diff,
                "max_difference_days": max_difference_days,
            },
        )

    return InvariantResult(
        passed=True,
        rule=rule_name,
        message=f"Dates fall within valid window of {abs_day_diff} day(s).",
        details={
            "earlier_date": dt1.strftime("%Y-%m-%d"),
            "later_date": dt2.strftime("%Y-%m-%d"),
            "day_difference": day_diff,
            "max_difference_days": max_difference_days,
        },
    )


def check_payment_invoice_link_invariant(
    payment: Union[pd.Series, dict],
    invoices: pd.DataFrame,
) -> InvariantResult:
    """Validate that if a payment declares a linked invoice, that invoice exists in the dataset.

    Args:
        payment: Single payment record (Series or dict).
        invoices: Full or normalized Invoices DataFrame.

    Returns:
        InvariantResult indicating if explicit link integrity is satisfied.
    """
    pay_id = _extract_val(payment, "payment_id_normalized")
    linked_inv_id = _extract_val(payment, "linked_invoice_id_normalized")

    if pd.isna(linked_inv_id) or str(linked_inv_id).strip() == "":
        return InvariantResult(
            passed=True,
            rule="PAYMENT_INVOICE_EXPLICIT_LINK",
            message="Payment has no explicit linked invoice ID specified; skipping reference check.",
            details={"payment_id": pay_id, "linked_invoice_id": None},
        )

    linked_inv_id_str = str(linked_inv_id).strip()

    # Determine index column name in invoices dataset
    col_to_check = (
        "invoice_id_normalized"
        if "invoice_id_normalized" in invoices.columns
        else "invoice_id"
    )

    exists = (
        invoices[col_to_check].astype(str).str.strip() == linked_inv_id_str
    ).any()

    if not exists:
        return InvariantResult(
            passed=False,
            rule="PAYMENT_INVOICE_EXPLICIT_LINK",
            message=(
                f"Payment '{pay_id}' references linked_invoice_id '{linked_inv_id_str}', "
                f"which does not exist in invoices dataset."
            ),
            details={
                "payment_id": pay_id,
                "linked_invoice_id": linked_inv_id_str,
                "found_in_invoices": False,
            },
        )

    return InvariantResult(
        passed=True,
        rule="PAYMENT_INVOICE_EXPLICIT_LINK",
        message=f"Payment '{pay_id}' correctly references existing invoice '{linked_inv_id_str}'.",
        details={
            "payment_id": pay_id,
            "linked_invoice_id": linked_inv_id_str,
            "found_in_invoices": True,
        },
    )


def check_payment_bank_invariant(
    payment: Union[pd.Series, dict],
    bank_transaction: Union[pd.Series, dict],
    amount_tolerance_minor: int = 0,
    max_date_difference_days: int = 7,
) -> List[InvariantResult]:
    """Validate financial amount and settlement date consistency between payment and bank transaction.

    Args:
        payment: Single payment record (Series or dict).
        bank_transaction: Single bank transaction record (Series or dict).
        amount_tolerance_minor: Allowed difference in minor units (e.g. cents). Default = 0.
        max_date_difference_days: Max calendar day difference between settlement and bank dates.

    Returns:
        List of InvariantResult checks covering amount match and date window validation.
    """
    results = []

    pay_id = _extract_val(payment, "payment_id_normalized")
    pay_net = _extract_val(payment, "net_settled_amount_normalized")
    pay_date = _extract_val(payment, "settlement_date_normalized")

    txn_id = _extract_val(bank_transaction, "transaction_id_normalized")
    txn_amount = _extract_val(bank_transaction, "amount_normalized")
    txn_date = _extract_val(bank_transaction, "date_normalized")

    # 1. Net Settlement Amount vs Bank Transaction Amount
    if pd.isna(pay_net) or pd.isna(txn_amount):
        results.append(
            InvariantResult(
                passed=False,
                rule="PAYMENT_BANK_AMOUNT_MATCH",
                message="Missing required numeric values for payment/bank amount comparison.",
                details={
                    "payment_id": pay_id,
                    "bank_transaction_id": txn_id,
                    "payment_net_settled_amount": pay_net,
                    "bank_amount": txn_amount,
                },
            )
        )
    else:
        diff_minor = abs(int(pay_net) - int(txn_amount))
        amt_passed = diff_minor <= amount_tolerance_minor

        results.append(
            InvariantResult(
                passed=amt_passed,
                rule="PAYMENT_BANK_AMOUNT_MATCH",
                message=(
                    f"Payment net amount ({pay_net}) matches bank amount ({txn_amount}) "
                    f"within tolerance {amount_tolerance_minor} (diff = {diff_minor})."
                    if amt_passed
                    else f"Amount discrepancy: payment net amount ({pay_net}) vs bank transaction "
                    f"amount ({txn_amount}) exceeds tolerance {amount_tolerance_minor} (diff = {diff_minor})."
                ),
                details={
                    "payment_id": pay_id,
                    "bank_transaction_id": txn_id,
                    "payment_net_settled_amount": pay_net,
                    "bank_amount": txn_amount,
                    "difference_minor": diff_minor,
                    "tolerance_minor": amount_tolerance_minor,
                },
            )
        )

    # 2. Settlement Date vs Bank Transaction Date
    date_res = check_date_window(
        earlier_date=pay_date,
        later_date=txn_date,
        max_difference_days=max_date_difference_days,
        allow_future_dates=True,
        rule_name="PAYMENT_BANK_DATE_WINDOW",
    )
    results.append(
        InvariantResult(
            passed=date_res.passed,
            rule=date_res.rule,
            message=f"Payment/Bank date window check: {date_res.message}",
            details={
                "payment_id": pay_id,
                "bank_transaction_id": txn_id,
                **date_res.details,
            },
        )
    )

    return results


def check_invoice_payment_invariant(
    invoice: Union[pd.Series, dict],
    payments: Union[pd.DataFrame, List[Union[pd.Series, dict]]],
    amount_tolerance_minor: int = 0,
    max_date_difference_days: int = 60,
) -> List[InvariantResult]:
    """Validate expected amount vs aggregated payment gross amounts and settlement date windows.

    Handles exact settlement, partial payments (underpayment), and overpayments without failing
    the invariant check automatically for partial payments.

    Args:
        invoice: Single invoice record (Series or dict).
        payments: Single or list/DataFrame of associated payment records.
        amount_tolerance_minor: Allowed difference in minor units. Default = 0.
        max_date_difference_days: Maximum days allowed between invoice date and payment settlement.

    Returns:
        List of InvariantResult checks detailing settlement status, amount balance, and date checks.
    """
    results = []

    inv_id = _extract_val(invoice, "invoice_id_normalized")
    expected_amount = _extract_val(invoice, "expected_amount_normalized")
    inv_date = _extract_val(invoice, "invoice_date_normalized")

    # Normalize payments container into list of pd.Series or dicts
    if isinstance(payments, pd.DataFrame):
        pay_list = [row for _, row in payments.iterrows()]
    elif isinstance(payments, (pd.Series, dict)):
        pay_list = [payments]
    elif isinstance(payments, list):
        pay_list = payments
    else:
        raise TypeError(
            f"Unsupported type for payments input: {type(payments).__name__}"
        )

    if not pay_list:
        results.append(
            InvariantResult(
                passed=False,
                rule="INVOICE_PAYMENT_SETTLEMENT_STATUS",
                message=f"No payment records provided for invoice '{inv_id}'.",
                details={
                    "invoice_id": inv_id,
                    "expected_amount": expected_amount,
                    "total_gross_paid": 0,
                },
            )
        )
        return results

    total_gross_paid = 0
    date_failures = 0

    for pay in pay_list:
        g_amt = _extract_val(pay, "gross_amount_normalized")
        if pd.notna(g_amt):
            total_gross_paid += int(g_amt)

        pay_date = _extract_val(pay, "settlement_date_normalized")
        d_res = check_date_window(
            earlier_date=inv_date,
            later_date=pay_date,
            max_difference_days=max_date_difference_days,
            allow_future_dates=False,
            rule_name="INVOICE_PAYMENT_DATE_WINDOW",
        )
        if not d_res.passed:
            date_failures += 1
            results.append(
                InvariantResult(
                    passed=False,
                    rule=d_res.rule,
                    message=f"Invoice '{inv_id}' payment date anomaly: {d_res.message}",
                    details={
                        "invoice_id": inv_id,
                        "payment_id": _extract_val(pay, "payment_id_normalized"),
                        **d_res.details,
                    },
                )
            )

    if date_failures == 0:
        results.append(
            InvariantResult(
                passed=True,
                rule="INVOICE_PAYMENT_DATE_WINDOW",
                message=f"All payment settlement dates fall within valid window for invoice '{inv_id}'.",
                details={"invoice_id": inv_id, "payment_count": len(pay_list)},
            )
        )

    # Balance evaluation
    diff_minor = int(expected_amount) - total_gross_paid

    if abs(diff_minor) <= amount_tolerance_minor:
        status = "EXACT_SETTLEMENT"
        msg = f"Invoice '{inv_id}' is fully and exactly settled."
        passed = True
    elif diff_minor > 0:
        status = "UNDERPAYMENT"
        msg = f"Invoice '{inv_id}' is partially paid (remaining balance: {diff_minor} minor units)."
        passed = True  # Partial payment is logically valid
    else:
        status = "OVERPAYMENT"
        over_amount = abs(diff_minor)
        msg = f"Invoice '{inv_id}' is overpaid by {over_amount} minor units."
        passed = False  # Overpayment breaks balance invariant

    results.append(
        InvariantResult(
            passed=passed,
            rule="INVOICE_PAYMENT_SETTLEMENT_STATUS",
            message=msg,
            details={
                "invoice_id": inv_id,
                "expected_amount": expected_amount,
                "total_gross_paid": total_gross_paid,
                "remaining_balance": diff_minor,
                "settlement_status": status,
                "tolerance_minor": amount_tolerance_minor,
            },
        )
    )

    return results


def validate_match_invariants(
    invoice: Optional[Union[pd.Series, dict]],
    payment: Optional[Union[pd.Series, dict]],
    bank_transaction: Optional[Union[pd.Series, dict]],
    all_invoices: Optional[pd.DataFrame] = None,
    amount_tolerance_minor: int = 0,
    max_payment_bank_date_days: int = 7,
    max_invoice_payment_date_days: int = 60,
) -> MatchInvariantSummary:
    """Orchestrate all cross-dataset invariant checks for a proposed match candidate set.

    Args:
        invoice: Optional invoice record in the candidate match set.
        payment: Optional payment record in the candidate match set.
        bank_transaction: Optional bank transaction record in the candidate match set.
        all_invoices: Full dataset of invoices (used to verify explicitly linked payment references).
        amount_tolerance_minor: Absolute minor units tolerance allowed across monetary checks.
        max_payment_bank_date_days: Allowed days between payment settlement and bank date.
        max_invoice_payment_date_days: Allowed days between invoice date and payment settlement.

    Returns:
        MatchInvariantSummary containing all individual result checks and overall pass/fail boolean.
    """
    all_results: List[InvariantResult] = []

    # 1. Payment -> Invoice Explicit Link Verification
    if payment is not None and all_invoices is not None:
        link_res = check_payment_invoice_link_invariant(payment, all_invoices)
        all_results.append(link_res)

    # 2. Payment <-> Bank Transaction Checks
    if payment is not None and bank_transaction is not None:
        pay_bank_res = check_payment_bank_invariant(
            payment=payment,
            bank_transaction=bank_transaction,
            amount_tolerance_minor=amount_tolerance_minor,
            max_date_difference_days=max_payment_bank_date_days,
        )
        all_results.extend(pay_bank_res)

    # 3. Invoice <-> Payment Balance and Date Checks
    if invoice is not None and payment is not None:
        inv_pay_res = check_invoice_payment_invariant(
            invoice=invoice,
            payments=payment,
            amount_tolerance_minor=amount_tolerance_minor,
            max_date_difference_days=max_invoice_payment_date_days,
        )
        all_results.extend(inv_pay_res)

    # Evaluate overall status: True if no invariant check failed
    overall_passed = all(res.passed for res in all_results) if all_results else True

    return MatchInvariantSummary(all_passed=overall_passed, results=all_results)