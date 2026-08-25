"""
NORMALIZER MODULE

The normalizer standardizes validated reconciliation data before it is passed to
the matching engines.

It operates after loader.py has successfully loaded and validated the CSV data.

Its responsibilities include:

- Standardizing IDs such as invoice IDs, transaction IDs, and payment IDs.
- Normalizing text fields by handling casing, whitespace, and common formatting differences.
- Converting descriptions and references into comparable forms.
- Standardizing monetary values and dates into consistent representations.
- Preserving the original business meaning of the records.

The normalizer does not perform reconciliation or decide whether records match.

Its purpose is to make records from different sources easier to compare.

Pipeline:

CSV Input
    ↓
Loader and Validation
    ↓
Normalizer
    ↓
Deterministic Matching
    ↓
Fuzzy Matching
    ↓
AI Tie-Breaker for unresolved ambiguous cases

The normalizer supports all matching stages, including deterministic matching,
fuzzy candidate scoring, and preparation of clean candidate data for the AI
tie-breaker.
"""

import re
from typing import List, Optional, Tuple, Union
import pandas as pd


def _validate_required_columns(
    df: pd.DataFrame, required_columns: List[str], dataset_name: str
) -> None:
    """Validate that columns required for normalization are present in the DataFrame.

    Raises ValueError if any required columns are missing.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for {dataset_name} normalization: {missing}"
        )


def _normalize_identifier(value: Optional[Union[str, float]]) -> Union[str, float]:
    """Normalize identifier-like values (IDs, references, statuses).

    Strips whitespace, converts to uppercase, and collapses multiple spaces into a single
    space while retaining hyphens and other non-whitespace characters. Preserves missing values
    (pd.NA, None, NaN) without stringifying them into 'NAN' or '<NA>'.
    """
    if pd.isna(value):
        return value

    val_str = str(value).strip()
    if not val_str:
        return pd.NA

    # Convert to upper case and collapse internal repeated whitespace
    val_upper = val_str.upper()
    return re.sub(r"\s+", " ", val_upper)


def _normalize_description(value: Optional[Union[str, float]]) -> Union[str, float]:
    """Normalize textual descriptions for downstream lexical matching.

    Converts to lowercase, strips leading/trailing whitespace, replaces common separators
    ('-', '_', '/') with spaces, normalizes punctuation, and collapses internal spaces.
    Preserves missing values.
    """
    if pd.isna(value):
        return value

    val_str = str(value).strip().lower()
    if not val_str:
        return pd.NA

    # Replace common separators with spaces
    val_clean = re.sub(r"[-_/]", " ", val_str)

    # Remove unexpected punctuation while preserving alphanumerics and whitespace
    val_clean = re.sub(r"[^\w\s]", "", val_clean)

    # Collapse multiple whitespaces
    return re.sub(r"\s+", " ", val_clean).strip()


def _normalize_amount(value: Optional[Union[int, float, str]]) -> Union[int, float]:
    """Convert monetary float amounts to integer minor units (e.g. cents/pence).

    Multiplies the amount by 100 and rounds to the nearest integer to prevent
    floating-point precision issues during deterministic comparison. Preserves missing values.
    Example: 1000.00 -> 100000, 999.50 -> 99950.
    """
    if pd.isna(value):
        return value

    try:
        float_val = float(value)
        return int(round(float_val * 100))
    except (ValueError, TypeError):
        return pd.NA


def normalize_invoices(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize an invoices DataFrame by adding standardized helper columns.

    Required input columns:
        - invoice_id
        - expected_amount
        - status
        - invoice_date
        - customer_ref
        - description

    Returns a new DataFrame containing all original columns plus normalized helper columns:
        - invoice_id_normalized
        - expected_amount_normalized
        - status_normalized
        - invoice_date_normalized
        - customer_ref_normalized
        - description_normalized
    """
    required_cols = [
        "invoice_id",
        "expected_amount",
        "status",
        "invoice_date",
        "customer_ref",
        "description",
    ]
    _validate_required_columns(df, required_cols, "invoice")

    out_df = df.copy()

    out_df["invoice_id_normalized"] = out_df["invoice_id"].apply(_normalize_identifier)
    out_df["expected_amount_normalized"] = out_df["expected_amount"].apply(
        _normalize_amount
    )
    out_df["status_normalized"] = out_df["status"].apply(_normalize_identifier)
    out_df["invoice_date_normalized"] = pd.to_datetime(
        out_df["invoice_date"], errors="coerce"
    ).dt.normalize()
    out_df["customer_ref_normalized"] = out_df["customer_ref"].apply(
        _normalize_identifier
    )
    out_df["description_normalized"] = out_df["description"].apply(
        _normalize_description
    )

    return out_df


def normalize_bank_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a bank transactions DataFrame by adding standardized helper columns.

    Required input columns:
        - transaction_id
        - amount
        - date
        - description
        - reference_no

    Returns a new DataFrame containing all original columns plus normalized helper columns:
        - transaction_id_normalized
        - amount_normalized
        - date_normalized
        - description_normalized
        - reference_no_normalized
    """
    required_cols = [
        "transaction_id",
        "amount",
        "date",
        "description",
        "reference_no",
    ]
    _validate_required_columns(df, required_cols, "bank_transactions")

    out_df = df.copy()

    out_df["transaction_id_normalized"] = out_df["transaction_id"].apply(
        _normalize_identifier
    )
    out_df["amount_normalized"] = out_df["amount"].apply(_normalize_amount)
    out_df["date_normalized"] = pd.to_datetime(
        out_df["date"], errors="coerce"
    ).dt.normalize()
    out_df["description_normalized"] = out_df["description"].apply(
        _normalize_description
    )
    out_df["reference_no_normalized"] = out_df["reference_no"].apply(
        _normalize_identifier
    )

    return out_df


def normalize_payments(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a payments DataFrame by adding standardized helper columns.

    Required input columns:
        - payment_id
        - gross_amount
        - fee
        - net_settled_amount
        - settlement_date
        - linked_invoice_id

    Returns a new DataFrame containing all original columns plus normalized helper columns:
        - payment_id_normalized
        - gross_amount_normalized
        - fee_normalized
        - net_settled_amount_normalized
        - settlement_date_normalized
        - linked_invoice_id_normalized
    """
    required_cols = [
        "payment_id",
        "gross_amount",
        "fee",
        "net_settled_amount",
        "settlement_date",
        "linked_invoice_id",
    ]
    _validate_required_columns(df, required_cols, "payments")

    out_df = df.copy()

    out_df["payment_id_normalized"] = out_df["payment_id"].apply(_normalize_identifier)
    out_df["gross_amount_normalized"] = out_df["gross_amount"].apply(_normalize_amount)
    out_df["fee_normalized"] = out_df["fee"].apply(_normalize_amount)
    out_df["net_settled_amount_normalized"] = out_df["net_settled_amount"].apply(
        _normalize_amount
    )
    out_df["settlement_date_normalized"] = pd.to_datetime(
        out_df["settlement_date"], errors="coerce"
    ).dt.normalize()
    out_df["linked_invoice_id_normalized"] = out_df["linked_invoice_id"].apply(
        _normalize_identifier
    )

    return out_df


def normalize_all(
    invoices: pd.DataFrame,
    bank_transactions: pd.DataFrame,
    payments: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalize all three datasets (invoices, bank transactions, payments) in batch.

    Calls individual normalization functions and returns normalized copies of all three DataFrames
    without mutating the input arguments.
    """
    normalized_invoices = normalize_invoices(invoices)
    normalized_bank_txns = normalize_bank_transactions(bank_transactions)
    normalized_payments = normalize_payments(payments)

    return normalized_invoices, normalized_bank_txns, normalized_payments

# Place at the very end of app/data/normalizer.py

def normalize_all(
    invoices: pd.DataFrame,
    bank_transactions: pd.DataFrame,
    payments: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalize all three datasets (invoices, bank transactions, payments) in batch."""
    normalized_invoices = normalize_invoices(invoices)
    normalized_bank_txns = normalize_bank_transactions(bank_transactions)
    normalized_payments = normalize_payments(payments)

    return normalized_invoices, normalized_bank_txns, normalized_payments


# Alias for backward compatibility with downstream modules like reconcile.py
normalize_datasets = normalize_all