"""Data loader module for raw reconciliation and settlement datasets.

This module resolves paths dynamically relative to its location and handles
loading, schema verification, strict type conversions, missing value rejection,
uniqueness checks, and domain invariant validations with detailed row-level error reporting.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import pandas as pd


class DataValidationError(Exception):
    """Raised when dataset loading or schema validation fails."""

    pass


def _get_raw_data_dir() -> Path:
    """Resolve raw data directory relative to loader.py location."""
    return Path(__file__).resolve().parent.parent.parent / "data" / "raw"


def _read_csv_strictly(file_path: Path) -> pd.DataFrame:
    """Read CSV file with strict error handling for malformed lines and empty files."""
    if not file_path.is_file():
        raise DataValidationError(f"File not found: {file_path}")

    try:
        df = pd.read_csv(file_path, dtype=str, on_bad_lines="error")
    except Exception as exc:
        raise DataValidationError(
            f"Failed to parse CSV file '{file_path.name}': {exc}"
        ) from exc

    if df.empty or len(df.columns) == 0:
        raise DataValidationError(f"Dataset '{file_path.name}' is empty.")

    df.columns = [str(col).strip() for col in df.columns]
    return df


def _check_required_columns(
    df: pd.DataFrame, file_name: str, required_columns: List[str]
) -> None:
    """Validate that all required columns exist in the DataFrame header."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise DataValidationError(
            f"{file_name} missing required column(s): {missing}"
        )


def _validate_duplicate_ids(
    df: pd.DataFrame, file_name: str, id_col: str
) -> List[str]:
    """Validate that primary IDs are strictly unique within the dataset."""
    errors = []
    # Identify duplicated ID values excluding NaN/null
    duplicated_mask = df[id_col].duplicated(keep=False) & df[id_col].notna()
    if duplicated_mask.any():
        duplicated_ids = df.loc[duplicated_mask, id_col].unique()
        for dup_id in duplicated_ids:
            indices = df.index[df[id_col] == dup_id].tolist()
            csv_rows = [idx + 2 for idx in indices]
            errors.append(
                f"Duplicate {id_col} '{dup_id}' found at CSV row(s): {csv_rows}"
            )
    return errors


def _validate_string_column(
    df: pd.DataFrame, col: str, id_col: str, allow_empty: bool = False
) -> List[str]:
    """Validate string columns, enforcing non-empty values unless explicitly allowed."""
    errors = []
    if allow_empty:
        return errors

    invalid_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
    bad_indices = df.index[invalid_mask].tolist()
    for idx in bad_indices:
        csv_row = idx + 2
        row_id = (
            df.loc[idx, id_col]
            if id_col in df.columns and pd.notna(df.loc[idx, id_col])
            else f"Index {idx}"
        )
        errors.append(
            f"Row {csv_row} ({id_col}: {row_id}): missing or empty required text in column '{col}'"
        )
    return errors


def _validate_numeric_column(
    df: pd.DataFrame, col: str, id_col: str
) -> Tuple[List[str], pd.Series]:
    """Validate numeric fields distinguishing missing values from invalid formats."""
    errors = []
    numeric_series = pd.to_numeric(df[col], errors="coerce")

    # 1. Check for missing/empty values
    missing_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
    for idx in df.index[missing_mask].tolist():
        csv_row = idx + 2
        row_id = (
            df.loc[idx, id_col]
            if id_col in df.columns and pd.notna(df.loc[idx, id_col])
            else f"Index {idx}"
        )
        errors.append(
            f"Row {csv_row} ({id_col}: {row_id}): missing mandatory numeric value in column '{col}'"
        )

    # 2. Check for invalid non-numeric strings
    invalid_mask = numeric_series.isna() & ~missing_mask
    for idx in df.index[invalid_mask].tolist():
        csv_row = idx + 2
        row_id = (
            df.loc[idx, id_col]
            if id_col in df.columns and pd.notna(df.loc[idx, id_col])
            else f"Index {idx}"
        )
        val = df.loc[idx, col]
        errors.append(
            f"Row {csv_row} ({id_col}: {row_id}): invalid numeric value '{val}' in column '{col}'"
        )

    return errors, numeric_series


def _validate_date_column(
    df: pd.DataFrame, col: str, id_col: str
) -> Tuple[List[str], pd.Series]:
    """Validate date fields distinguishing missing values from invalid formats."""
    errors = []
    date_series = pd.to_datetime(df[col], errors="coerce")

    # 1. Check for missing/empty values
    missing_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
    for idx in df.index[missing_mask].tolist():
        csv_row = idx + 2
        row_id = (
            df.loc[idx, id_col]
            if id_col in df.columns and pd.notna(df.loc[idx, id_col])
            else f"Index {idx}"
        )
        errors.append(
            f"Row {csv_row} ({id_col}: {row_id}): missing mandatory date in column '{col}'"
        )

    # 2. Check for invalid date formats
    invalid_mask = date_series.isna() & ~missing_mask
    for idx in df.index[invalid_mask].tolist():
        csv_row = idx + 2
        row_id = (
            df.loc[idx, id_col]
            if id_col in df.columns and pd.notna(df.loc[idx, id_col])
            else f"Index {idx}"
        )
        val = df.loc[idx, col]
        errors.append(
            f"Row {csv_row} ({id_col}: {row_id}): invalid date format '{val}' in column '{col}'"
        )

    return errors, date_series


def load_invoices(
    data_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Load and validate invoices.csv."""
    raw_dir = Path(data_dir) if data_dir else _get_raw_data_dir()
    file_path = raw_dir / "invoices.csv"

    df = _read_csv_strictly(file_path)

    required_columns = [
        "invoice_id",
        "expected_amount",
        "status",
        "invoice_date",
        "customer_ref",
        "description",
    ]
    _check_required_columns(df, "invoices.csv", required_columns)
    df = df[required_columns].copy()

    error_messages = []

    # Validate primary ID uniqueness
    error_messages.extend(
        _validate_duplicate_ids(df, "invoices.csv", "invoice_id")
    )

    # Validate mandatory string columns
    for col in ["invoice_id", "status", "customer_ref", "description"]:
        error_messages.extend(
            _validate_string_column(df, col, id_col="invoice_id")
        )

    # Validate mandatory numeric column
    num_errs, expected_amount_series = _validate_numeric_column(
        df, "expected_amount", id_col="invoice_id"
    )
    error_messages.extend(num_errs)

    # Validate mandatory date column
    date_errs, invoice_date_series = _validate_date_column(
        df, "invoice_date", id_col="invoice_id"
    )
    error_messages.extend(date_errs)

    if error_messages:
        raise DataValidationError(
            f"invoices.csv validation failed with {len(error_messages)} error(s):\n"
            + "\n".join(f"  - {err}" for err in error_messages)
        )

    df["expected_amount"] = expected_amount_series
    df["invoice_date"] = invoice_date_series
    return df


def load_bank_transactions(
    data_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Load and validate bank_transactions.csv."""
    raw_dir = Path(data_dir) if data_dir else _get_raw_data_dir()
    file_path = raw_dir / "bank_transactions.csv"

    df = _read_csv_strictly(file_path)

    required_columns = [
        "transaction_id",
        "amount",
        "date",
        "description",
        "reference_no",
    ]
    _check_required_columns(df, "bank_transactions.csv", required_columns)
    df = df[required_columns].copy()

    error_messages = []

    # Validate primary ID uniqueness
    error_messages.extend(
        _validate_duplicate_ids(df, "bank_transactions.csv", "transaction_id")
    )

    # Validate mandatory string columns
    for col in ["transaction_id", "description", "reference_no"]:
        error_messages.extend(
            _validate_string_column(df, col, id_col="transaction_id")
        )

    # Validate mandatory numeric column
    num_errs, amount_series = _validate_numeric_column(
        df, "amount", id_col="transaction_id"
    )
    error_messages.extend(num_errs)

    # Validate mandatory date column
    date_errs, date_series = _validate_date_column(
        df, "date", id_col="transaction_id"
    )
    error_messages.extend(date_errs)

    if error_messages:
        raise DataValidationError(
            f"bank_transactions.csv validation failed with {len(error_messages)} error(s):\n"
            + "\n".join(f"  - {err}" for err in error_messages)
        )

    df["amount"] = amount_series
    df["date"] = date_series
    return df


def load_payments(
    data_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Load and validate payments.csv."""
    raw_dir = Path(data_dir) if data_dir else _get_raw_data_dir()
    file_path = raw_dir / "payments.csv"

    df = _read_csv_strictly(file_path)

    required_columns = [
        "payment_id",
        "gross_amount",
        "fee",
        "net_settled_amount",
        "settlement_date",
        "linked_invoice_id",
    ]
    _check_required_columns(df, "payments.csv", required_columns)
    df = df[required_columns].copy()

    error_messages = []

    # Validate primary ID uniqueness
    error_messages.extend(
        _validate_duplicate_ids(df, "payments.csv", "payment_id")
    )

    # Validate mandatory string column (linked_invoice_id allowed to be empty/NaN)
    error_messages.extend(
        _validate_string_column(
            df, "payment_id", id_col="payment_id", allow_empty=False
        )
    )
    error_messages.extend(
        _validate_string_column(
            df, "linked_invoice_id", id_col="payment_id", allow_empty=True
        )
    )

    # Validate mandatory numeric columns
    numeric_results = {}
    for col in ["gross_amount", "fee", "net_settled_amount"]:
        num_errs, series = _validate_numeric_column(
            df, col, id_col="payment_id"
        )
        error_messages.extend(num_errs)
        numeric_results[col] = series

    # Validate mandatory date column
    date_errs, settlement_date_series = _validate_date_column(
        df, "settlement_date", id_col="payment_id"
    )
    error_messages.extend(date_errs)

    # Validate payment financial invariant: gross_amount - fee == net_settled_amount
    gross = numeric_results["gross_amount"]
    fee = numeric_results["fee"]
    net = numeric_results["net_settled_amount"]

    # Evaluate invariant only on rows where all three numeric fields parsed successfully
    valid_numeric_rows_mask = gross.notna() & fee.notna() & net.notna()
    for idx in df.index[valid_numeric_rows_mask].tolist():
        g_val = gross.loc[idx]
        f_val = fee.loc[idx]
        n_val = net.loc[idx]
        expected_net = g_val - f_val

        if abs(expected_net - n_val) > 0.01:
            csv_row = idx + 2
            pay_id = df.loc[idx, "payment_id"]
            error_messages.append(
                f"Row {csv_row} (payment_id: {pay_id}): financial invariant failed "
                f"(gross_amount {g_val:.2f} - fee {f_val:.2f} = {expected_net:.2f}, "
                f"but net_settled_amount is {n_val:.2f})"
            )

    if error_messages:
        raise DataValidationError(
            f"payments.csv validation failed with {len(error_messages)} error(s):\n"
            + "\n".join(f"  - {err}" for err in error_messages)
        )

    df["gross_amount"] = gross
    df["fee"] = fee
    df["net_settled_amount"] = net
    df["settlement_date"] = settlement_date_series
    return df


def load_all_data(
    data_dir: Optional[Union[str, Path]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and return all raw datasets cleanly."""
    invoices = load_invoices(data_dir=data_dir)
    bank_transactions = load_bank_transactions(data_dir=data_dir)
    payments = load_payments(data_dir=data_dir)
    return invoices, bank_transactions, payments


if __name__ == "__main__":
    try:
        invoices, bank_transactions, payments = load_all_data()
        print(
            f"Successfully loaded {len(invoices)} invoices, "
            f"{len(bank_transactions)} bank txns, and {len(payments)} payments."
        )
    except DataValidationError as e:
        print(f"\n{e}")