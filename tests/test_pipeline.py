"""Integration tests for the financial reconciliation pipeline.

Tests end-to-end integration across:
    loader.py -> normalizer.py -> engine/invariants.py -> engine/deterministic.py
"""
import sys
from pathlib import Path

# Ensures 'app' can be imported during pytest discovery
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from typing import Dict
import pandas as pd
import pytest

from app.data.loader import load_all_data_from_files
from app.data.normalizer import normalize_all
from app.engine.invariants import check_date_window
from app.engine.deterministic import run_deterministic_matching


@pytest.fixture
def sample_csv_files(tmp_path) -> Dict[str, str]:
    """Helper fixture to generate temporary CSV files for testing loader integration."""
    inv_path = tmp_path / "invoices.csv"
    pay_path = tmp_path / "payments.csv"
    bank_path = tmp_path / "bank.csv"

    # Raw CSV data adhering to typical loader field definitions
    inv_df = pd.DataFrame(
        [
            {
                "invoice_id": "INV_101",
                "expected_amount": 1000.00,
                "status": "open",
                "invoice_date": "2026-08-01",
                "customer_ref": "CUST_A",
                "description": "Invoice 101",
            },
            {
                "invoice_id": "INV_102",
                "expected_amount": 500.00,
                "status": "open",
                "invoice_date": "2026-08-02",
                "customer_ref": "CUST_B",
                "description": "Invoice 102",
            },
        ]
    )

    pay_df = pd.DataFrame(
        [
            {
                "payment_id": "PAY_201",
                "linked_invoice_id": "INV_101",
                "gross_amount": 1000.00,
                "fee": 0.00,
                "net_settled_amount": 1000.00,
                "settlement_date": "2026-08-02",
            },
            {
                "payment_id": "PAY_202",
                "linked_invoice_id": "",
                "gross_amount": 500.00,
                "fee": 0.00,
                "net_settled_amount": 500.00,
                "settlement_date": "2026-08-03",
            },
        ]
    )

    bank_df = pd.DataFrame(
        [
            {
                "transaction_id": "BANK_301",
                "amount": 1000.00,
                "date": "2026-08-03",
                "description": "Settlement 301",
                "reference_no": "REF_NONE",
            },
            {
                "transaction_id": "BANK_302",
                "amount": 500.00,
                "date": "2026-08-04",
                "description": "Settlement 302",
                "reference_no": "REF_NONE",
            },
        ]
    )

    inv_df.to_csv(inv_path, index=False)
    pay_df.to_csv(pay_path, index=False)
    bank_df.to_csv(bank_path, index=False)

    return {
        "invoices_path": str(inv_path),
        "payments_path": str(pay_path),
        "bank_transactions_path": str(bank_path),
    }


def test_end_to_end_successful_deterministic_pipeline(sample_csv_files):
    """1. Test end-to-end flow from CSV load through normalization to deterministic matches."""
    # Step 1: Load CSV files
    loaded_invoices, loaded_bank, loaded_payments = load_all_data_from_files(
        invoices_file=sample_csv_files["invoices_path"],
        bank_transactions_file=sample_csv_files["bank_transactions_path"],
        payments_file=sample_csv_files["payments_path"],
    )

    # Step 2: Normalize
    norm_invoices, norm_bank, norm_payments = normalize_all(
        invoices=loaded_invoices,
        bank_transactions=loaded_bank,
        payments=loaded_payments,
    )

    # Step 3: Run Deterministic Engine
    result = run_deterministic_matching(
        invoices=norm_invoices,
        payments=norm_payments,
        bank_transactions=norm_bank,
        amount_tolerance_minor=0,
        max_date_difference_days=7,
    )

    # Assert Invoice <-> Payment explicit match
    assert len(result.invoice_payment_matches) == 1
    inv_match = result.invoice_payment_matches[0]
    assert inv_match.match_type == "PAYMENT_INVOICE_EXPLICIT_LINK"
    assert inv_match.left_id == "PAY_201"
    assert inv_match.right_id == "INV_101"

    # Assert Payment <-> Bank exact amount/date match
    assert len(result.payment_bank_matches) == 2
    match_pairs = {(m.left_id, m.right_id) for m in result.payment_bank_matches}
    assert ("PAY_201", "BANK_301") in match_pairs
    assert ("PAY_202", "BANK_302") in match_pairs

    # Assert no residual unmatched records or ambiguous matches
    assert len(result.unmatched_invoice_ids) == 1
    assert result.unmatched_invoice_ids[0] == "INV_102"
    assert len(result.unmatched_payment_ids) == 0
    assert len(result.unmatched_bank_transaction_ids) == 0
    assert len(result.ambiguous_matches) == 0


def test_partial_payment_scenario():
    """2. Test multiple payments linking to a single invoice (Partial Payments)."""
    invoices = pd.DataFrame(
        [
            {
                "invoice_id_normalized": "INV_PARTIAL",
                "gross_amount_normalized": 1000000,  # $10,000.00 in minor units
                "invoice_date_normalized": pd.Timestamp("2026-08-01"),
            }
        ]
    )

    payments = pd.DataFrame(
        [
            {
                "payment_id_normalized": "PAY_P1",
                "linked_invoice_id_normalized": "INV_PARTIAL",
                "net_settled_amount_normalized": 600000,
                "settlement_date_normalized": pd.Timestamp("2026-08-02"),
            },
            {
                "payment_id_normalized": "PAY_P2",
                "linked_invoice_id_normalized": "INV_PARTIAL",
                "net_settled_amount_normalized": 400000,
                "settlement_date_normalized": pd.Timestamp("2026-08-03"),
            },
        ]
    )

    bank_transactions = pd.DataFrame(
        columns=[
            "transaction_id_normalized",
            "amount_normalized",
            "date_normalized",
            "reference_no_normalized",
        ]
    )

    result = run_deterministic_matching(
        invoices=invoices,
        payments=payments,
        bank_transactions=bank_transactions,
    )

    # Confirm both partial payments linked to the same invoice
    assert len(result.invoice_payment_matches) == 2
    linked_inv_ids = {m.right_id for m in result.invoice_payment_matches}
    linked_pay_ids = {m.left_id for m in result.invoice_payment_matches}

    assert linked_inv_ids == {"INV_PARTIAL"}
    assert linked_pay_ids == {"PAY_P1", "PAY_P2"}


def test_unmatched_records():
    """3. Test standalone payments and bank transactions end up in unmatched sets."""
    invoices = pd.DataFrame(
        [
            {
                "invoice_id_normalized": "INV_ORPHAN",
                "gross_amount_normalized": 10000,
                "invoice_date_normalized": pd.Timestamp("2026-08-01"),
            }
        ]
    )

    payments = pd.DataFrame(
        [
            {
                "payment_id_normalized": "PAY_UNMATCHED",
                "linked_invoice_id_normalized": None,
                "net_settled_amount_normalized": 88888,
                "settlement_date_normalized": pd.Timestamp("2026-08-01"),
            }
        ]
    )

    bank_transactions = pd.DataFrame(
        [
            {
                "transaction_id_normalized": "BANK_UNMATCHED",
                "amount_normalized": 99999,
                "date_normalized": pd.Timestamp("2026-08-01"),
                "reference_no_normalized": None,
            }
        ]
    )

    result = run_deterministic_matching(
        invoices=invoices,
        payments=payments,
        bank_transactions=bank_transactions,
    )

    assert "PAY_UNMATCHED" in result.unmatched_payment_ids
    assert "BANK_UNMATCHED" in result.unmatched_bank_transaction_ids
    assert "INV_ORPHAN" in result.unmatched_invoice_ids
    assert len(result.payment_bank_matches) == 0


def test_ambiguous_payment_bank_match():
    """4. Test 1:N amount/date tie produces an AmbiguousMatch and no confirmed match."""
    invoices = pd.DataFrame(columns=["invoice_id_normalized"])

    payments = pd.DataFrame(
        [
            {
                "payment_id_normalized": "PAY_AMBIGUOUS",
                "linked_invoice_id_normalized": None,
                "net_settled_amount_normalized": 500000,
                "settlement_date_normalized": pd.Timestamp("2026-08-05"),
            }
        ]
    )

    bank_transactions = pd.DataFrame(
        [
            {
                "transaction_id_normalized": "BANK_B1",
                "amount_normalized": 500000,
                "date_normalized": pd.Timestamp("2026-08-05"),
                "reference_no_normalized": None,
            },
            {
                "transaction_id_normalized": "BANK_B2",
                "amount_normalized": 500000,
                "date_normalized": pd.Timestamp("2026-08-06"),
                "reference_no_normalized": None,
            },
        ]
    )

    result = run_deterministic_matching(
        invoices=invoices,
        payments=payments,
        bank_transactions=bank_transactions,
    )

    # Verify no confirmed match was produced
    assert len(result.payment_bank_matches) == 0

    # Verify AmbiguousMatch objects are present
    assert len(result.ambiguous_matches) > 0
    pay_ambiguous_records = [
        a for a in result.ambiguous_matches if a.source_id == "PAY_AMBIGUOUS"
    ]
    assert len(pay_ambiguous_records) == 1
    assert set(pay_ambiguous_records[0].candidate_ids) == {"BANK_B1", "BANK_B2"}

    # Ensure ambiguous IDs are quarantined (not treated as standard unmatched or matched)
    assert "PAY_AMBIGUOUS" not in result.unmatched_payment_ids
    assert "BANK_B1" not in result.unmatched_bank_transaction_ids
    assert "BANK_B2" not in result.unmatched_bank_transaction_ids


def test_invalid_invoice_link():
    """5. Test payments with non-existent linked invoice IDs remain unmatched."""
    invoices = pd.DataFrame(
        [
            {
                "invoice_id_normalized": "INV_VALID",
                "gross_amount_normalized": 50000,
                "invoice_date_normalized": pd.Timestamp("2026-08-01"),
            }
        ]
    )

    payments = pd.DataFrame(
        [
            {
                "payment_id_normalized": "PAY_INVALID_LINK",
                "linked_invoice_id_normalized": "INV_DOES_NOT_EXIST",
                "net_settled_amount_normalized": 50000,
                "settlement_date_normalized": pd.Timestamp("2026-08-01"),
            }
        ]
    )

    bank_transactions = pd.DataFrame(
        columns=[
            "transaction_id_normalized",
            "amount_normalized",
            "date_normalized",
            "reference_no_normalized",
        ]
    )

    result = run_deterministic_matching(
        invoices=invoices,
        payments=payments,
        bank_transactions=bank_transactions,
    )

    assert len(result.invoice_payment_matches) == 0
    assert "PAY_INVALID_LINK" in result.unmatched_payment_ids
    assert "INV_VALID" in result.unmatched_invoice_ids


def test_date_outside_matching_window():
    """6. Test payment and bank transaction outside date window fail to match."""
    invoices = pd.DataFrame(columns=["invoice_id_normalized"])

    pay_date = pd.Timestamp("2026-08-01")
    bank_date = pd.Timestamp("2026-08-20")  # 19 days difference

    # Explicit invariant validation check
    date_res = check_date_window(
        earlier_date=pay_date,
        later_date=bank_date,
        max_difference_days=7,
        allow_future_dates=True,
        rule_name="TEST_WINDOW",
    )
    assert date_res.passed is False

    payments = pd.DataFrame(
        [
            {
                "payment_id_normalized": "PAY_OUT_OF_WINDOW",
                "linked_invoice_id_normalized": None,
                "net_settled_amount_normalized": 100000,
                "settlement_date_normalized": pay_date,
            }
        ]
    )

    bank_transactions = pd.DataFrame(
        [
            {
                "transaction_id_normalized": "BANK_OUT_OF_WINDOW",
                "amount_normalized": 100000,
                "date_normalized": bank_date,
                "reference_no_normalized": None,
            }
        ]
    )

    result = run_deterministic_matching(
        invoices=invoices,
        payments=payments,
        bank_transactions=bank_transactions,
        max_date_difference_days=7,
    )

    assert len(result.payment_bank_matches) == 0
    assert "PAY_OUT_OF_WINDOW" in result.unmatched_payment_ids
    assert "BANK_OUT_OF_WINDOW" in result.unmatched_bank_transaction_ids