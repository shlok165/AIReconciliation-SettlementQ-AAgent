"""Evaluation runner script to execute reconciliation against raw CSVs and compare with ground truth."""

import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.data.loader import load_all_data
from app.engine.reconcile import reconcile


def main():
    # 1. Load raw datasets using the actual loader API (Order: invoices, bank_transactions, payments)
    try:
        invoices, bank_transactions, payments = load_all_data()
    except Exception as exc:
        print(f"Error loading datasets via app.data.loader: {exc}")
        sys.exit(1)

    # 2. Run reconciliation engine
    result = reconcile(
        invoices=invoices,
        payments=payments,
        bank_transactions=bank_transactions
    )

    # 3. Print Reconciliation Results formatted to specification
    print("=== RECONCILIATION RESULTS ===\n")
    print(f"Total invoices: {result.metrics.total_invoices}")
    print(f"Total payments: {result.metrics.total_payments}")
    print(f"Total bank transactions: {result.metrics.total_bank_transactions}\n")

    deterministic_count = result.metrics.deterministic_confirmed_matches
    auto_matches_count = result.metrics.fuzzy_auto_matches
    review_count = result.metrics.manual_review_candidates
    rejected_count = result.metrics.rejected_fuzzy_candidates

    print(f"Deterministic matches: {deterministic_count}")
    print(f"Fuzzy auto matches: {auto_matches_count}")
    print(f"Review candidates: {review_count}")
    print(f"Rejected candidates: {rejected_count}\n")

    print(f"Unresolved invoices: {result.metrics.unresolved_invoices}")
    print(f"Unresolved payments: {result.metrics.unresolved_payments}")
    print(f"Unresolved bank transactions: {result.metrics.unresolved_bank_transactions}\n")

    print(f"Invoice match rate: {result.metrics.invoice_match_rate}%")
    print(f"Payment match rate: {result.metrics.payment_match_rate}%")
    print(f"Bank transaction match rate: {result.metrics.bank_transaction_match_rate}%\n")

    print("=== EXCEPTIONS ===")
    if result.exceptions:
        for exc in result.exceptions:
            print(f"- [{exc.exception_type}] Record ID: {exc.record_id} ({exc.record_type}) -> {exc.description}")
    else:
        print("No exceptions recorded.")
    print()

    # 4. Compare with Ground Truth Evaluation
    ground_truth_path = project_root / "data" / "ground_truth" / "ground_truth.csv"
    if not ground_truth_path.exists():
        print("=== GROUND TRUTH EVALUATION ===")
        print(f"Ground truth file not found at {ground_truth_path}. Skipping evaluation breakdown.")
        return

    gt_df = pd.read_csv(ground_truth_path)

    # Extract actual resolved match relationships from engine output with composite keys
    actual_matches = set()
    
    for m in getattr(result, "confirmed_invoice_payment_matches", []):
        actual_matches.add(("INVOICE_PAYMENT", str(m.right_id), str(m.left_id)))
    for m in getattr(result, "confirmed_payment_bank_matches", []):
        actual_matches.add(("PAYMENT_BANK", str(m.left_id), str(m.right_id)))
        
    for am in getattr(result, "auto_matches", []):
        c = am.candidate
        if "PAYMENT_BANK" in c.match_type:
            actual_matches.add(("PAYMENT_BANK", str(c.left_id), str(c.right_id)))
        else:
            actual_matches.add(("INVOICE_PAYMENT", str(c.right_id), str(c.left_id)))

    # Parse ground truth expected relationships based on actual schema:
    # ['case_id', 'invoice_id', 'transaction_id', 'payment_id', 'expected_result', 'exception_reason']
    expected_matches = set()
    expected_exceptions_count = 0
    
    for _, row in gt_df.iterrows():
        status = str(row.get("expected_result", "")).strip().upper()
        inv = str(row.get("invoice_id", "")).strip()
        pay = str(row.get("payment_id", "")).strip()
        txn = str(row.get("transaction_id", "")).strip()

        if status == "MATCH":
            if inv and pay and inv != "nan" and pay != "nan":
                expected_matches.add(("INVOICE_PAYMENT", inv, pay))
            if pay and txn and pay != "nan" and txn != "nan":
                expected_matches.add(("PAYMENT_BANK", pay, txn))
        else:
            expected_exceptions_count += 1

    # Calculate True Positives, False Positives, False Negatives
    true_positives = len(actual_matches.intersection(expected_matches))
    false_positives = len(actual_matches - expected_matches)
    false_negatives = len(expected_matches - actual_matches)

    precision = (true_positives / len(actual_matches) * 100.0) if len(actual_matches) > 0 else 0.0
    recall = (true_positives / len(expected_matches) * 100.0) if len(expected_matches) > 0 else 0.0

    # Exception evaluation metrics using actual unresolved IDs/exceptions against ground truth expectations
    unresolved_ids = set(result.unresolved_invoice_ids + result.unresolved_payment_ids + result.unresolved_bank_transaction_ids)
    
    correctly_flagged = 0
    incorrectly_flagged = 0
    
    for exc in result.exceptions:
        if exc.record_id in unresolved_ids or exc.exception_type in ["NO_MATCH_FOUND", "MANUAL_REVIEW_REQUIRED", "CONFLICTING_AUTO_MATCH", "AMBIGUOUS_DETERMINISTIC_MATCH"]:
            correctly_flagged += 1
        else:
            incorrectly_flagged += 1

    fp_set = sorted(list(actual_matches - expected_matches))
    fn_set = sorted(list(expected_matches - actual_matches))

    print("=== GROUND TRUTH EVALUATION ===\n")
    print(f"Expected match relationships: {len(expected_matches)}")
    print(f"Actual resolved matches: {len(actual_matches)}")
    print(f"True positives: {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%\n")
    print(f"Correctly flagged exceptions: {correctly_flagged}")
    print(f"Incorrectly flagged exceptions: {incorrectly_flagged}\n")

    if fp_set:
        print("=== FALSE POSITIVE MATCHES ===")
        for rel_type, left, right in fp_set:
            print(f"- [{rel_type}] {left} <-> {right}")
        print()

    if fn_set:
        print("=== FALSE NEGATIVE MATCHES ===")
        for rel_type, left, right in fn_set:
            print(f"- [{rel_type}] {left} <-> {right}")
        print()


if __name__ == "__main__":
    main()