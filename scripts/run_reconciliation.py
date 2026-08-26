"""Evaluation runner script to execute reconciliation against raw CSVs and compare with ground truth."""

import sys
from pathlib import Path

# Ensure project root is in python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.data.loader import load_all_data
from app.engine.reconcile import reconcile
from app.evaluation.metrics import calculate_evaluation_metrics


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

    evaluation = calculate_evaluation_metrics(
        result,
        ground_truth_path=ground_truth_path,
        elapsed_seconds=0.0,
    )

    print("=== GROUND TRUTH EVALUATION ===\n")
    print("-- Transaction-level (Primary) --")
    print(f"Total transactions: {evaluation.total_transactions}")
    print(f"Correctly resolved transactions: {evaluation.correctly_resolved_transactions}")
    print(f"Transactions requiring review: {evaluation.transactions_requiring_review}")
    print(f"Unresolved transactions: {evaluation.unresolved_transactions}")
    print(f"Incorrectly resolved transactions: {evaluation.incorrectly_resolved_transactions}")
    print(f"Needs attention transactions: {evaluation.needs_attention_transactions}")
    print(f"Transaction resolution accuracy: {evaluation.transaction_resolution_accuracy:.2f}%\n")

    print("Transaction stage breakdown:")
    for key, value in evaluation.transaction_resolution_stage_breakdown.items():
        print(f"- {key}: {value}")
    print()

    print("-- Relationship-level (Secondary) --")
    print(f"Expected match relationships: {evaluation.expected_match_relationships}")
    print(f"Actual resolved relationships: {evaluation.resolved_match_relationships}")
    print(f"True positives: {evaluation.true_positives}")
    print(f"False positives: {evaluation.false_positives}")
    print(f"False negatives: {evaluation.false_negatives}")
    print(f"Precision: {evaluation.precision:.2f}%")
    print(f"Recall: {evaluation.recall:.2f}%\n")

    print("-- Stage identification matrix (Secondary) --")
    for stage_name, stage_metrics in evaluation.identification_matrix.items():
        print(
            f"{stage_name}: identified={stage_metrics['identified_relationships']}, "
            f"correct={stage_metrics['correct_relationships']}, "
            f"incorrect={stage_metrics['incorrect_relationships']}, "
            f"precision={stage_metrics['precision']}%, "
            f"coverage_contribution={stage_metrics['coverage_contribution']}%"
        )


if __name__ == "__main__":
    main()