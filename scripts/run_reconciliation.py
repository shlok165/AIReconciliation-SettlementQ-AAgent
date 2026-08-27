"""Evaluation runner script to execute reconciliation against raw CSVs and compare with ground truth."""

import sys
import argparse
from pathlib import Path

# Ensure project root is in python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.data.loader import load_all_data
from app.engine.reconcile import reconcile
from app.evaluation.metrics import calculate_evaluation_metrics


def main():
    parser = argparse.ArgumentParser(description="Run reconciliation and evaluation.")
    parser.add_argument(
        "--llm-eval", action="store_true",
        help="Enable LLM evaluation of unmatched transactions.",
    )
    parser.add_argument(
        "--llm-min-confidence", type=float, default=70.0,
        help="Minimum LLM confidence to accept a MATCH resolution (default: 70).",
    )
    args = parser.parse_args()

    # 1. Load raw datasets using the actual loader API (Order: invoices, bank_transactions, payments)
    try:
        invoices, bank_transactions, payments = load_all_data()
    except Exception as exc:
        print(f"Error loading datasets via app.data.loader: {exc}")
        sys.exit(1)

    # 2. Optionally create LLM clients
    llm_tie_client = None
    llm_eval_client = None
    ground_truth_path = project_root / "data" / "ground_truth" / "ground_truth.csv"

    if args.llm_eval:
        try:
            from app.agent.client import PollinationsClient
            llm_eval_client = PollinationsClient()
            llm_tie_client = PollinationsClient()
            print("LLM evaluation enabled. Pollinations client created.\n")
        except Exception as exc:
            print(f"Warning: Could not create LLM client: {exc}")
            print("Proceeding without LLM evaluation.\n")

    # 3. Run reconciliation engine
    result = reconcile(
        invoices=invoices,
        payments=payments,
        bank_transactions=bank_transactions,
        llm_tie_breaker_client=llm_tie_client,
        llm_evaluation_client=llm_eval_client,
        ground_truth_path=str(ground_truth_path) if ground_truth_path.exists() else None,
    )

    # 4. Print Reconciliation Results formatted to specification
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

    # 5. LLM Evaluation Results (if available)
    if result.llm_evaluation_result is not None:
        llm_res = result.llm_evaluation_result
        print("=== LLM EVALUATION OF UNMATCHED TRANSACTIONS ===\n")
        print(f"Total cases evaluated by LLM: {llm_res.total_cases_evaluated}")
        print(f"LLM correctly resolved (matched): {llm_res.llm_resolved_count}")
        print(f"LLM correctly identified as exception: {llm_res.llm_correct_exception_count}")
        print(f"LLM incorrectly resolved: {llm_res.llm_incorrect_count}")
        print(f"LLM resolution accuracy: {llm_res.llm_resolution_accuracy:.2f}%\n")

        if llm_res.details:
            print("-- Per-case LLM decisions --")
            for d in llm_res.details:
                verdict_marker = "+" if "CORRECT" in d.get("verdict", "") else "X"
                print(
                    f"  [{verdict_marker}] {d['case_id']} | "
                    f"Record: {d['record_id']} ({d['record_type']}) | "
                    f"LLM: {d['llm_resolution']} | "
                    f"Expected: {d['expected_result']} | "
                    f"Verdict: {d['verdict']} | "
                    f"Confidence: {d['llm_confidence']:.0f}"
                )
                print(f"      Justification: {d['llm_justification']}")
            print()

    # 6. Compare with Ground Truth Evaluation
    if not ground_truth_path.exists():
        print("=== GROUND TRUTH EVALUATION ===")
        print(f"Ground truth file not found at {ground_truth_path}. Skipping evaluation breakdown.")
        return

    evaluation = calculate_evaluation_metrics(
        result,
        ground_truth_path=ground_truth_path,
        elapsed_seconds=0.0,
        llm_resolution_result=result.llm_evaluation_result,
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

    if evaluation.llm_cases_evaluated > 0:
        print("-- LLM Resolution Metrics --")
        print(f"LLM cases evaluated: {evaluation.llm_cases_evaluated}")
        print(f"LLM correctly resolved (matched): {evaluation.llm_resolved_transactions}")
        print(f"LLM correctly identified as exception: {evaluation.llm_correct_exception_determinations}")
        print(f"LLM incorrectly resolved: {evaluation.llm_incorrect_resolutions}")
        print(f"LLM resolution accuracy: {evaluation.llm_resolution_accuracy:.2f}%\n")

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
