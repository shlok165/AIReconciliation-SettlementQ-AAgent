"""FastAPI entry point for the grounded Settlement Q&A flow."""

from functools import lru_cache
from dataclasses import asdict
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agent.client import LLMClientError, PollinationsClient
from app.agent.qa import SettlementQAAgent
from app.agent.tools import SettlementTools
from app.data.loader import load_all_data
from app.engine.reconcile import reconcile
from app.evaluation.metrics import calculate_evaluation_metrics
from app.reporting.report_generator import generate_final_report
from scripts.generate_data import generate_dataset

logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI(title="AI Reconciliation & Settlement Q&A Agent")

# Stores the latest LLM evaluation result after /reconcile runs.
_latest_llm_eval: Dict[str, Any] = {}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)


class AskResponse(BaseModel):
    answer: str
    tool_trace: List[Dict[str, Any]]


class ReconcileResponse(BaseModel):
    metrics: Dict[str, Any]
    llm_resolved_matches: int
    remaining_manual_reviews: int
    llm_evaluation: Dict[str, Any] = {}


class GenerateDataRequest(BaseModel):
    size: int = Field(default=100, ge=10, le=5000)


class GenerateDataResponse(BaseModel):
    size_requested: int
    total_invoices: int
    total_payments: int
    total_bank_transactions: int
    total_ground_truth_cases: int
    total_records: int
    output_dir: str


class BaselineRun:
    def __init__(self, result, evaluation) -> None:
        self.result = result
        self.evaluation = evaluation


def _invalidate_runtime_caches() -> None:
    """Clear cached views so endpoints reflect the latest dataset on disk."""
    _agent.cache_clear()
    _baseline_run.cache_clear()
    _latest_llm_eval.clear()


@lru_cache(maxsize=1)
def _agent() -> SettlementQAAgent:
    invoices, bank_transactions, payments = load_all_data()
    result = reconcile(invoices=invoices, payments=payments, bank_transactions=bank_transactions)
    return SettlementQAAgent(
        SettlementTools(invoices, payments, bank_transactions, result),
        PollinationsClient(),
    )


@lru_cache(maxsize=1)
def _baseline_run() -> BaselineRun:
    """Fast cached run without LLM evaluation for initial load."""
    started_at = perf_counter()
    invoices, bank_transactions, payments = load_all_data()
    result = reconcile(invoices=invoices, payments=payments, bank_transactions=bank_transactions)
    elapsed_seconds = perf_counter() - started_at
    evaluation = calculate_evaluation_metrics(
        result,
        ground_truth_path=Path("data/ground_truth/ground_truth.csv"),
        elapsed_seconds=elapsed_seconds,
    )
    return BaselineRun(result, evaluation)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    logger.info("Settlement QA request received: %s", request.question.strip())
    try:
        response = _agent().answer(request.question)
    except (ValueError, LLMClientError) as exc:
        logger.warning("Settlement QA request failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    logger.info("Settlement QA request completed with %d tool calls", len(response.get("tool_trace", [])))
    return AskResponse(**response)


@app.post("/reconcile", response_model=ReconcileResponse)
def reconcile_with_tie_breaker() -> ReconcileResponse:
    """Run all passes including LLM evaluation of unmatched transactions."""
    _invalidate_runtime_caches()
    try:
        invoices, bank_transactions, payments = load_all_data()
        client = PollinationsClient()
        gt_path = Path("data/ground_truth/ground_truth.csv")
        result = reconcile(
            invoices=invoices,
            payments=payments,
            bank_transactions=bank_transactions,
            llm_tie_breaker_client=client,
            llm_evaluation_client=client,
            ground_truth_path=str(gt_path) if gt_path.exists() else None,
        )
    except (ValueError, LLMClientError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    llm_resolved = sum(
        any("LLM tie-breaker selected" in reason for reason in match.reasons)
        for match in result.auto_matches
    )
    llm_eval_dict = {}
    if result.llm_evaluation_result is not None:
        llm_eval_dict = {
            "total_cases_evaluated": result.llm_evaluation_result.total_cases_evaluated,
            "llm_resolved_count": result.llm_evaluation_result.llm_resolved_count,
            "llm_incorrect_count": result.llm_evaluation_result.llm_incorrect_count,
            "llm_correct_exception_count": result.llm_evaluation_result.llm_correct_exception_count,
            "llm_resolution_accuracy": result.llm_evaluation_result.llm_resolution_accuracy,
            "details": result.llm_evaluation_result.details,
        }
    _latest_llm_eval.update(llm_eval_dict)
    return ReconcileResponse(
        metrics=asdict(result.metrics),
        llm_resolved_matches=llm_resolved,
        remaining_manual_reviews=len(result.review_candidates),
        llm_evaluation=llm_eval_dict,
    )


@app.post("/llm-evaluate")
def llm_evaluate_only() -> Dict[str, Any]:
    """Run reconciliation without tie-breaker, then LLM-evaluate unmatched cases only."""
    try:
        invoices, bank_transactions, payments = load_all_data()
        client = PollinationsClient()
        gt_path = Path("data/ground_truth/ground_truth.csv")
        result = reconcile(
            invoices=invoices,
            payments=payments,
            bank_transactions=bank_transactions,
            llm_evaluation_client=client,
            ground_truth_path=str(gt_path) if gt_path.exists() else None,
        )
    except (ValueError, LLMClientError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    llm_eval_dict = {}
    if result.llm_evaluation_result is not None:
        llm_eval_dict = {
            "total_cases_evaluated": result.llm_evaluation_result.total_cases_evaluated,
            "llm_resolved_count": result.llm_evaluation_result.llm_resolved_count,
            "llm_incorrect_count": result.llm_evaluation_result.llm_incorrect_count,
            "llm_correct_exception_count": result.llm_evaluation_result.llm_correct_exception_count,
            "llm_resolution_accuracy": result.llm_evaluation_result.llm_resolution_accuracy,
            "details": result.llm_evaluation_result.details,
        }
    _latest_llm_eval.update(llm_eval_dict)
    return llm_eval_dict


@app.get("/unresolved")
def unresolved_transactions() -> Dict[str, Any]:
    """Return grouped unresolved and matched transactions for display."""
    import pandas as pd
    from app.agent.llm_resolver import build_unmatched_cases

    invoices, bank_transactions, payments = load_all_data()
    gt_path = Path("data/ground_truth/ground_truth.csv")
    result = reconcile(
        invoices=invoices,
        payments=payments,
        bank_transactions=bank_transactions,
    )

    already_matched: set = set()
    for m in result.confirmed_payment_bank_matches:
        already_matched.add(m.left_id); already_matched.add(m.right_id)
    for m in result.confirmed_invoice_payment_matches:
        already_matched.add(m.left_id); already_matched.add(m.right_id)
    for auto in result.auto_matches:
        already_matched.add(auto.candidate.left_id); already_matched.add(auto.candidate.right_id)

    gt_df = pd.read_csv(gt_path) if gt_path.exists() else None
    gt_lookup: Dict[str, Dict[str, Any]] = {}
    if gt_df is not None:
        for _, row in gt_df.iterrows():
            for col in ("payment_id", "invoice_id", "transaction_id"):
                rid = str(row.get(col, "")).strip()
                if rid and rid.lower() != "nan":
                    gt_lookup[rid] = {
                        "case_id": str(row.get("case_id", "")),
                        "expected_result": str(row.get("expected_result", "")),
                        "category": str(row.get("category", "")),
                        "payment_id": str(row.get("payment_id", "")),
                        "invoice_id": str(row.get("invoice_id", "")),
                        "transaction_id": str(row.get("transaction_id", "")),
                    }

    # Build matched relationships
    matched_relationships = []
    for m in result.confirmed_payment_bank_matches:
        matched_relationships.append({
            "left_id": m.left_id, "right_id": m.right_id,
            "type": "PAYMENT_BANK", "stage": "deterministic",
            "reasons": [f"{m.match_type}: {m.evidence}"],
            "ground_truth": gt_lookup.get(m.left_id) or gt_lookup.get(m.right_id),
        })
    for m in result.confirmed_invoice_payment_matches:
        matched_relationships.append({
            "left_id": m.left_id, "right_id": m.right_id,
            "type": "INVOICE_PAYMENT", "stage": "deterministic",
            "reasons": [f"{m.match_type}: {m.evidence}"],
            "ground_truth": gt_lookup.get(m.left_id) or gt_lookup.get(m.right_id),
        })
    for auto in result.auto_matches:
        stage = "llm" if any("LLM tie-breaker" in r for r in auto.reasons) else "fuzzy"
        matched_relationships.append({
            "left_id": auto.candidate.left_id, "right_id": auto.candidate.right_id,
            "type": auto.candidate.match_type, "stage": stage,
            "reasons": auto.reasons,
            "ground_truth": gt_lookup.get(auto.candidate.left_id) or gt_lookup.get(auto.candidate.right_id),
        })

    cases = build_unmatched_cases(
        result, invoices, payments, bank_transactions,
        ground_truth_path=gt_path if gt_path.exists() else None,
    )

    # Attach LLM decisions (if any) to each case
    llm_decision_map: Dict[str, Any] = {}
    for detail in _latest_llm_eval.get("details", []):
        rec_id = str(detail.get("record_id", "")).upper()
        llm_decision_map[rec_id] = detail

    return {
        "unresolved_payments": result.unresolved_payment_ids,
        "unresolved_bank_txns": result.unresolved_bank_transaction_ids,
        "unresolved_invoices": result.unresolved_invoice_ids,
        "review_candidate_ids": [r.candidate.left_id for r in result.review_candidates],
        "matched": matched_relationships,
        "llm_cases": [
            {
                "case_id": c.case_id,
                "record_type": c.record_type,
                "record_id": c.record_id,
                "exception_type": c.exception_type,
                "description": c.description,
                "related_ids": c.related_ids,
                "ground_truth": gt_lookup.get(c.record_id),
                "llm_decision": llm_decision_map.get(c.record_id.upper()),
                "candidates": [
                    {"id": cand.get("id"), "type": cand.get("type"), "amount": cand.get("amount"), "diff": cand.get("amount_diff")}
                    for cand in c.potential_candidates
                ],
            }
            for c in cases
        ],
    }


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    run = _baseline_run()
    evaluation_dict = run.evaluation.as_dict()
    if _latest_llm_eval:
        llm_resolved_count = _latest_llm_eval.get("llm_resolved_count", 0)
        evaluation_dict["llm_cases_evaluated"] = _latest_llm_eval.get("total_cases_evaluated", 0)
        evaluation_dict["llm_resolved_transactions"] = llm_resolved_count
        evaluation_dict["llm_incorrect_resolutions"] = _latest_llm_eval.get("llm_incorrect_count", 0)
        evaluation_dict["llm_correct_exception_determinations"] = _latest_llm_eval.get("llm_correct_exception_count", 0)
        evaluation_dict["llm_resolution_accuracy"] = _latest_llm_eval.get("llm_resolution_accuracy", 0.0)
        # Reflect LLM-resolved transactions in the bar chart breakdown
        if llm_resolved_count > 0:
            breakdown = evaluation_dict.get("transaction_resolution_stage_breakdown", {})
            breakdown["llm_resolved_transactions"] = breakdown.get("llm_resolved_transactions", 0) + llm_resolved_count
            evaluation_dict["transaction_resolution_stage_breakdown"] = breakdown
            evaluation_dict["correctly_resolved_transactions"] = evaluation_dict.get("correctly_resolved_transactions", 0) + llm_resolved_count
            evaluation_dict["transaction_resolution_accuracy"] = round(
                evaluation_dict["correctly_resolved_transactions"] / evaluation_dict["total_transactions"] * 100.0, 2
            ) if evaluation_dict.get("total_transactions") else 0.0
    return {"reconciliation": asdict(run.result.metrics), "evaluation": evaluation_dict}


@app.get("/exceptions")
def exceptions() -> List[Dict[str, Any]]:
    return [asdict(exception) for exception in _baseline_run().result.exceptions]


@app.get("/dataset")
def dataset() -> Dict[str, Any]:
    """Return the current generated dataset for display and assistant context."""
    import pandas as pd
    invoices, bank_transactions, payments = load_all_data()

    def _df_to_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
        return [{k: (None if pd.isna(v) else str(v)) for k, v in row.items()} for _, row in df.iterrows()]

    return {
        "invoices": _df_to_list(invoices),
        "payments": _df_to_list(payments),
        "bank_transactions": _df_to_list(bank_transactions),
        "invoice_count": len(invoices),
        "payment_count": len(payments),
        "bank_transaction_count": len(bank_transactions),
    }


@app.post("/generate-data", response_model=GenerateDataResponse)
def generate_data(request: GenerateDataRequest) -> GenerateDataResponse:
    generated = generate_dataset(size=request.size, output_dir=Path("data"))
    _invalidate_runtime_caches()
    return GenerateDataResponse(**generated)


@app.post("/report")
def report() -> Dict[str, str]:
    run = _baseline_run()
    return generate_final_report(run.result, run.evaluation, output_dir=Path("reports"))
