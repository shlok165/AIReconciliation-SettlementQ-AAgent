"""FastAPI entry point for the grounded Settlement Q&A flow."""

from functools import lru_cache
from dataclasses import asdict
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

load_dotenv()
app = FastAPI(title="AI Reconciliation & Settlement Q&A Agent")
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


class BaselineRun:
    def __init__(self, result, evaluation) -> None:
        self.result = result
        self.evaluation = evaluation


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
    """Cached non-LLM batch used by operational reporting endpoints."""
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
    try:
        response = _agent().answer(request.question)
    except (ValueError, LLMClientError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return AskResponse(**response)


@app.post("/reconcile", response_model=ReconcileResponse)
def reconcile_with_tie_breaker() -> ReconcileResponse:
    """Run all passes; Pass 3 calls the LLM only for competing fuzzy candidates."""
    try:
        invoices, bank_transactions, payments = load_all_data()
        result = reconcile(
            invoices=invoices,
            payments=payments,
            bank_transactions=bank_transactions,
            llm_tie_breaker_client=PollinationsClient(),
        )
    except (ValueError, LLMClientError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    llm_resolved = sum(
        any("LLM tie-breaker selected" in reason for reason in match.reasons)
        for match in result.auto_matches
    )
    return ReconcileResponse(
        metrics=asdict(result.metrics),
        llm_resolved_matches=llm_resolved,
        remaining_manual_reviews=len(result.review_candidates),
    )


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    run = _baseline_run()
    return {"reconciliation": asdict(run.result.metrics), "evaluation": run.evaluation.as_dict()}


@app.get("/exceptions")
def exceptions() -> List[Dict[str, Any]]:
    return [asdict(exception) for exception in _baseline_run().result.exceptions]


@app.post("/report")
def report() -> Dict[str, str]:
    run = _baseline_run()
    return generate_final_report(run.result, run.evaluation, output_dir=Path("reports"))
