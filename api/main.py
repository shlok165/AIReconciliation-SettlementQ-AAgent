"""FastAPI entry point for the grounded Settlement Q&A flow."""

from functools import lru_cache
from dataclasses import asdict
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent.client import LLMClientError, PollinationsClient
from app.agent.qa import SettlementQAAgent
from app.agent.tools import SettlementTools
from app.data.loader import load_all_data
from app.engine.reconcile import reconcile

load_dotenv()
app = FastAPI(title="AI Reconciliation & Settlement Q&A Agent")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)


class AskResponse(BaseModel):
    answer: str
    tool_trace: List[Dict[str, Any]]


class ReconcileResponse(BaseModel):
    metrics: Dict[str, Any]
    llm_resolved_matches: int
    remaining_manual_reviews: int


@lru_cache(maxsize=1)
def _agent() -> SettlementQAAgent:
    invoices, bank_transactions, payments = load_all_data()
    result = reconcile(invoices=invoices, payments=payments, bank_transactions=bank_transactions)
    return SettlementQAAgent(
        SettlementTools(invoices, payments, bank_transactions, result),
        PollinationsClient(),
    )


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
