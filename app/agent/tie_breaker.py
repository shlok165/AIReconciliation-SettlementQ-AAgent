"""LLM pass for ambiguous reconciliation candidates only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from app.agent.client import PollinationsClient
from app.engine.scoring import ScoredCandidate


@dataclass(frozen=True)
class TieBreakerDecision:
    source_id: str
    match_type: str
    chosen_candidate_id: Optional[str]
    confidence: float
    justification: str
    accepted: bool


def _row(df: pd.DataFrame, column: str, value: str) -> Dict[str, Any]:
    matched = df[df[column].astype(str).str.upper() == value.upper()]
    if matched.empty:
        return {"id": value, "found": False}
    raw = matched.iloc[0].to_dict()
    return {key: (None if pd.isna(item) else str(item)) for key, item in raw.items() if not key.endswith("_normalized")}


def _parse_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        raise ValueError("The LLM did not return a JSON decision.")
    return json.loads(match.group())


def resolve_ambiguous_candidates(
    candidates: Iterable[ScoredCandidate], invoices: pd.DataFrame, payments: pd.DataFrame,
    bank_transactions: pd.DataFrame, client: PollinationsClient, *, min_confidence: float = 90.0,
    log_path: Path = Path("reports/llm_decisions.jsonl"),
) -> List[TieBreakerDecision]:
    """Ask the LLM to select only among supplied review candidates; never force a match."""
    grouped: Dict[tuple[str, str], List[ScoredCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault((candidate.candidate.match_type, candidate.candidate.left_id), []).append(candidate)

    decisions: List[TieBreakerDecision] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for (match_type, source_id), options in grouped.items():
        # A single low-confidence candidate is not a tie. Keep it in manual
        # review instead of spending an LLM call or turning a weak signal into
        # an automatic match. Pass 3 handles only competing fuzzy candidates.
        if len(options) < 2:
            continue
        is_payment_bank = "PAYMENT_BANK" in match_type
        source = _row(payments, "payment_id", source_id)
        option_rows = []
        for option in options:
            record = _row(bank_transactions if is_payment_bank else invoices, "transaction_id" if is_payment_bank else "invoice_id", option.candidate.right_id)
            option_rows.append({"candidate_id": option.candidate.right_id, "record": record, "engine_confidence": option.confidence_score, "evidence": option.reasons})
        prompt = {
            "task": "Select the best reconciliation candidate only if evidence supports it. You may return null for no match. Never invent IDs or force a match.",
            "source": source, "candidates": option_rows,
            "response_schema": {"chosen_candidate_id": "candidate ID or null", "confidence": "0-100", "justification": "short evidence-based explanation"},
        }
        message = client.chat([
            {"role": "system", "content": "You are a cautious financial reconciliation reviewer. Return JSON only."},
            {"role": "user", "content": json.dumps(prompt)},
        ])
        parsed = _parse_json(message.get("content", ""))
        chosen = parsed.get("chosen_candidate_id")
        chosen = str(chosen).upper() if chosen is not None else None
        permitted = {item.candidate.right_id.upper() for item in options}
        confidence = float(parsed.get("confidence", 0))
        accepted = chosen in permitted and confidence >= min_confidence
        decision = TieBreakerDecision(source_id, match_type, chosen if accepted else None, confidence, str(parsed.get("justification", "")), accepted)
        decisions.append(decision)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"input": prompt, "output": message, "decision": asdict(decision)}) + "\n")
    return decisions
