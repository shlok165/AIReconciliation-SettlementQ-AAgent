"""Function-calling Settlement Q&A agent.

The model has no direct ledger context.  It must call read-only tools and the
final response is therefore grounded in the reconciliation data supplied here.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from app.agent.client import PollinationsClient
from app.agent.tools import SettlementTools


SYSTEM_PROMPT = """You are the Settlement Q&A assistant. Use the provided tools before
answering any question about a transaction, payment, invoice, fee, exception, or metric.
State amounts and IDs exactly as returned by tools. If a record is not found or is unresolved,
say so plainly. Never infer facts that a tool did not return."""


class SettlementQAAgent:
    def __init__(self, tools: SettlementTools, client: PollinationsClient) -> None:
        self.tools = tools
        self.client = client

    def answer(self, question: str, *, max_tool_rounds: int = 5) -> Dict[str, Any]:
        if not question.strip():
            raise ValueError("Question must not be empty.")
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question.strip()},
        ]
        tool_trace: List[Dict[str, Any]] = []
        for _ in range(max_tool_rounds):
            message = self.client.chat(messages, tools=self.tools.definitions())
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return {"answer": message.get("content") or "I could not produce an answer.", "tool_trace": tool_trace}
            messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": tool_calls})
            for call in tool_calls:
                function = call.get("function", {})
                name = function.get("name", "")
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                result = self.tools.execute(name, arguments)
                tool_trace.append({"tool": name, "arguments": arguments, "result": result})
                messages.append({"role": "tool", "tool_call_id": call.get("id", name), "content": json.dumps(result, default=str)})
        return {"answer": "I could not complete the lookup within the permitted tool-call limit.", "tool_trace": tool_trace}
