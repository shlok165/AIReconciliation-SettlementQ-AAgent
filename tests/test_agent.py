import json

import pandas as pd

from app.agent.qa import SettlementQAAgent
from app.agent.tools import SettlementTools
from app.engine.reconcile import ReconciliationMetrics, ReconciliationResult
from app.engine.deterministic import DeterministicMatchResult
from app.engine.fuzzy import FuzzyMatchResult
from app.engine.scoring import ScoringResult


def _result():
    return ReconciliationResult(
        DeterministicMatchResult(), FuzzyMatchResult([], [], [], [], []), ScoringResult([], [], [], []),
        [], [], [], [], [], [], [], [], [],
        ReconciliationMetrics(1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )


def test_tool_returns_gateway_fee():
    tools = SettlementTools(
        pd.DataFrame([{"invoice_id": "INV-1", "expected_amount": 100.0}]),
        pd.DataFrame([{"payment_id": "PAY-1", "gross_amount": 100.0, "fee": 3.2, "net_settled_amount": 96.8, "linked_invoice_id": "INV-1"}]),
        pd.DataFrame([{"transaction_id": "TXN-1", "amount": 96.8}]), _result(),
    )
    assert tools.get_gateway_fee("pay-1") == {"payment_id": "pay-1", "gross_amount": 100.0, "fee": 3.2, "net_settled_amount": 96.8}


def test_agent_executes_tool_call_then_returns_answer():
    class FakeClient:
        def __init__(self): self.calls = 0
        def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"tool_calls": [{"id": "call_1", "function": {"name": "get_gateway_fee", "arguments": json.dumps({"payment_id": "PAY-1"})}}]}
            assert messages[-1]["role"] == "tool"
            return {"content": "PAY-1 has a gateway fee of 3.2."}
    tools = SettlementTools(pd.DataFrame(), pd.DataFrame([{"payment_id": "PAY-1", "gross_amount": 100, "fee": 3.2, "net_settled_amount": 96.8}]), pd.DataFrame(), _result())
    response = SettlementQAAgent(tools, FakeClient()).answer("What is PAY-1's fee?")
    assert response["answer"] == "PAY-1 has a gateway fee of 3.2."
    assert response["tool_trace"][0]["tool"] == "get_gateway_fee"
