"""Grounded, read-only tools exposed to the Settlement Q&A agent."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, List

import pandas as pd

from app.engine.reconcile import ReconciliationResult


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value.item() if hasattr(value, "item") else value


class SettlementTools:
    """Read-only data layer. It deliberately contains no model or prompt logic."""

    def __init__(
        self, invoices: pd.DataFrame, payments: pd.DataFrame, bank_transactions: pd.DataFrame,
        result: ReconciliationResult,
    ) -> None:
        self.tables = {
            "invoice": ("invoice_id", invoices.copy()),
            "payment": ("payment_id", payments.copy()),
            "transaction": ("transaction_id", bank_transactions.copy()),
        }
        self.result = result

    @staticmethod
    def definitions() -> List[Dict[str, Any]]:
        id_param = {"type": "string", "description": "Record ID, for example PAY-0001."}
        return [
            {"type": "function", "function": {"name": "get_transaction", "description": "Look up a bank transaction by ID.", "parameters": {"type": "object", "properties": {"transaction_id": id_param}, "required": ["transaction_id"], "additionalProperties": False}}},
            {"type": "function", "function": {"name": "get_payment", "description": "Look up a gateway payment by ID.", "parameters": {"type": "object", "properties": {"payment_id": id_param}, "required": ["payment_id"], "additionalProperties": False}}},
            {"type": "function", "function": {"name": "get_invoice", "description": "Look up an invoice by ID.", "parameters": {"type": "object", "properties": {"invoice_id": id_param}, "required": ["invoice_id"], "additionalProperties": False}}},
            {"type": "function", "function": {"name": "get_linked_invoice", "description": "Return the invoice linked to a payment and its match status.", "parameters": {"type": "object", "properties": {"payment_id": id_param}, "required": ["payment_id"], "additionalProperties": False}}},
            {"type": "function", "function": {"name": "get_gateway_fee", "description": "Return gross amount, fee, and net settlement for a payment.", "parameters": {"type": "object", "properties": {"payment_id": id_param}, "required": ["payment_id"], "additionalProperties": False}}},
            {"type": "function", "function": {"name": "get_exception_reason", "description": "Return unresolved or review reasons for a record.", "parameters": {"type": "object", "properties": {"record_id": id_param}, "required": ["record_id"], "additionalProperties": False}}},
            {"type": "function", "function": {"name": "get_metrics", "description": "Return reconciliation totals and match rates.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
        ]

    def _record(self, record_type: str, record_id: str) -> Dict[str, Any]:
        column, table = self.tables[record_type]
        rows = table[table[column].astype(str).str.upper() == record_id.upper()]
        if rows.empty:
            return {"found": False, "record_id": record_id}
        return {"found": True, "record": {key: _json_value(value) for key, value in rows.iloc[0].to_dict().items() if not key.endswith("_normalized")}}

    def get_transaction(self, transaction_id: str) -> Dict[str, Any]:
        return self._record("transaction", transaction_id)

    def get_payment(self, payment_id: str) -> Dict[str, Any]:
        return self._record("payment", payment_id)

    def get_invoice(self, invoice_id: str) -> Dict[str, Any]:
        return self._record("invoice", invoice_id)

    def get_linked_invoice(self, payment_id: str) -> Dict[str, Any]:
        payment = self.get_payment(payment_id)
        if not payment["found"]:
            return payment
        linked_id = payment["record"].get("linked_invoice_id")
        confirmed = [m for m in self.result.confirmed_invoice_payment_matches if m.left_id.upper() == payment_id.upper()]
        return {"payment_id": payment_id, "linked_invoice_id": linked_id, "confirmed": bool(confirmed), "invoice": self.get_invoice(str(linked_id)) if linked_id else None}

    def get_gateway_fee(self, payment_id: str) -> Dict[str, Any]:
        payment = self.get_payment(payment_id)
        if not payment["found"]:
            return payment
        record = payment["record"]
        return {"payment_id": payment_id, "gross_amount": record["gross_amount"], "fee": record["fee"], "net_settled_amount": record["net_settled_amount"]}

    def get_exception_reason(self, record_id: str) -> Dict[str, Any]:
        items = [asdict(item) for item in self.result.exceptions if item.record_id.upper() == record_id.upper()]
        return {"record_id": record_id, "exceptions": items, "found": bool(items)}

    def get_metrics(self) -> Dict[str, Any]:
        return asdict(self.result.metrics)

    def execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        functions: Dict[str, Callable[..., Dict[str, Any]]] = {
            "get_transaction": self.get_transaction, "get_payment": self.get_payment,
            "get_invoice": self.get_invoice, "get_linked_invoice": self.get_linked_invoice,
            "get_gateway_fee": self.get_gateway_fee, "get_exception_reason": self.get_exception_reason,
            "get_metrics": self.get_metrics,
        }
        if name not in functions:
            return {"error": f"Unknown tool: {name}"}
        try:
            return functions[name](**arguments)
        except TypeError:
            return {"error": f"Invalid arguments for tool: {name}"}
