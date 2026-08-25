"""Small Pollinations client using its OpenAI-compatible Chat Completions API."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import PollinationsSettings


class LLMClientError(RuntimeError):
    """Raised when a model request cannot be completed safely."""


class PollinationsClient:
    """Backend-only client; API keys are never sent to the browser."""

    def __init__(self, settings: Optional[PollinationsSettings] = None) -> None:
        self.settings = settings or PollinationsSettings.from_environment()

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload.update({"tools": tools, "tool_choice": "auto"})

        request = Request(
            f"{self.settings.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                # Avoid the generic Python-urllib signature being treated as an
                # automated browser request by edge protection.
                "User-Agent": "AI-Reconciliation-Settlement-QA/1.0",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMClientError(f"Pollinations returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise LLMClientError(f"Could not reach Pollinations: {exc.reason}") from exc

        try:
            parsed = json.loads(body)
            return parsed["choices"][0]["message"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LLMClientError("Pollinations returned an unexpected chat-completion response.") from exc
