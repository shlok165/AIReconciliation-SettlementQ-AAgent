"""Runtime configuration for external model access."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PollinationsSettings:
    api_key: str = "sk_CWvqQ40a9Xa8pDOGXkUKvxEsv7TWwqy3"
    model: str = "gpt-5.4-mini"
    base_url: str = "https://gen.pollinations.ai/v1"
    timeout_seconds: int = 300

    @classmethod
    def from_environment(cls) -> "PollinationsSettings":
        return cls()