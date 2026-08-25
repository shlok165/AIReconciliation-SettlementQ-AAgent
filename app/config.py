"""Runtime configuration for external model access."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class PollinationsSettings:
    api_key: str
    model: str = "openai-fast"
    base_url: str = "https://gen.pollinations.ai/v1"
    timeout_seconds: int = 30

    @classmethod
    def from_environment(cls) -> "PollinationsSettings":
        api_key = os.getenv("POLLINATIONS_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "POLLINATIONS_API_KEY is not configured. Add it to your environment or .env file."
            )
        return cls(api_key=api_key, model=os.getenv("POLLINATIONS_MODEL", "openai-fast"))
