"""Provider-agnostic LLM client.

Ollama, Groq, Gemini and Hugging Face all offer an OpenAI-compatible API,
so one client works for all of them. Which provider is used is decided by
environment variables, never hard-coded.
"""


from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()  # read settings from a local .env file, if there is one

PROVIDERS = {
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5-coder:3b"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-20b"},
}


class LLMError(RuntimeError):
    """Anything that goes wrong talking to the LLM."""

@dataclass
class LLMConfig:
    provider: str
    base_url: str
    model: str
    api_key: str =field(repr =False)
    
    
def load_config() -> LLMConfig:
    """Read the environment variables and return a config object."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider not in PROVIDERS:
        raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Use one of: {', '.join(PROVIDERS)}")
    defaults = PROVIDERS[provider]
    # Ollama ignores the key but the client needs one; real providers must have it set.
    api_key = os.getenv("LLM_API_KEY", "ollama" if provider == "ollama" else "")
    if not api_key:
        raise LLMError(f"LLM_API_KEY is not set for provider '{provider}'.")
    return LLMConfig(
        provider=provider,
        base_url=os.getenv("LLM_BASE_URL", defaults["base_url"]),
        model=os.getenv("LLM_MODEL", defaults["model"]),
        api_key=api_key,
    )

class LLMClient:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or load_config()
        self._client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=60,
            max_retries=2,
        )

    def chat_json(self, system: str, user: str) -> dict:
        """Send a prompt and return the reply parsed as JSON."""
        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
        except OpenAIError as e:
            raise LLMError(f"Request to {self.config.provider} failed: {e}") from e

        content = response.choices[0].message.content or ""
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMError(f"The model did not return valid JSON: {content[:200]}") from e