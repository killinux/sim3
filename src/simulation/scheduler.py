from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import openai

logger = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    request_id: str
    system_prompt: str
    user_prompt: str
    future: asyncio.Future | None = None


@dataclass
class LLMResponse:
    request_id: str
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None


PROVIDER_CONFIGS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "base_url": None,
        "default_model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
}


class LLMScheduler:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 300,
        max_concurrent: int = 64,
        timeout_seconds: float = 30.0,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
    ):
        if provider and provider in PROVIDER_CONFIGS:
            cfg = PROVIDER_CONFIGS[provider]
            base_url = base_url or cfg["base_url"]
            if not model or model == "gpt-4o-mini":
                model = cfg["default_model"]
            if not api_key:
                import os
                api_key = os.environ.get(cfg["env_key"])

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client: openai.AsyncOpenAI | None = None
        self._api_key = api_key
        self._base_url = base_url
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.total_errors = 0
        self._start_time = time.monotonic()

    def _get_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    async def request(self, system_prompt: str, user_prompt: str) -> str:
        async with self._semaphore:
            return await self._call_llm(system_prompt, user_prompt)

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()
        start = time.monotonic()
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                ),
                timeout=self.timeout_seconds,
            )
            self.total_requests += 1
            usage = response.usage
            if usage:
                self.total_input_tokens += usage.prompt_tokens
                self.total_output_tokens += usage.completion_tokens
            return response.choices[0].message.content or ""
        except Exception as e:
            self.total_errors += 1
            logger.warning("LLM call failed: %s", e)
            return ""

    async def batch_request(
        self, prompts: list[tuple[str, str]]
    ) -> list[str]:
        tasks = [self.request(sys, usr) for sys, usr in prompts]
        return await asyncio.gather(*tasks)

    def get_stats(self) -> dict:
        elapsed = time.monotonic() - self._start_time
        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "error_rate": (
                self.total_errors / max(1, self.total_requests + self.total_errors)
            ),
            "elapsed_seconds": round(elapsed, 1),
            "requests_per_second": round(self.total_requests / max(0.1, elapsed), 2),
            "model": self.model,
        }

    async def close(self):
        if self._client:
            await self._client.close()
            self._client = None
