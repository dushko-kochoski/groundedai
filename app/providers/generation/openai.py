import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from pydantic import ValidationError

from app.providers.generation.base import (
    GenerationProviderError,
    GenerationResult,
    GroundedModelOutput,
    TokenUsage,
)
from app.providers.generation.prompt import GroundedPrompt

TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class OpenAIGenerationProvider:
    """Responses API provider using strict JSON Schema Structured Outputs."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_retries: int,
        max_output_tokens: int,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._max_output_tokens = max_output_tokens
        self._transport = transport
        self._sleep = sleep

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    async def generate_grounded_answer(self, prompt: GroundedPrompt) -> GenerationResult:
        async with httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self._timeout_seconds,
            transport=self._transport,
        ) as client:
            response = await self._request(client, prompt)
        return self._parse_response(response)

    async def _request(
        self, client: httpx.AsyncClient, prompt: GroundedPrompt
    ) -> httpx.Response:
        response: httpx.Response | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await client.post(
                    "/responses",
                    json={
                        "model": self.model,
                        "instructions": prompt.instructions,
                        "input": prompt.input_text,
                        "max_output_tokens": self._max_output_tokens,
                        "store": False,
                        "text": {
                            "format": {
                                "type": "json_schema",
                                "name": "grounded_answer",
                                "strict": True,
                                "schema": GroundedModelOutput.model_json_schema(),
                            }
                        },
                    },
                )
                if response.status_code not in TRANSIENT_STATUS_CODES:
                    break
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt >= self._max_retries:
                    raise GenerationProviderError(
                        "provider_timeout", "The generation provider timed out."
                    ) from exc
            if attempt < self._max_retries:
                await self._sleep(0.25 * (2**attempt))

        if response is None:
            raise GenerationProviderError(
                "provider_unavailable", "The generation provider is unavailable."
            )
        if response.status_code in TRANSIENT_STATUS_CODES:
            raise GenerationProviderError(
                "provider_unavailable", "The generation provider is temporarily unavailable."
            )
        if response.status_code >= 400:
            raise GenerationProviderError(
                "provider_request_rejected", "The generation provider rejected the request."
            )
        return response

    def _parse_response(self, response: httpx.Response) -> GenerationResult:
        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise GenerationProviderError(
                "malformed_provider_output", "The generation provider returned invalid JSON."
            ) from exc
        if not isinstance(payload, dict) or payload.get("status") != "completed":
            raise GenerationProviderError(
                "provider_incomplete", "The generation provider did not complete the response."
            )

        output_text: str | None = None
        for item in payload.get("output", []):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "refusal":
                    raise GenerationProviderError(
                        "provider_refusal", "The generation provider declined the request."
                    )
                if content.get("type") == "output_text" and isinstance(
                    content.get("text"), str
                ):
                    output_text = content["text"]
                    break
        if output_text is None:
            raise GenerationProviderError(
                "malformed_provider_output", "The generation provider returned no output text."
            )

        try:
            output = GroundedModelOutput.model_validate(json.loads(output_text))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise GenerationProviderError(
                "malformed_provider_output",
                "The generation provider returned an invalid structured answer.",
            ) from exc
        return GenerationResult(output=output, usage=self._parse_usage(payload.get("usage")))

    @staticmethod
    def _parse_usage(value: object) -> TokenUsage | None:
        if not isinstance(value, dict):
            return None
        input_tokens = value.get("input_tokens")
        output_tokens = value.get("output_tokens")
        total_tokens = value.get("total_tokens")
        if not isinstance(input_tokens, int) or isinstance(input_tokens, bool):
            return None
        if not isinstance(output_tokens, int) or isinstance(output_tokens, bool):
            return None
        if not isinstance(total_tokens, int) or isinstance(total_tokens, bool):
            return None
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
