import asyncio
import math
from collections.abc import Awaitable, Callable

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from app.providers.embeddings.base import EmbeddingProviderError


class _EmbeddingItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    embedding: list[float]
    index: int


class _EmbeddingResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    data: list[_EmbeddingItem]
    model: str


class OpenAIEmbeddingProvider:
    """Validated, batched client for OpenAI's embeddings endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimension: int,
        timeout_seconds: float,
        max_retries: int,
        batch_size: int,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI embedding API key is required")
        self._api_key = api_key
        self._model = model
        self._dimension = dimension
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._batch_size = batch_size
        self._transport = transport
        self._sleep = sleep

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    @property
    def version(self) -> str:
        return "1"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_text(self, text: str) -> list[float]:
        vectors = await self.embed_batch([text])
        return vectors[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        async with httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self._timeout_seconds,
            transport=self._transport,
        ) as client:
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                vectors.extend(await self._embed_request(client, batch))
        return vectors

    async def _embed_request(
        self, client: httpx.AsyncClient, texts: list[str]
    ) -> list[list[float]]:
        response: httpx.Response | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await client.post(
                    "/embeddings",
                    json={
                        "input": texts,
                        "model": self.model,
                        "dimensions": self.dimension,
                        "encoding_format": "float",
                    },
                )
                if response.status_code not in {429, 500, 502, 503, 504}:
                    break
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt >= self._max_retries:
                    raise EmbeddingProviderError("OpenAI embedding request failed") from exc
            if attempt < self._max_retries:
                delay = 0.25 * (2**attempt)
                await self._sleep(delay)

        if response is None:
            raise EmbeddingProviderError("OpenAI embedding request failed")
        if response.status_code >= 400:
            raise EmbeddingProviderError(
                f"OpenAI embedding request returned HTTP {response.status_code}"
            )
        return self._parse_response(response, expected_count=len(texts))

    def _parse_response(
        self, response: httpx.Response, *, expected_count: int
    ) -> list[list[float]]:
        try:
            parsed = _EmbeddingResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise EmbeddingProviderError("OpenAI returned a malformed embedding response") from exc
        if parsed.model != self.model:
            raise EmbeddingProviderError("OpenAI returned an unexpected embedding model")
        ordered = sorted(parsed.data, key=lambda item: item.index)
        if len(ordered) != expected_count or [item.index for item in ordered] != list(
            range(expected_count)
        ):
            raise EmbeddingProviderError("OpenAI returned an unexpected number of embeddings")
        vectors = [item.embedding for item in ordered]
        if any(len(vector) != self.dimension for vector in vectors):
            raise EmbeddingProviderError("OpenAI returned an embedding with the wrong dimension")
        if any(not all(math.isfinite(value) for value in vector) for vector in vectors):
            raise EmbeddingProviderError("OpenAI returned a non-finite embedding value")
        return vectors
