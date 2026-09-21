import json

import httpx
import pytest

from app.api.dependencies import get_embedding_provider
from app.core.config import Settings
from app.providers.embeddings import (
    DeterministicEmbeddingProvider,
    EmbeddingProviderError,
    OpenAIEmbeddingProvider,
)


def make_provider(
    handler: httpx.MockTransport,
    *,
    dimension: int = 3,
    retries: int = 0,
    batch_size: int = 2,
) -> OpenAIEmbeddingProvider:
    async def no_sleep(_: float) -> None:
        return None

    return OpenAIEmbeddingProvider(
        api_key="test-placeholder",
        model="text-embedding-3-small",
        dimension=dimension,
        timeout_seconds=1,
        max_retries=retries,
        batch_size=batch_size,
        transport=handler,
        sleep=no_sleep,
    )


async def test_openai_provider_batches_and_restores_response_order() -> None:
    calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload["input"])
        data = [
            {"index": index, "embedding": [float(index + 1), 0.0, 0.0]}
            for index in reversed(range(len(payload["input"])))
        ]
        return httpx.Response(
            200,
            json={"data": data, "model": "text-embedding-3-small"},
        )

    provider = make_provider(httpx.MockTransport(handler))
    vectors = await provider.embed_batch(["one", "two", "three"])

    assert calls == [["one", "two"], ["three"]]
    assert vectors == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([{"index": 0, "embedding": [1.0]}], "wrong dimension"),
        ([], "unexpected number"),
    ],
)
async def test_openai_provider_rejects_malformed_vector_shape(
    data: list[dict[str, object]], message: str
) -> None:
    transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200, json={"data": data, "model": "text-embedding-3-small"}
        )
    )

    with pytest.raises(EmbeddingProviderError, match=message):
        await make_provider(transport).embed_text("test")


async def test_openai_provider_translates_timeout_and_retries() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(EmbeddingProviderError, match="request failed"):
        await make_provider(httpx.MockTransport(handler), retries=2).embed_text("test")

    assert attempts == 3


def test_provider_selection_remains_offline_by_default() -> None:
    provider = get_embedding_provider(Settings())

    assert isinstance(provider, DeterministicEmbeddingProvider)
    assert provider.dimension == 1536


def test_provider_selection_constructs_openai_from_valid_settings() -> None:
    settings = Settings(embedding_provider="openai", embedding_api_key="test-placeholder")

    provider = get_embedding_provider(settings)

    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert provider.model == "text-embedding-3-small"
    assert provider.dimension == 1536
