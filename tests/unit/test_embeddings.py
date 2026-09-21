import math

from app.providers.embeddings import DeterministicEmbeddingProvider


async def test_embedding_is_deterministic_and_has_configured_dimension() -> None:
    provider = DeterministicEmbeddingProvider(dimension=256)

    first = await provider.embed_text("Refunds are available within thirty days.")
    second = await provider.embed_text("Refunds are available within thirty days.")

    assert first == second
    assert len(first) == 256
    assert math.isclose(math.sqrt(sum(value * value for value in first)), 1.0)


async def test_shared_tokens_produce_shared_features() -> None:
    provider = DeterministicEmbeddingProvider(dimension=256)

    first, second = await provider.embed_batch(["refund policy", "refund window"])

    assert any(left != 0 and right != 0 for left, right in zip(first, second, strict=True))

