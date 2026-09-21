from typing import Protocol


class EmbeddingProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed_text(self, text: str) -> list[float]: ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingProviderError(RuntimeError):
    """A safe provider failure that never contains credentials or raw response bodies."""
