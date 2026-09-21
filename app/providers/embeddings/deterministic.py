import hashlib
import math
import re


class DeterministicEmbeddingProvider:
    """Normalized lexical feature hashing for local, repeatable development."""

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in re.findall(r"[\w']+", text.casefold()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            bucket = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[bucket] += sign

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude:
            vector = [value / magnitude for value in vector]
        return vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_text(text) for text in texts]

