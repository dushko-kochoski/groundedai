from app.providers.generation.base import GenerationResult, GroundedModelOutput
from app.providers.generation.prompt import GroundedPrompt


class DeterministicGenerationProvider:
    """Offline provider that returns the first bounded evidence chunk with a trusted alias."""

    @property
    def name(self) -> str:
        return "deterministic"

    @property
    def model(self) -> str:
        return "extractive-v1"

    async def generate_grounded_answer(self, prompt: GroundedPrompt) -> GenerationResult:
        if not prompt.sources:
            raise ValueError("deterministic provider requires at least one source")
        first = prompt.sources[0]
        return GenerationResult(
            output=GroundedModelOutput(
                answer=first.text,
                citations=[first.alias],
                abstained=False,
                abstention_reason=None,
            )
        )
