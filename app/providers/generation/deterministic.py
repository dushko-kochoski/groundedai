from app.providers.generation.base import GenerationResult, GroundedModelOutput
from app.providers.generation.prompt import GroundedPrompt


class DeterministicGenerationProvider:
    """Offline extractive provider for repeatable architecture tests, not semantic generation."""

    @property
    def name(self) -> str:
        return "deterministic"

    @property
    def model(self) -> str:
        return "extractive-v2"

    async def generate_grounded_answer(self, prompt: GroundedPrompt) -> GenerationResult:
        if not prompt.sources:
            raise ValueError("deterministic provider requires at least one source")
        selected = [prompt.sources[0]]
        if " and " in prompt.question.casefold():
            seen_documents = {prompt.sources[0].document_id}
            for source in prompt.sources[1:]:
                if source.document_id not in seen_documents:
                    selected.append(source)
                    seen_documents.add(source.document_id)
        return GenerationResult(
            output=GroundedModelOutput(
                answer="\n\n".join(source.text for source in selected),
                citations=[source.alias for source in selected],
                abstained=False,
                abstention_reason=None,
            )
        )
