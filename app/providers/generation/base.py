from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from app.providers.generation.prompt import GroundedPrompt


class GroundedModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str | None
    citations: list[str]
    abstained: bool
    abstention_reason: str | None

    @field_validator("citations")
    @classmethod
    def validate_source_aliases(cls, value: list[str]) -> list[str]:
        invalid = any(
            not alias.startswith("S") or not alias[1:].isdigit() or alias[1] == "0"
            for alias in value
        )
        if invalid:
            raise ValueError("citations must use source aliases such as S1")
        return value

    @model_validator(mode="after")
    def validate_consistency(self) -> "GroundedModelOutput":
        if self.abstained:
            if self.answer is not None or self.citations:
                raise ValueError("abstained output cannot contain an answer or citations")
            if self.abstention_reason is None or not self.abstention_reason.strip():
                raise ValueError("abstained output requires an abstention reason")
        else:
            if self.answer is None or not self.answer.strip():
                raise ValueError("non-abstained output requires an answer")
            if not self.citations:
                raise ValueError("non-abstained output requires at least one citation")
            if self.abstention_reason is not None:
                raise ValueError("non-abstained output cannot contain an abstention reason")
        return self


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class GenerationResult:
    output: GroundedModelOutput
    usage: TokenUsage | None = None


class GenerationProviderError(RuntimeError):
    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


class GenerationProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def generate_grounded_answer(self, prompt: GroundedPrompt) -> GenerationResult: ...
