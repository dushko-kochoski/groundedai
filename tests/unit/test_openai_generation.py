import json

import httpx
import pytest

from app.api.dependencies import get_generation_provider
from app.core.config import Settings
from app.providers.generation import (
    DeterministicGenerationProvider,
    GenerationProviderError,
    OpenAIGenerationProvider,
)
from app.providers.generation.prompt import GroundedPrompt, PromptSource


def prompt() -> GroundedPrompt:
    return GroundedPrompt(
        version="grounded_answer_v1",
        question="What is the policy?",
        instructions="Use evidence only.",
        input_text='UNTRUSTED EVIDENCE [{"source_id":"S1","text":"Policy text"}]',
        sources=(
            PromptSource(
                alias="S1",
                chunk_id="chunk",
                document_id="document",
                filename="policy.txt",
                page_number=None,
                section=None,
                text="Policy text",
            ),
        ),
    )


def response_body(
    output: dict[str, object] | None = None,
    *,
    usage: dict[str, int] | None = None,
) -> dict[str, object]:
    structured = output or {
        "answer": "Policy text",
        "citations": ["S1"],
        "abstained": False,
        "abstention_reason": None,
    }
    return {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": json.dumps(structured)}],
            }
        ],
        "usage": usage or {"input_tokens": 10, "output_tokens": 6, "total_tokens": 16},
    }


def provider(
    transport: httpx.MockTransport,
    *,
    retries: int = 0,
) -> OpenAIGenerationProvider:
    async def no_sleep(_: float) -> None:
        return None

    return OpenAIGenerationProvider(
        api_key="test-placeholder",
        model="gpt-5-mini",
        timeout_seconds=1,
        max_retries=retries,
        max_output_tokens=400,
        transport=transport,
        sleep=no_sleep,
    )


async def test_responses_request_uses_strict_schema_and_extracts_usage() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=response_body())

    result = await provider(httpx.MockTransport(handler)).generate_grounded_answer(prompt())
    body = captured["body"]
    assert isinstance(body, dict)

    assert captured["path"] == "/v1/responses"
    assert body["model"] == "gpt-5-mini"
    assert body["store"] is False
    assert body["input"] == prompt().input_text
    text_format = body["text"]["format"]
    assert text_format["type"] == "json_schema"
    assert text_format["strict"] is True
    assert text_format["schema"]["additionalProperties"] is False
    assert result.output.citations == ["S1"]
    assert result.usage is not None
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 6
    assert result.usage.total_tokens == 16


async def test_transient_error_retries_then_succeeds() -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=response_body())

    result = await provider(httpx.MockTransport(handler), retries=1).generate_grounded_answer(
        prompt()
    )

    assert result.output.answer == "Policy text"
    assert attempts == 2


async def test_timeout_is_retried_and_translated() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("timeout", request=request)

    with pytest.raises(GenerationProviderError) as error:
        await provider(httpx.MockTransport(handler), retries=2).generate_grounded_answer(prompt())

    assert error.value.category == "provider_timeout"
    assert attempts == 3


async def test_non_retryable_error_is_not_retried() -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400)

    with pytest.raises(GenerationProviderError) as error:
        await provider(httpx.MockTransport(handler), retries=2).generate_grounded_answer(prompt())

    assert error.value.category == "provider_request_rejected"
    assert attempts == 1


async def test_refusal_and_malformed_structured_output_fail_safely() -> None:
    refusal = httpx.MockTransport(
        lambda _: httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {"type": "message", "content": [{"type": "refusal", "refusal": "No"}]}
                ],
            },
        )
    )
    malformed = httpx.MockTransport(
        lambda _: httpx.Response(
            200,
            json=response_body(
                {
                    "answer": None,
                    "citations": [],
                    "abstained": False,
                    "abstention_reason": None,
                }
            ),
        )
    )

    with pytest.raises(GenerationProviderError) as refusal_error:
        await provider(refusal).generate_grounded_answer(prompt())
    with pytest.raises(GenerationProviderError) as malformed_error:
        await provider(malformed).generate_grounded_answer(prompt())

    assert refusal_error.value.category == "provider_refusal"
    assert malformed_error.value.category == "malformed_provider_output"


def test_generation_provider_configuration_and_selection() -> None:
    deterministic = get_generation_provider(Settings(_env_file=None))
    openai = get_generation_provider(
        Settings(
            _env_file=None,
            generation_provider="openai",
            openai_api_key="test-placeholder",
        )
    )

    assert isinstance(deterministic, DeterministicGenerationProvider)
    assert isinstance(openai, OpenAIGenerationProvider)
    assert openai.model == "gpt-5-mini"
