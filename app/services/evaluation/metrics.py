import math
import re
import statistics
from collections.abc import Sequence
from typing import Any

from app.schemas.evaluation import ExpectedFact

_NON_WORD = re.compile(r"[^\w]+", flags=re.UNICODE)


def normalize_text(value: str) -> str:
    return " ".join(_NON_WORD.sub(" ", value.casefold()).split())


def match_facts(answer: str | None, facts: Sequence[ExpectedFact]) -> tuple[int, int, float | None]:
    expected = len(facts)
    if expected == 0:
        return 0, 0, None
    normalized_answer = normalize_text(answer or "")
    matched = sum(
        any(normalize_text(phrase) in normalized_answer for phrase in fact.acceptable_phrases)
        for fact in facts
    )
    return matched, expected, matched / expected


def retrieval_metrics(
    expected_sources: Sequence[str], retrieved_sources: Sequence[str]
) -> tuple[bool, float, float]:
    if not expected_sources:
        return False, 0.0, 0.0
    expected = set(expected_sources)
    found = expected.intersection(retrieved_sources)
    first_rank = next(
        (index for index, source in enumerate(retrieved_sources, start=1) if source in expected),
        None,
    )
    return bool(found), len(found) / len(expected), 0.0 if first_rank is None else 1 / first_rank


def citation_source_metrics(
    expected_sources: Sequence[str], cited_sources: Sequence[str]
) -> tuple[float | None, float | None]:
    if not cited_sources:
        return None, 0.0 if expected_sources else None
    expected = set(expected_sources)
    cited = set(cited_sources)
    precision = len(expected.intersection(cited)) / len(cited)
    recall = len(expected.intersection(cited)) / len(expected) if expected else None
    return precision, recall


def percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def latency_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
    }


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def aggregate_case_metrics(
    cases: Sequence[dict[str, Any]], *, top_k: int, include_generation: bool = True
) -> dict[str, Any]:
    answerable = [case for case in cases if case["answerable"]]
    unanswerable = [case for case in cases if not case["answerable"]]
    completed = [case for case in cases if case.get("error_category") is None]
    correct_answers = sum(
        case["answerable"] and case.get("abstained") is False for case in completed
    )
    correct_abstentions = sum(
        not case["answerable"] and case.get("abstained") is True for case in completed
    )
    false_answers = sum(
        not case["answerable"] and case.get("abstained") is False for case in completed
    )
    false_abstentions = sum(
        case["answerable"] and case.get("abstained") is True for case in completed
    )
    answerable_count = len(answerable)
    total_citations = sum(int(case.get("citations_returned", 0)) for case in cases)
    valid_citations = sum(int(case.get("citations_valid", 0)) for case in cases)
    expected_citations = sum(len(case["expected_sources"]) for case in answerable)
    matching_citations = sum(int(case.get("matching_citations", 0)) for case in cases)
    facts_expected = sum(int(case.get("facts_expected", 0)) for case in answerable)
    facts_matched = sum(int(case.get("facts_matched") or 0) for case in answerable)
    answered_with_tokens = [
        int(case["total_tokens"])
        for case in cases
        if case.get("abstained") is False and case.get("total_tokens") is not None
    ]
    input_tokens = [
        case.get("input_tokens") for case in cases if case.get("input_tokens") is not None
    ]
    output_tokens = [
        case.get("output_tokens") for case in cases if case.get("output_tokens") is not None
    ]
    total_tokens = [
        case.get("total_tokens") for case in cases if case.get("total_tokens") is not None
    ]
    result: dict[str, Any] = {
        "retrieval": {
            "k": top_k,
            "hit_at_k": _safe_ratio(
                sum(bool(case["source_hit"]) for case in answerable), answerable_count
            ),
            "recall_at_k": _safe_ratio(
                sum(float(case["source_recall"]) for case in answerable), answerable_count
            ),
            "mrr": _safe_ratio(
                sum(float(case["reciprocal_rank"]) for case in answerable), answerable_count
            ),
            "zero_result_rate": _safe_ratio(
                sum(not case["retrieved_sources"] for case in cases), len(cases)
            ),
            "average_retrieved_count": _safe_ratio(
                sum(len(case["retrieved_sources"]) for case in cases), len(cases)
            ),
        },
        "answers": {
            "mean_fact_coverage": _safe_ratio(facts_matched, facts_expected),
            "facts_matched": facts_matched,
            "facts_expected": facts_expected,
        },
        "citations": {
            "citation_validity_rate": _safe_ratio(valid_citations, total_citations),
            "source_citation_precision": _safe_ratio(matching_citations, total_citations),
            "source_citation_recall": _safe_ratio(matching_citations, expected_citations),
            "uncited_answer_count": sum(
                case.get("abstained") is False and not case.get("citations_returned")
                for case in completed
            ),
            "invalid_citation_count": sum(
                int(case.get("invalid_citation_count", 0)) for case in cases
            ),
        },
        "abstention": {
            "correct_answers": correct_answers,
            "correct_abstentions": correct_abstentions,
            "false_answers": false_answers,
            "false_abstentions": false_abstentions,
            "abstention_accuracy": _safe_ratio(correct_answers + correct_abstentions, len(cases)),
            "abstention_precision": _safe_ratio(
                correct_abstentions, correct_abstentions + false_abstentions
            ),
            "abstention_recall": _safe_ratio(correct_abstentions, len(unanswerable)),
            "false_answer_rate": _safe_ratio(false_answers, len(unanswerable)),
            "error_count": len(cases) - len(completed),
        },
        "latency": {
            "retrieval": latency_summary([float(case["retrieval_ms"]) for case in cases]),
            "generation": latency_summary(
                [
                    float(case["generation_ms"])
                    for case in cases
                    if case.get("generation_ms") is not None
                ]
            ),
            "total": latency_summary([float(case["total_ms"]) for case in cases]),
        },
        "tokens": {
            "total_input_tokens": sum(input_tokens) if input_tokens else None,
            "total_output_tokens": sum(output_tokens) if output_tokens else None,
            "total_tokens": sum(total_tokens) if total_tokens else None,
            "mean_tokens_per_answered_query": (
                statistics.fmean(answered_with_tokens) if answered_with_tokens else None
            ),
        },
        "cost": {"estimated_cost": None, "currency": None, "pricing_version": None},
    }
    if not include_generation:
        result["answers"] = {
            "mean_fact_coverage": None,
            "facts_matched": None,
            "facts_expected": None,
        }
        result["citations"] = {
            "citation_validity_rate": None,
            "source_citation_precision": None,
            "source_citation_recall": None,
            "uncited_answer_count": None,
            "invalid_citation_count": None,
        }
        result["abstention"] = {
            "correct_answers": None,
            "correct_abstentions": None,
            "false_answers": None,
            "false_abstentions": None,
            "abstention_accuracy": None,
            "abstention_precision": None,
            "abstention_recall": None,
            "false_answer_rate": None,
            "error_count": len(cases) - len(completed),
        }
    return result


def compare_aggregates(
    baseline: dict[str, Any], current: dict[str, Any]
) -> dict[str, float | None]:
    paths = {
        "hit_at_k": ("retrieval", "hit_at_k"),
        "mrr": ("retrieval", "mrr"),
        "fact_coverage": ("answers", "mean_fact_coverage"),
        "citation_precision": ("citations", "source_citation_precision"),
        "false_answer_rate": ("abstention", "false_answer_rate"),
        "p95_total_ms": ("latency", "total", "p95_ms"),
    }
    deltas: dict[str, float | None] = {}
    for name, path in paths.items():
        old: Any = baseline
        new: Any = current
        for key in path:
            old = old.get(key) if isinstance(old, dict) else None
            new = new.get(key) if isinstance(new, dict) else None
        deltas[name] = None if old is None or new is None else float(new) - float(old)
    return deltas
