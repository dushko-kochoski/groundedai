import pytest

from app.schemas.evaluation import ExpectedFact
from app.services.evaluation.metrics import (
    aggregate_case_metrics,
    citation_source_metrics,
    compare_aggregates,
    match_facts,
    percentile,
    retrieval_metrics,
)


def test_hit_recall_and_mrr_use_expected_source_identity() -> None:
    hit, recall, reciprocal_rank = retrieval_metrics(["a", "b"], ["x", "b", "a"])

    assert hit is True
    assert recall == 1.0
    assert reciprocal_rank == 0.5


def test_missed_retrieval_has_zero_metrics() -> None:
    assert retrieval_metrics(["a"], ["x"]) == (False, 0.0, 0.0)


def test_fact_matching_normalizes_case_whitespace_and_punctuation() -> None:
    facts = [ExpectedFact(id="window", acceptable_phrases=["within 30 calendar days"])]

    assert match_facts("Refunds apply WITHIN  30 calendar-days.", facts) == (1, 1, 1.0)


def test_partial_fact_coverage_is_deterministic() -> None:
    facts = [
        ExpectedFact(id="one", acceptable_phrases=["alpha beta"]),
        ExpectedFact(id="two", acceptable_phrases=["gamma delta", "alternate phrase"]),
    ]

    assert match_facts("Alpha beta only", facts) == (1, 2, 0.5)


def test_citation_precision_and_recall() -> None:
    assert citation_source_metrics(["a", "b"], ["a", "x"]) == (0.5, 0.5)
    assert citation_source_metrics(["a"], []) == (None, 0.0)


def test_percentile_uses_nearest_rank_and_rejects_invalid_probability() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
    with pytest.raises(ValueError):
        percentile([1.0], 1.1)


def _case(**overrides: object) -> dict[str, object]:
    case: dict[str, object] = {
        "answerable": True,
        "expected_sources": ["a"],
        "retrieved_sources": ["a"],
        "source_hit": True,
        "source_recall": 1.0,
        "reciprocal_rank": 1.0,
        "facts_expected": 1,
        "facts_matched": 1,
        "abstained": False,
        "citations_returned": 1,
        "citations_valid": 1,
        "invalid_citation_count": 0,
        "matching_citations": 1,
        "retrieval_ms": 1.0,
        "generation_ms": 2.0,
        "total_ms": 3.0,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "error_category": None,
    }
    case.update(overrides)
    return case


def test_aggregate_metrics_include_abstention_confusion_and_missing_tokens() -> None:
    cases = [
        _case(),
        _case(
            answerable=False,
            expected_sources=[],
            retrieved_sources=[],
            source_hit=False,
            source_recall=0.0,
            reciprocal_rank=0.0,
            facts_expected=0,
            facts_matched=0,
            abstained=True,
            citations_returned=0,
            citations_valid=0,
            matching_citations=0,
        ),
        _case(
            answerable=False,
            expected_sources=[],
            source_hit=False,
            source_recall=0.0,
            reciprocal_rank=0.0,
            facts_expected=0,
            facts_matched=0,
            abstained=False,
            matching_citations=0,
        ),
        _case(abstained=True, citations_returned=0, citations_valid=0, matching_citations=0),
    ]

    result = aggregate_case_metrics(cases, top_k=5)

    assert result["retrieval"]["hit_at_k"] == 1.0
    assert result["abstention"]["correct_answers"] == 1
    assert result["abstention"]["correct_abstentions"] == 1
    assert result["abstention"]["false_answers"] == 1
    assert result["abstention"]["false_abstentions"] == 1
    assert result["abstention"]["false_answer_rate"] == 0.5
    assert result["tokens"]["total_tokens"] is None
    assert result["cost"]["estimated_cost"] is None


def test_run_comparison_reports_metric_deltas() -> None:
    baseline = aggregate_case_metrics([_case(total_ms=10.0)], top_k=5)
    current = aggregate_case_metrics([_case(total_ms=8.0)], top_k=5)

    deltas = compare_aggregates(baseline, current)

    assert deltas["hit_at_k"] == 0.0
    assert deltas["p95_total_ms"] == -2.0


def test_run_comparison_derives_historical_false_abstention_rate() -> None:
    baseline = {
        "abstention": {"correct_answers": 20, "false_abstentions": 0},
    }
    current = {
        "abstention": {
            "correct_answers": 17,
            "false_abstentions": 3,
            "false_abstention_rate": 0.15,
        },
    }

    deltas = compare_aggregates(baseline, current)

    assert deltas["false_abstention_rate"] == pytest.approx(0.15)


def test_retrieval_only_aggregate_marks_generation_metrics_unavailable() -> None:
    result = aggregate_case_metrics([_case()], top_k=5, include_generation=False)

    assert result["retrieval"]["hit_at_k"] == 1.0
    assert result["answers"]["mean_fact_coverage"] is None
    assert result["citations"]["citation_validity_rate"] is None
    assert result["abstention"]["false_answer_rate"] is None
