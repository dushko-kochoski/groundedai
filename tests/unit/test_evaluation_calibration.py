from typing import Any

from app.repositories.evaluations import CalibrationCaseSnapshot
from app.services.evaluation.calibration import (
    failure_taxonomy,
    pareto_thresholds,
    score_distribution,
    sweep_thresholds,
)


def snapshot(
    case_id: str,
    *,
    answerable: bool,
    score: float | None,
    abstained: bool,
    facts_expected: int = 1,
    facts_matched: int = 1,
) -> CalibrationCaseSnapshot:
    expected = ["policy"] if answerable else []
    return CalibrationCaseSnapshot(
        case_id=case_id,
        answerable=answerable,
        expected_sources=expected,
        retrieved_sources=["policy"] if score is not None else [],
        facts_expected=facts_expected if answerable else 0,
        facts_matched=facts_matched if answerable else 0,
        abstained=abstained,
        citations_returned=0 if abstained else 1,
        citations_valid=0 if abstained else 1,
        invalid_citation_count=0,
        citation_source_precision=1.0 if answerable and not abstained else 0.0,
        retrieval_ms=1.0,
        generation_ms=0.0 if abstained else 1.0,
        total_ms=1.0 if abstained else 2.0,
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        error_category=None,
        source_hit=answerable and score is not None,
        source_recall=1.0 if answerable and score is not None else 0.0,
        reciprocal_rank=1.0 if answerable and score is not None else 0.0,
        retrieval_snapshot=[] if score is None else [{"score": score}],
    )


def metrics_at(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    return next(row["metrics"] for row in rows if row["evidence_min_score"] == threshold)


def test_threshold_sweep_reports_false_answer_false_abstention_tradeoff() -> None:
    cases = [
        snapshot("strong", answerable=True, score=0.8, abstained=False),
        snapshot("weak-valid", answerable=True, score=0.2, abstained=False),
        snapshot("weak-invalid", answerable=False, score=0.19, abstained=False),
    ]

    rows = sweep_thresholds(cases, [0.15, 0.21], top_k=5)
    low = metrics_at(rows, 0.15)
    high = metrics_at(rows, 0.21)

    assert low["abstention"]["false_answers"] == 1
    assert low["abstention"]["false_abstentions"] == 0
    assert high["abstention"]["false_answers"] == 0
    assert high["abstention"]["false_abstentions"] == 1
    assert high["retrieval"] == low["retrieval"]


def test_threshold_boundary_is_inclusive() -> None:
    rows = sweep_thresholds(
        [snapshot("boundary", answerable=True, score=0.26, abstained=False)],
        [0.26],
        top_k=5,
    )

    assert metrics_at(rows, 0.26)["abstention"]["correct_answers"] == 1


def test_score_distribution_separates_outcome_groups() -> None:
    distribution = score_distribution(
        [
            snapshot("answer", answerable=True, score=0.8, abstained=False),
            snapshot("false", answerable=False, score=0.2, abstained=False),
            snapshot("correct", answerable=False, score=None, abstained=True),
        ]
    )

    assert distribution["answerable_expected_retrieved"]["top1"]["mean"] == 0.8
    assert distribution["unanswerable_false_answer"]["top1"]["mean"] == 0.2
    assert distribution["unanswerable_correct_abstention"]["top1"]["mean"] is None


def test_failure_taxonomy_is_reproducible() -> None:
    cases = [
        snapshot("false", answerable=False, score=0.2, abstained=False),
        snapshot(
            "missing-fact",
            answerable=True,
            score=0.8,
            abstained=False,
            facts_expected=2,
            facts_matched=1,
        ),
    ]

    taxonomy = failure_taxonomy(cases)

    assert taxonomy["false_answer"] == ["false"]
    assert taxonomy["weak_irrelevant_evidence_passed_threshold"] == ["false"]
    assert taxonomy["expected_fact_missing"] == ["missing-fact"]


def test_pareto_reporting_excludes_dominated_configuration() -> None:
    rows = [
        {
            "evidence_min_score": 0.1,
            "metrics": {"abstention": {"false_answers": 3, "false_abstentions": 0}},
        },
        {
            "evidence_min_score": 0.2,
            "metrics": {"abstention": {"false_answers": 1, "false_abstentions": 1}},
        },
        {
            "evidence_min_score": 0.3,
            "metrics": {"abstention": {"false_answers": 2, "false_abstentions": 2}},
        },
    ]

    assert pareto_thresholds(rows) == {0.1, 0.2}
