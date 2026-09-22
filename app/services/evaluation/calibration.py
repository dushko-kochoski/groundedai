import statistics
import uuid
from typing import Any

from app.core.exceptions import ApplicationError
from app.repositories.evaluations import CalibrationCaseSnapshot, EvaluationRepository
from app.services.evaluation.metrics import aggregate_case_metrics


def top_scores(case: CalibrationCaseSnapshot) -> tuple[float | None, float | None]:
    scores = [
        float(item["score"])
        for item in case.retrieval_snapshot
        if isinstance(item.get("score"), (int, float))
    ]
    return (scores[0] if scores else None, scores[1] if len(scores) > 1 else None)


def score_distribution(cases: list[CalibrationCaseSnapshot]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[CalibrationCaseSnapshot]] = {
        "answerable_expected_retrieved": [],
        "answerable_expected_not_retrieved": [],
        "unanswerable_false_answer": [],
        "unanswerable_correct_abstention": [],
    }
    for case in cases:
        if case.answerable and case.source_hit:
            groups["answerable_expected_retrieved"].append(case)
        elif case.answerable:
            groups["answerable_expected_not_retrieved"].append(case)
        elif case.abstained:
            groups["unanswerable_correct_abstention"].append(case)
        else:
            groups["unanswerable_false_answer"].append(case)
    return {name: _distribution(items) for name, items in groups.items()}


def _distribution(cases: list[CalibrationCaseSnapshot]) -> dict[str, Any]:
    pairs = [top_scores(case) for case in cases]
    top1 = [pair[0] for pair in pairs if pair[0] is not None]
    top2 = [pair[1] for pair in pairs if pair[1] is not None]
    gaps = [a - b for a, b in pairs if a is not None and b is not None]
    return {
        "case_count": len(cases),
        "top1": _summary(top1),
        "top2": _summary(top2),
        "top1_top2_gap": _summary(gaps),
    }


def _summary(values: list[float]) -> dict[str, float | None]:
    return {
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "max": max(values) if values else None,
    }


def failure_taxonomy(cases: list[CalibrationCaseSnapshot]) -> dict[str, list[str]]:
    return {
        "weak_irrelevant_evidence_passed_threshold": [
            case.case_id for case in cases if not case.answerable and case.abstained is False
        ],
        "false_answer": [
            case.case_id for case in cases if not case.answerable and case.abstained is False
        ],
        "false_abstention": [
            case.case_id for case in cases if case.answerable and case.abstained is True
        ],
        "expected_fact_missing": [
            case.case_id
            for case in cases
            if case.answerable and (case.facts_matched or 0) < case.facts_expected
        ],
        "multi_source_evidence_not_fully_used": [
            case.case_id
            for case in cases
            if len(case.expected_sources) > 1
            and (case.facts_matched or 0) < case.facts_expected
        ],
        "wrong_or_extra_source_cited": [
            case.case_id
            for case in cases
            if case.answerable
            and case.citation_source_precision is not None
            and case.citation_source_precision < 1.0
        ],
    }


def sweep_thresholds(
    cases: list[CalibrationCaseSnapshot], thresholds: list[float], *, top_k: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in sorted(set(thresholds)):
        metrics = aggregate_case_metrics(
            [_simulate(case, threshold) for case in cases], top_k=top_k
        )
        rows.append({"evidence_min_score": threshold, "metrics": metrics})
    pareto = pareto_thresholds(rows)
    for row in rows:
        row["pareto_relevant"] = row["evidence_min_score"] in pareto
    return rows


def _simulate(case: CalibrationCaseSnapshot, threshold: float) -> dict[str, Any]:
    top1, _ = top_scores(case)
    insufficient = top1 is None or top1 < threshold
    citations = 0 if insufficient else case.citations_returned
    matching = (
        0
        if insufficient or case.citation_source_precision is None
        else round(case.citation_source_precision * case.citations_returned)
    )
    return {
        "answerable": case.answerable,
        "expected_sources": case.expected_sources,
        "retrieved_sources": case.retrieved_sources,
        "source_hit": case.source_hit,
        "source_recall": case.source_recall,
        "reciprocal_rank": case.reciprocal_rank,
        "facts_expected": case.facts_expected,
        "facts_matched": 0 if insufficient else case.facts_matched,
        "abstained": True if insufficient else case.abstained,
        "citations_returned": citations,
        "citations_valid": 0 if insufficient else case.citations_valid,
        "invalid_citation_count": case.invalid_citation_count,
        "matching_citations": matching,
        "retrieval_ms": case.retrieval_ms,
        "generation_ms": 0.0 if insufficient else case.generation_ms,
        "total_ms": case.retrieval_ms if insufficient else case.total_ms,
        "input_tokens": None if insufficient else case.input_tokens,
        "output_tokens": None if insufficient else case.output_tokens,
        "total_tokens": None if insufficient else case.total_tokens,
        "error_category": case.error_category,
    }


def pareto_thresholds(rows: list[dict[str, Any]]) -> set[float]:
    points = [
        (
            float(row["evidence_min_score"]),
            int(row["metrics"]["abstention"]["false_answers"]),
            int(row["metrics"]["abstention"]["false_abstentions"]),
        )
        for row in rows
    ]
    return {
        threshold
        for threshold, false_answers, false_abstentions in points
        if not any(
            (other_answers <= false_answers and other_abstentions <= false_abstentions)
            and (other_answers < false_answers or other_abstentions < false_abstentions)
            for other_threshold, other_answers, other_abstentions in points
            if other_threshold != threshold
        )
    }


class CalibrationService:
    def __init__(self, repository: EvaluationRepository) -> None:
        self.repository = repository

    async def analyze(
        self, run_id: uuid.UUID, thresholds: list[float]
    ) -> dict[str, Any]:
        run = await self.repository.get_run(run_id)
        if run is None:
            raise ApplicationError(
                status_code=404,
                code="evaluation_run_not_found",
                message="The requested baseline evaluation run does not exist.",
            )
        cases = await self.repository.calibration_snapshots(run_id)
        return {
            "baseline_run_id": str(run_id),
            "dataset_name": run.dataset_name,
            "dataset_version": run.dataset_version,
            "embedding_identity": {
                "provider": run.embedding_provider,
                "model": run.embedding_model,
                "version": run.embedding_version,
            },
            "failure_taxonomy": failure_taxonomy(cases),
            "score_distributions": score_distribution(cases),
            "threshold_sweep": sweep_thresholds(
                cases, thresholds, top_k=run.retrieval_top_k
            ),
        }
