import argparse
import asyncio
import sys
from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.db.models import EvaluationMode
from app.services.evaluation.factory import build_evaluation_service


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a GroundedAI evaluation dataset")
    parser.add_argument("--dataset", default="golden_v1")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in EvaluationMode],
        default=EvaluationMode.END_TO_END.value,
    )
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


async def _run(args: argparse.Namespace) -> int:
    settings = Settings()
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with session_factory() as session:
            service = build_evaluation_service(session, settings)
            result = await service.run(
                args.dataset,
                EvaluationMode(args.mode),
                top_k=args.top_k,
                min_score=args.min_score,
            )
        if args.as_json:
            print(result.model_dump_json(indent=2))
        else:
            print(
                _summary(
                    result.aggregate_metrics, result.dataset_name, result.case_count, args.mode
                )
            )
        return 0
    except Exception as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        await engine.dispose()


def _summary(metrics: dict[str, Any], dataset: str, cases: int, mode: str) -> str:
    retrieval = metrics["retrieval"]
    answers = metrics["answers"]
    citations = metrics["citations"]
    abstention = metrics["abstention"]
    latency = metrics["latency"]["total"]
    return "\n".join(
        [
            f"Dataset: {dataset}",
            f"Cases: {cases}",
            f"Mode: {mode}",
            "",
            "Retrieval",
            f"Hit@{retrieval['k']}: {_format(retrieval['hit_at_k'])}",
            f"Recall@{retrieval['k']}: {_format(retrieval['recall_at_k'])}",
            f"MRR: {_format(retrieval['mrr'])}",
            "",
            "Answers",
            f"Mean fact coverage: {_format(answers['mean_fact_coverage'])}",
            "",
            "Citations",
            f"Source precision: {_format(citations['source_citation_precision'])}",
            f"Source recall: {_format(citations['source_citation_recall'])}",
            "",
            "Abstention",
            f"Correct abstentions: {abstention['correct_abstentions']}",
            f"False answers: {abstention['false_answers']}",
            f"False abstentions: {abstention['false_abstentions']}",
            "",
            "Latency",
            f"p50 total: {_format(latency['p50_ms'])} ms",
            f"p95 total: {_format(latency['p95_ms'])} ms",
        ]
    )


def _format(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "unavailable"
    return f"{value:.3f}"


def main() -> None:
    raise SystemExit(asyncio.run(_run(_parser().parse_args())))


if __name__ == "__main__":
    main()
