import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.repositories.evaluations import EvaluationRepository
from app.services.evaluation.calibration import CalibrationService

DEFAULT_THRESHOLDS = [0.15, 0.18, 0.20, 0.22, 0.24, 0.25, 0.251, 0.259, 0.26, 0.28, 0.30]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate evidence strength from a frozen run")
    parser.add_argument("--run-id", type=uuid.UUID, required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--output", type=Path)
    return parser


async def _run(args: argparse.Namespace) -> int:
    settings = Settings()
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessions() as session:
            result = await CalibrationService(EvaluationRepository(session)).analyze(
                args.run_id, args.thresholds
            )
        rendered = json.dumps(result, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(_table(result))
        return 0
    except Exception as exc:
        print(f"calibration failed: {exc}", file=sys.stderr)
        return 1
    finally:
        await engine.dispose()


def _table(result: dict[str, object]) -> str:
    rows = result["threshold_sweep"]
    assert isinstance(rows, list)
    lines = ["score   CA  CAb  FA  FAb  FAR    facts  hit@k  MRR    pareto"]
    for row in rows:
        assert isinstance(row, dict)
        metrics = row["metrics"]
        assert isinstance(metrics, dict)
        abstention = metrics["abstention"]
        answers = metrics["answers"]
        retrieval = metrics["retrieval"]
        assert isinstance(abstention, dict)
        assert isinstance(answers, dict)
        assert isinstance(retrieval, dict)
        lines.append(
            f"{float(row['evidence_min_score']):.3f}  "
            f"{abstention['correct_answers']:>2}  {abstention['correct_abstentions']:>3}  "
            f"{abstention['false_answers']:>2}  {abstention['false_abstentions']:>3}  "
            f"{float(abstention['false_answer_rate']):.3f}  "
            f"{float(answers['mean_fact_coverage']):.3f}  "
            f"{float(retrieval['hit_at_k']):.3f}  {float(retrieval['mrr']):.3f}  "
            f"{'yes' if row['pareto_relevant'] else 'no'}"
        )
    return "\n".join(lines)


def main() -> None:
    raise SystemExit(asyncio.run(_run(_parser().parse_args())))


if __name__ == "__main__":
    main()
