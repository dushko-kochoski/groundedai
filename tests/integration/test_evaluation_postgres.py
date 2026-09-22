import json
import os
from pathlib import Path

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.db.models import EvaluationCaseResult, EvaluationMode, EvaluationRun
from app.services.evaluation.factory import build_evaluation_service

pytestmark = pytest.mark.integration


async def test_evaluation_persists_run_cases_and_matching_aggregates(tmp_path: Path) -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not configured")

    evaluation_root = tmp_path / "evals"
    fixtures = evaluation_root / "fixtures"
    datasets = evaluation_root / "datasets"
    fixtures.mkdir(parents=True)
    datasets.mkdir(parents=True)
    (fixtures / "evaluation-integration-policy.txt").write_text(
        "Remote work employees may work remotely two days each week.", encoding="utf-8"
    )
    payload = {
        "name": "golden_v1",
        "version": "integration-1",
        "documents": [{"id": "policy", "filename": "evaluation-integration-policy.txt"}],
        "cases": [
            {
                "id": "supported",
                "question": "remote work employees two days each week",
                "answerable": True,
                "expected_facts": [
                    {
                        "id": "remote_days",
                        "acceptable_phrases": ["two days each week"],
                    }
                ],
                "expected_sources": ["policy"],
            },
            {
                "id": "absent",
                "question": "volcano magma tectonic plates",
                "answerable": False,
                "expected_facts": [],
                "expected_sources": [],
            },
        ],
    }
    (datasets / "golden_v1.json").write_text(json.dumps(payload), encoding="utf-8")

    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url=database_url,
        upload_dir=tmp_path / "uploads",
        evaluation_root=evaluation_root,
        chunk_target_chars=300,
        chunk_overlap_chars=20,
        retrieval_top_k=3,
        retrieval_min_score=0.15,
    )
    engine = create_async_engine(database_url, pool_pre_ping=True)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    run_id = None
    try:
        async with sessions() as session:
            service = build_evaluation_service(session, settings)
            response = await service.run("golden_v1", EvaluationMode.END_TO_END)
            run_id = response.run_id
            fetched = await service.get_run(run_id)
            cases = await service.list_cases(run_id)

        assert fetched.status.value == "completed"
        assert fetched.evidence_policy_version == "evidence_policy_v1"
        assert fetched.evidence_min_score == 0.26
        assert len(cases.items) == 2
        assert response.aggregate_metrics["retrieval"]["hit_at_k"] == 1.0  # type: ignore[index]
        assert response.aggregate_metrics["answers"]["mean_fact_coverage"] == 1.0  # type: ignore[index]
        assert response.aggregate_metrics["abstention"]["correct_abstentions"] == 1  # type: ignore[index]
        supported = next(case for case in cases.items if case.case_id == "supported")
        absent = next(case for case in cases.items if case.case_id == "absent")
        assert supported.query_run_id is not None
        assert supported.citation_source_precision == 1.0
        assert absent.abstained is True

        async with sessions() as session:
            persisted_run = await session.scalar(
                select(EvaluationRun).where(EvaluationRun.id == run_id)
            )
            persisted_cases = list(
                (
                    await session.scalars(
                        select(EvaluationCaseResult).where(
                            EvaluationCaseResult.evaluation_run_id == run_id
                        )
                    )
                ).all()
            )
        assert persisted_run is not None
        assert persisted_run.aggregate_metrics == response.aggregate_metrics
        assert persisted_run.evidence_policy_version == "evidence_policy_v1"
        assert persisted_run.evidence_min_score == 0.26
        assert len(persisted_cases) == 2
    finally:
        async with engine.begin() as connection:
            if run_id is not None:
                await connection.execute(
                    text("DELETE FROM evaluation_runs WHERE id = :run_id"), {"run_id": run_id}
                )
            await connection.execute(
                text("DELETE FROM query_runs WHERE question IN (:supported, :absent)"),
                {
                    "supported": "remote work employees two days each week",
                    "absent": "volcano magma tectonic plates",
                },
            )
            await connection.execute(
                text("DELETE FROM documents WHERE filename = 'evaluation-integration-policy.txt'")
            )
        await engine.dispose()
