import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.exceptions import ApplicationError
from app.schemas.evaluation import GoldenDataset
from app.services.evaluation.datasets import DatasetRegistry


def test_golden_v1_dataset_is_valid_and_versioned() -> None:
    dataset = DatasetRegistry(Path("evals")).load("golden_v1")

    assert dataset.name == "golden_v1"
    assert dataset.version == "1.0.0"
    assert len(dataset.cases) == 30
    assert sum(not case.answerable for case in dataset.cases) == 10
    assert any("pdf" in case.tags for case in dataset.cases)
    assert any("docx" in case.tags for case in dataset.cases)


def test_holdout_v1_dataset_is_registered_valid_and_versioned() -> None:
    dataset = DatasetRegistry(Path("evals")).load("holdout_v1")

    assert dataset.name == "holdout_v1"
    assert dataset.version == "1.0.0"
    assert len(dataset.cases) == 14
    assert sum(case.answerable for case in dataset.cases) == 9
    assert sum(not case.answerable for case in dataset.cases) == 5
    assert any("pdf" in case.tags for case in dataset.cases)
    assert any("docx" in case.tags for case in dataset.cases)
    assert any("multi-document" in case.tags for case in dataset.cases)


def test_unknown_dataset_cannot_be_used_as_a_path() -> None:
    with pytest.raises(ApplicationError, match="not registered"):
        DatasetRegistry(Path("evals")).load("../../secrets")


def test_invalid_dataset_rejects_answerable_case_without_expectations() -> None:
    payload = {
        "name": "test",
        "version": "1",
        "documents": [{"id": "doc", "filename": "doc.txt"}],
        "cases": [
            {
                "id": "bad",
                "question": "Question?",
                "answerable": True,
                "expected_facts": [],
                "expected_sources": [],
            }
        ],
    }

    with pytest.raises(ValidationError):
        GoldenDataset.model_validate_json(json.dumps(payload))
