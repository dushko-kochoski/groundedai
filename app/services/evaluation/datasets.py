import json
from pathlib import Path

from pydantic import ValidationError

from app.core.exceptions import ApplicationError
from app.schemas.evaluation import GoldenDataset


class DatasetRegistry:
    """Resolve only application-owned dataset names, never client paths."""

    DATASETS = {"golden_v1": "golden_v1.json"}

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def load(self, name: str) -> GoldenDataset:
        filename = self.DATASETS.get(name)
        if filename is None:
            raise ApplicationError(
                status_code=404,
                code="evaluation_dataset_not_found",
                message="The requested evaluation dataset is not registered.",
                details={"dataset": name},
            )
        path = self.root / "datasets" / filename
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            dataset = GoldenDataset.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise ApplicationError(
                status_code=500,
                code="invalid_evaluation_dataset",
                message="The registered evaluation dataset could not be loaded safely.",
                details={"dataset": name},
            ) from exc
        if dataset.name != name:
            raise ApplicationError(
                status_code=500,
                code="invalid_evaluation_dataset",
                message="The dataset name does not match its registry entry.",
            )
        return dataset

    def fixture_path(self, filename: str) -> Path:
        path = (self.root / "fixtures" / filename).resolve()
        fixtures_root = (self.root / "fixtures").resolve()
        if path.parent != fixtures_root:
            raise ApplicationError(
                status_code=500,
                code="invalid_evaluation_fixture",
                message="An evaluation fixture path escaped the controlled fixture directory.",
            )
        if not path.is_file():
            raise ApplicationError(
                status_code=500,
                code="evaluation_fixture_missing",
                message="A registered evaluation fixture is missing.",
                details={"filename": filename},
            )
        return path
