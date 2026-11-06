"""The evaluation documentation names every field the code writes.

These checks keep ``docs/architecture.md`` and the README from drifting away
from the record formats. They read the documents only.
"""

import json
from pathlib import Path

import pytest
from evaluation_fakes import QUESTION_CASE, ModelServer, write_cases

from mascot_lab.cases import load_cases
from mascot_lab.engine import ChatEngine
from mascot_lab.measurements import MEASUREMENT_FIELDS
from mascot_lab.ollama import OllamaClient
from mascot_lab.paths import PROJECT_ROOT
from mascot_lab.recorder import RECORD_COLUMNS, RUN_FILES
from mascot_lab.review import DIMENSIONS, GRADES, REVIEW_COLUMNS
from mascot_lab.runner import EvaluationRunner, RunStatus, plan_evaluation

ARCHITECTURE = (PROJECT_ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
README = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("column", RECORD_COLUMNS)
def test_every_records_csv_column_is_documented(column: str) -> None:
    assert f"`{column}`" in ARCHITECTURE


@pytest.mark.parametrize("column", REVIEW_COLUMNS)
def test_every_review_sheet_column_is_documented(column: str) -> None:
    assert f"`{column}`" in ARCHITECTURE


@pytest.mark.parametrize("field", MEASUREMENT_FIELDS)
def test_every_measurement_is_documented_with_its_null_rule(field: str) -> None:
    assert f"`{field}`" in ARCHITECTURE
    assert f"`{field}`" in README


def test_run_files_statuses_dimensions_and_grades_are_documented() -> None:
    for name in RUN_FILES.values():
        assert f"`{name}`" in ARCHITECTURE and f"`{name}`" in README
    for status in RunStatus:
        assert f"`{status.value}`" in ARCHITECTURE
    for dimension in DIMENSIONS:
        assert f"`{dimension.id}`" in README
    for grade in GRADES:
        assert f"`{grade}`" in ARCHITECTURE


def test_every_case_key_is_documented() -> None:
    for key in ("id", "category", "message", "expected_behavior", "expected_route", "notes"):
        assert f"| `{key}` |" in ARCHITECTURE


def test_every_manifest_and_record_key_is_documented(tmp_path: Path) -> None:
    engine = ChatEngine.from_files(client=OllamaClient(transport=ModelServer()))
    case_set = load_cases(write_cases(tmp_path / "cases.jsonl", QUESTION_CASE), route_ids=engine.router.config.route_ids)
    result = EvaluationRunner(engine, results_dir=tmp_path / "results").run(
        plan_evaluation(engine, case_set, models=["granite4:micro-h"])
    )
    manifest = json.loads((result.directory / "manifest.json").read_text(encoding="utf-8"))
    record = json.loads((result.directory / "records.jsonl").read_text(encoding="utf-8"))
    for key in [*manifest, *record]:
        assert f"`{key}`" in ARCHITECTURE, key
