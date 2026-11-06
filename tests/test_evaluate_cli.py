"""Tests for ``python -m mascot_lab evaluate``.

Cases are synthetic, model responses come from ``ModelServer`` (no test
contacts a real Ollama server or runs a model), and runs are written under
``tmp_path``, never to the project's ``results/``.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from evaluation_fakes import AMBIGUOUS_CASE, NAVIGATION_CASE, QUESTION_CASE, ModelServer, write_cases
from ollama_fakes import chat_body

from mascot_lab.cli import main
from mascot_lab.ollama import OllamaError, OllamaErrorCode
from mascot_lab.paths import PROJECT_ROOT


@pytest.fixture
def cases(tmp_path: Path) -> Path:
    return write_cases(tmp_path / "cases.jsonl", NAVIGATION_CASE, QUESTION_CASE, AMBIGUOUS_CASE)


def evaluate(
    capsys: pytest.CaptureFixture[str], *argv: str, transport: ModelServer | None = None
) -> tuple[int, str, str]:
    status = main(["evaluate", *argv], transport=transport)
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def run_dirs(results: Path) -> list[Path]:
    return sorted(path for path in results.iterdir()) if results.exists() else []


def test_a_dry_run_prints_the_plan_and_contacts_nothing(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, cases: Path
) -> None:
    server = ModelServer()
    results = tmp_path / "results"
    status, out, err = evaluate(capsys, "--cases", str(cases), "--results-dir", str(results), "--dry-run", transport=server)
    assert (status, err) == (0, "")
    assert out.startswith("Evaluation plan (dry run: nothing is sent to Ollama and no files are written).")
    assert "3 of 3 selected; categories: synthetic-navigation 2, synthetic-education 1" in out
    assert "records    18 (3 cases x 6 models x 1 repetition); Ollama chat requests: 6" in out
    assert "synthetic-nav-qmind" in out and "application: routed project_qmind  expected route: match" in out
    assert "application: ambiguous_destination project_qmind, project_ingenuity  expected route: match" in out
    assert "route expectations: 3 checked, 0 mismatched" in out
    assert server.requests == []
    assert not results.exists()


def test_a_dry_run_reports_route_mismatches(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    wrong = dict(NAVIGATION_CASE, expected_route={"status": "routed", "route_id": "project_satira"})
    cases = write_cases(tmp_path / "cases.jsonl", wrong)
    status, out, _ = evaluate(capsys, "--cases", str(cases), "--results-dir", str(tmp_path / "r"), "--dry-run")
    assert status == 0
    assert "expected route: mismatch" in out
    assert "route expectations: 1 checked, 1 mismatched" in out


def test_a_run_records_every_response_and_reports_the_results(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, cases: Path
) -> None:
    server = ModelServer()
    results = tmp_path / "results"
    argv = ("--cases", str(cases), "--results-dir", str(results), "--model", "granite4:micro-h", "--model", "Qwen3.5 2B")
    status, out, err = evaluate(capsys, *argv, "--repeat", "2", "--note", "synthetic smoke test", transport=server)
    assert (status, err) == (0, "")
    (directory,) = run_dirs(results)
    assert sorted(path.name for path in directory.iterdir()) == [
        "manifest.json",
        "records.csv",
        "records.jsonl",
        "review.csv",
    ]
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed" and manifest["records_written"] == 12
    assert manifest["note"] == "synthetic smoke test"
    assert [model["tag"] for model in manifest["models"]] == ["granite4:micro-h", "qwen3.5:2b-q4_K_M"]
    assert len(server.chats) == 4 and len(server.shows) == 2
    assert "Every selected model is installed (Ollama /api/show; no model was run)." in out
    assert f"run        {directory.name}" in out
    assert "[ 1/12] granite4:micro-h  synthetic-nav-qmind  r1  routed" in out
    assert "[12/12] qwen3.5:2b-q4_K_M  synthetic-ambiguous  r2  ambiguous_destination" in out
    assert "Run completed: 12 of 12 planned records written." in out
    assert "statuses: answered 4, routed 4, ambiguous_destination 4" in out
    assert "route checks (records): 12 match, 0 mismatch" in out
    assert "Grade in a copy of review.csv using eval/rubric.md" in out


def test_selected_cases_run_in_file_order(capsys: pytest.CaptureFixture[str], tmp_path: Path, cases: Path) -> None:
    results = tmp_path / "results"
    argv = ("--cases", str(cases), "--results-dir", str(results), "--case", AMBIGUOUS_CASE["id"], "--case", NAVIGATION_CASE["id"])
    status, out, _ = evaluate(capsys, *argv, "--model", "granite4:micro-h", transport=ModelServer())
    assert status == 0
    (directory,) = run_dirs(results)
    records = [json.loads(line) for line in (directory / "records.jsonl").read_text(encoding="utf-8").split("\n") if line]
    assert [record["case"]["id"] for record in records] == [NAVIGATION_CASE["id"], AMBIGUOUS_CASE["id"]]


def test_missing_models_exit_with_status_one_and_write_nothing(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, cases: Path
) -> None:
    server = ModelServer(missing={"gemma:2b-instruct-q4_K_M"})
    results = tmp_path / "results"
    status, _, err = evaluate(capsys, "--cases", str(cases), "--results-dir", str(results), transport=server)
    assert status == 1
    assert "error: not installed in Ollama: gemma:2b-instruct-q4_K_M; nothing was run or written" in err
    assert server.chats == []
    assert not results.exists()


def test_an_unreachable_ollama_exits_with_status_one(capsys: pytest.CaptureFixture[str], tmp_path: Path, cases: Path) -> None:
    server = ModelServer()
    server.show_failure = OllamaError(OllamaErrorCode.UNAVAILABLE, "cannot reach Ollama (synthetic)")
    status, _, err = evaluate(capsys, "--cases", str(cases), "--results-dir", str(tmp_path / "results"), transport=server)
    assert status == 1
    assert "cannot reach Ollama (synthetic); nothing was run or written" in err


def test_a_run_that_stops_early_exits_with_status_one(capsys: pytest.CaptureFixture[str], tmp_path: Path, cases: Path) -> None:
    def answer(model: str, message: str) -> object:
        if model == "qwen3.5:2b-q4_K_M":
            return OllamaError(OllamaErrorCode.UNAVAILABLE, "cannot reach Ollama (synthetic)")
        return chat_body("Rastin studies at Queen's University.", model=model)

    status, out, _ = evaluate(
        capsys, "--cases", str(cases), "--results-dir", str(tmp_path / "results"), transport=ModelServer(answer)
    )
    assert status == 1
    assert "Run aborted: 11 of 18 planned records written." in out
    assert "stopped: Ollama could not be reached" in out


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--repeat", "0"), "--repeat must be at least 1"),
        (("--case", "synthetic-absent"), "no case with ID 'synthetic-absent'"),
        (("--model", "llama3:8b"), "not a configured model"),
        (("--note", "   "), "note"),
        (("--ollama-url", "http://example.com:11434"), "loopback"),
        (("--timeout", "-1"), "timeout"),
    ],
)
def test_invalid_arguments_exit_with_status_two(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, cases: Path, extra: tuple[str, ...], message: str
) -> None:
    server = ModelServer()
    results = tmp_path / "results"
    status, out, err = evaluate(capsys, "--cases", str(cases), "--results-dir", str(results), *extra, transport=server)
    assert status == 2
    assert err.startswith("error: ") and message in err
    assert server.requests == []
    assert not results.exists()


def test_invalid_cases_exit_with_status_two(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"id": "synthetic-1", "category": "synthetic", "message": "hi"}\n{"id": "x"}\n', encoding="utf-8")
    status, _, err = evaluate(capsys, "--cases", str(cases), "--results-dir", str(tmp_path / "results"), "--dry-run")
    assert status == 2
    assert f"error: {cases}:2: missing required key(s) category, message" in err


def test_a_missing_case_file_exits_with_status_two(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    status, _, err = evaluate(capsys, "--cases", str(tmp_path / "absent.jsonl"), "--dry-run")
    assert status == 2
    assert "evaluation case file not found" in err


def test_results_inside_the_repository_but_outside_results_are_refused(
    capsys: pytest.CaptureFixture[str], cases: Path
) -> None:
    target = PROJECT_ROOT / "docs" / "synthetic-results"
    status, _, err = evaluate(capsys, "--cases", str(cases), "--results-dir", str(target), "--dry-run")
    assert status == 2
    assert "only results/ is ignored by Git" in err
    assert not target.exists()


def test_the_module_entry_point_runs_a_dry_run(tmp_path: Path, cases: Path) -> None:
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    completed = subprocess.run(
        [sys.executable, "-m", "mascot_lab", "evaluate", "--dry-run", "--cases", str(cases), "--results-dir", str(tmp_path / "r")],
        capture_output=True,
        cwd=PROJECT_ROOT,
        encoding="utf-8",
        env=environment,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "route expectations: 3 checked, 0 mismatched" in completed.stdout
    assert not (tmp_path / "r").exists()
