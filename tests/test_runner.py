"""Tests for the evaluation runner and recorder.

Every case is synthetic, every model response comes from ``ModelServer``
(no test contacts a real Ollama server or runs a model), and every run is
written under ``tmp_path``, never to the project's ``results/``. Counts and
durations in mocked responses are synthetic test data.
"""

import csv
import json
import random
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from evaluation_fakes import (
    AMBIGUOUS_CASE,
    ANSWER,
    NAVIGATION_CASE,
    QUESTION_CASE,
    ModelServer,
    answer_with,
    write_cases,
)
from ollama_fakes import chat_body

from mascot_lab import KnowledgeBlock, KnowledgeError, ModelsConfig, load_knowledge, load_system_instruction
from mascot_lab._textfile import sha256_hex
from mascot_lab.cases import CaseSet, load_cases
from mascot_lab.chat_config import load_chat_config
from mascot_lab.config import load_models_config
from mascot_lab.engine import ChatEngine, ChatStatus
from mascot_lab.errors import ConfigError
from mascot_lab.measurements import MEASUREMENT_FIELDS, tokens_per_second, wrapped_line_count
from mascot_lab.ollama import HttpResponse, OllamaClient, OllamaError, OllamaErrorCode
from mascot_lab.output import cloud_preview
from mascot_lab.paths import PROJECT_ROOT
from mascot_lab.prompt import PromptBuilder
from mascot_lab.recorder import (
    RECORD_COLUMNS,
    RESULTS_DIR,
    RUN_FILES,
    check_results_dir,
    csv_cell,
    json_line,
    record_row,
)
from mascot_lab.review import DIMENSIONS, REVIEW_COLUMNS, RUBRIC_PATH
from mascot_lab.router import Router, RouteStatus
from mascot_lab.runner import (
    EvaluationRunner,
    ModelsNotInstalledError,
    RecordProgress,
    RouteCheck,
    RunResult,
    RunStart,
    RunStatus,
    new_run_id,
    plan_evaluation,
    record_id,
)

ROUTE_IDS = Router.from_file().config.route_ids
FIXED_NOW = datetime(2026, 11, 6, 12, 0, 0, tzinfo=UTC)
RUN_ID = "synthetic-run"
OLLAMA_FIELDS = (
    "load_duration_ns",
    "prompt_eval_duration_ns",
    "eval_duration_ns",
    "total_duration_ns",
    "prompt_tokens",
    "output_tokens",
)


class Ticker:
    """A clock that advances by ``step`` seconds each time it is read."""

    def __init__(self, step: float) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.value
        self.value += self.step
        return value


@pytest.fixture
def server() -> ModelServer:
    return ModelServer()


def make_engine(
    server: ModelServer,
    *,
    knowledge_loader: Callable[[], KnowledgeBlock] | None = None,
    client_clock: Callable[[], float] | None = None,
) -> ChatEngine:
    client = OllamaClient(transport=server) if client_clock is None else OllamaClient(transport=server, clock=client_clock)
    if knowledge_loader is None:
        return ChatEngine.from_files(client=client)
    return ChatEngine(
        models=load_models_config(),
        router=Router.from_file(),
        chat=load_chat_config(),
        instruction=load_system_instruction(),
        knowledge_loader=knowledge_loader,
        client=client,
    )


def load(tmp_path: Path, *cases: dict[str, Any]) -> CaseSet:
    return load_cases(write_cases(tmp_path / "cases.jsonl", *cases), route_ids=ROUTE_IDS)


def runner_for(engine: ChatEngine, tmp_path: Path, **options: Any) -> EvaluationRunner:
    return EvaluationRunner(engine, results_dir=tmp_path / "results", now=lambda: FIXED_NOW, **options)


def run(engine: ChatEngine, case_set: CaseSet, tmp_path: Path, **plan_options: Any) -> RunResult:
    plan = plan_evaluation(engine, case_set, **plan_options)
    return runner_for(engine, tmp_path).run(plan, run_id=RUN_ID)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return list(reader.fieldnames or []), rows


def outputs(result: RunResult) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    directory = result.directory
    manifest = json.loads((directory / RUN_FILES["manifest"]).read_text(encoding="utf-8"))
    records = read_jsonl(directory / RUN_FILES["records_jsonl"])
    _, rows = read_csv(directory / RUN_FILES["records_csv"])
    _, review = read_csv(directory / RUN_FILES["review_sheet"])
    return manifest, records, rows, review


# ------------------------------------------------------------------ the run


def test_every_model_gets_every_case_model_by_model(
    tmp_path: Path, server: ModelServer, models_config: ModelsConfig
) -> None:
    case_set = load(tmp_path, NAVIGATION_CASE, QUESTION_CASE, AMBIGUOUS_CASE)
    result = run(make_engine(server), case_set, tmp_path, repetitions=2)
    assert result.status is RunStatus.COMPLETED
    assert result.stop_reason is None
    assert result.planned_records == result.records_written == 3 * 6 * 2
    _, records, rows, review = outputs(result)
    order = [(record["model"]["tag"], record["repetition"], record["case"]["id"]) for record in records]
    assert order == [
        (tag, repetition, case_id)
        for tag in models_config.tags
        for repetition in (1, 2)
        for case_id in case_set.ids
    ]
    assert [record["sequence"] for record in records] == list(range(1, 37))
    assert len(rows) == len(review) == 36
    assert result.status_counts == {"answered": 12, "routed": 12, "ambiguous_destination": 12}


def test_only_cases_that_need_a_model_reach_ollama(
    tmp_path: Path, server: ModelServer, models_config: ModelsConfig
) -> None:
    result = run(make_engine(server), load(tmp_path, NAVIGATION_CASE, QUESTION_CASE), tmp_path)
    assert [request.payload["model"] for request in server.chats] == list(models_config.tags)
    assert all(request.payload["messages"][-1]["content"] == QUESTION_CASE["message"] for request in server.chats)
    assert [request.payload for request in server.shows] == [{"model": tag} for tag in models_config.tags]
    _, records, _, _ = outputs(result)
    routed = [record for record in records if record["case"]["id"] == NAVIGATION_CASE["id"]]
    assert len(routed) == 6
    for record in routed:
        assert record["status"] == "routed" and record["request_sent"] is False
        assert record["response"]["output"] is None and record["response"]["prompt"] is None
        assert record["response"]["actions"] == [{"type": "navigate", "route_id": "project_qmind"}]


def test_every_model_receives_identical_messages_and_settings(tmp_path: Path, server: ModelServer) -> None:
    run(make_engine(server), load(tmp_path, QUESTION_CASE), tmp_path)
    payloads = [request.payload for request in server.chats]
    assert len(payloads) == 6
    first = payloads[0]
    for payload in payloads:
        assert {key: value for key, value in payload.items() if key != "model"} == {
            key: value for key, value in first.items() if key != "model"
        }
    assert first["think"] is False and first["stream"] is False
    assert first["options"] == {"num_ctx": 4096, "temperature": 0.0, "seed": 42, "top_p": 1.0, "num_predict": 160}


def test_reviewer_guidance_and_author_notes_never_reach_a_model(tmp_path: Path, server: ModelServer) -> None:
    run(make_engine(server), load(tmp_path, QUESTION_CASE), tmp_path)
    sent = json.dumps([request.payload for request in server.requests])
    assert QUESTION_CASE["expected_behavior"] not in sent
    assert QUESTION_CASE["notes"] not in sent


def test_records_keep_the_prompt_the_raw_and_the_validated_answer_and_the_hashes(
    tmp_path: Path, knowledge: KnowledgeBlock
) -> None:
    raw = "**Rastin** studies at Queen's University. See https://example.com/profile"
    server = ModelServer(answer_with(raw))
    engine = make_engine(server)
    result = run(engine, load(tmp_path, QUESTION_CASE), tmp_path, models=["granite4:micro-h"])
    manifest, (record,), (row,), _ = outputs(result)
    response = record["response"]
    assert response["output"]["content"] == raw  # raw, unchanged
    assert response["answer_text"] == "Rastin studies at Queen's University. See (link removed)"
    assert response["output_issues"] == ["link_removed", "formatting_removed"]
    assert record["model"] == {"name": "Granite 4.0 H Micro", "tag": "granite4:micro-h"}
    assert response["configs"] == dict(engine.configs)
    prompt = response["prompt"]
    assert prompt["user_message"] == QUESTION_CASE["message"]
    assert prompt["knowledge_sha256"] == knowledge.sha256
    assert "system_message" not in prompt  # kept once, in the manifest
    # The exact prompt is recoverable: the manifest's system message is what was sent.
    (sent,) = server.chats
    system_message = manifest["system_messages"][prompt["system_message_sha256"]]
    assert sent.payload["messages"] == [
        {"role": "system", "content": system_message},
        {"role": "user", "content": QUESTION_CASE["message"]},
    ]
    assert sha256_hex(system_message) == prompt["system_message_sha256"]
    assert (row["raw_output"], row["answer_text"]) == (raw, response["answer_text"])
    assert row["knowledge_sha256"] == knowledge.sha256
    assert row["models_sha256"] == engine.configs["models"]
    assert row["system_instruction_sha256"] == engine.configs["system_instruction"]
    assert row["case_sha256"] == record["case"]["sha256"] == load_cases(
        tmp_path / "cases.jsonl", route_ids=ROUTE_IDS
    ).cases[0].sha256


# ------------------------------------------------------------- measurements


def test_reported_measurements_pass_through_unchanged(tmp_path: Path) -> None:
    metrics = {
        "total_duration": 2_500_000_000,
        "load_duration": 900_000_000,
        "prompt_eval_count": 1_800,
        "prompt_eval_duration": 1_000_000_000,
        "eval_count": 40,
        "eval_duration": 1_600_000_000,
    }
    server = ModelServer(answer_with(ANSWER, **metrics))
    engine = make_engine(server, client_clock=Ticker(0.25))
    runner = runner_for(engine, tmp_path, clock=Ticker(2.0))
    plan = plan_evaluation(engine, load(tmp_path, QUESTION_CASE), models=["qwen3.5:2b-q4_K_M"])
    result = runner.run(plan, run_id=RUN_ID)
    _, (record,), (row,), _ = outputs(result)
    measured = record["measurements"]
    assert measured == {
        "pipeline_ms": 2000.0,  # the runner's synthetic clock
        "elapsed_ms": 250.0,  # the Ollama client's synthetic clock
        "time_to_first_token_ms": None,
        "load_duration_ns": 900_000_000,
        "prompt_eval_duration_ns": 1_000_000_000,
        "eval_duration_ns": 1_600_000_000,
        "total_duration_ns": 2_500_000_000,
        "prompt_tokens": 1_800,
        "output_tokens": 40,
        "tokens_per_second": 25.0,  # 40 tokens in 1.6 s
        "answer_characters": len(ANSWER),
        "word_count": len(ANSWER.split()),
        "answer_lines": 2,
    }
    assert record["response"]["output"]["metrics"] == metrics
    assert record["response"]["output"]["elapsed_ms"] == 250.0
    assert {name: row[name] for name in MEASUREMENT_FIELDS} == {
        name: csv_cell(value) for name, value in measured.items()
    }
    assert row["time_to_first_token_ms"] == ""


def test_unreported_measurements_are_null_never_estimated(tmp_path: Path) -> None:
    body = chat_body(ANSWER)
    for name in ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration"):
        del body[name]
    server = ModelServer(lambda model, message: body)
    result = run(make_engine(server), load(tmp_path, QUESTION_CASE), tmp_path)
    _, records, rows, _ = outputs(result)
    for record, row in zip(records, rows, strict=True):
        measured = record["measurements"]
        for name in (*OLLAMA_FIELDS, "tokens_per_second", "time_to_first_token_ms"):
            assert measured[name] is None, name
            assert row[name] == "", name
        assert isinstance(measured["elapsed_ms"], float) and measured["elapsed_ms"] >= 0
        assert isinstance(measured["pipeline_ms"], float) and measured["pipeline_ms"] >= 0
        assert set(record["response"]["output"]["metrics"].values()) == {None}


@pytest.mark.parametrize(
    ("output_tokens", "eval_duration_ns", "expected"),
    [
        (40, 1_600_000_000, 25.0),
        (1, 3_000_000_000, 0.333),
        (0, 1_000_000_000, 0.0),
        (40, 0, None),
        (None, 1_000_000_000, None),
        (40, None, None),
        (None, None, None),
    ],
)
def test_tokens_per_second_is_computed_only_from_reported_values(
    output_tokens: int | None, eval_duration_ns: int | None, expected: float | None
) -> None:
    assert tokens_per_second(output_tokens, eval_duration_ns) == expected


def test_application_replies_carry_no_model_measurements(tmp_path: Path, server: ModelServer) -> None:
    result = run(make_engine(server), load(tmp_path, NAVIGATION_CASE, AMBIGUOUS_CASE), tmp_path)
    _, records, rows, _ = outputs(result)
    assert server.requests == []  # no case needed a model, so Ollama was not even asked
    for record, row in zip(records, rows, strict=True):
        measured = record["measurements"]
        assert record["request_sent"] is False
        assert measured["pipeline_ms"] is not None
        for name in (name for name in MEASUREMENT_FIELDS if name != "pipeline_ms"):
            assert measured[name] is None, name
            assert row[name] == "", name


def test_text_measurements_describe_the_displayed_model_answer(tmp_path: Path) -> None:
    answers = {
        "granite4:micro-h": "Rastin studies at Queen's University.",
        "qwen3.5:2b-q4_K_M": " ".join(["Rastin is a third-year Computing student at Queen's University."] * 4),
    }
    server = ModelServer(lambda model, message: chat_body(answers[model], model=model))
    engine = make_engine(server)
    result = run(engine, load(tmp_path, QUESTION_CASE), tmp_path, models=list(answers))
    _, records, _, _ = outputs(result)
    chat = engine.chat_config
    for record in records:
        text = record["response"]["answer_text"]
        measured = record["measurements"]
        assert measured["answer_characters"] == len(text)
        assert measured["word_count"] == len(text.split())
        assert measured["answer_lines"] == wrapped_line_count(text, chat.cloud_line_characters)
        exceeds = record["response"]["continuation"] is not None
        assert exceeds == (measured["answer_lines"] > chat.cloud_max_lines)
    assert [record["response"]["continuation"] is not None for record in records] == [False, True]


def test_answer_lines_agree_with_the_cloud_preview() -> None:
    chat = load_chat_config()
    words = ["a", "Rastin", "Queen's", "interpretability", "x" * 30, "test-time", "2028.", "QMIND,"]
    generator = random.Random(20261106)
    for _ in range(500):
        text = " ".join(generator.choice(words) for _ in range(generator.randint(1, 40)))
        _, cut = cloud_preview(text, max_lines=chat.cloud_max_lines, line_characters=chat.cloud_line_characters)
        assert (wrapped_line_count(text, chat.cloud_line_characters) > chat.cloud_max_lines) == cut, text


# ------------------------------------------------------------------ errors


def test_failed_and_withheld_responses_are_recorded_and_the_run_continues(
    tmp_path: Path, models_config: ModelsConfig
) -> None:
    leak = load_system_instruction().template.split("<approved_facts>")[0]
    tags = models_config.tags
    outcomes: dict[str, Any] = {
        tags[0]: OllamaError(OllamaErrorCode.TIMEOUT, "no complete response within 120 s"),
        tags[1]: HttpResponse(500, b'{"error": "synthetic server failure"}'),
        tags[2]: HttpResponse(200, b"not json"),
        tags[3]: chat_body("", model=tags[3]),
        tags[4]: chat_body(leak, model=tags[4]),
        tags[5]: chat_body(ANSWER, model=tags[5]),
    }
    server = ModelServer(lambda model, message: outcomes[model])
    result = run(make_engine(server, client_clock=Ticker(0.5)), load(tmp_path, QUESTION_CASE), tmp_path)
    assert result.status is RunStatus.COMPLETED
    assert result.status_counts == {"answered": 1, "withheld": 1, "error": 4}
    _, records, rows, _ = outputs(result)
    summary = [(record["status"], (record["response"]["error"] or {}).get("code")) for record in records]
    assert summary == [
        ("error", "timeout"),
        ("error", "ollama_error"),
        ("error", "invalid_response"),
        ("error", "empty_output"),
        ("withheld", None),
        ("answered", None),
    ]
    for record in records:  # a request was sent each time, and its duration measured
        assert record["request_sent"] is True
        assert record["measurements"]["elapsed_ms"] == 500.0
    assert rows[1]["http_status"] == "500"
    assert "synthetic server failure" in rows[1]["error_message"]
    assert records[4]["response"]["output"]["content"] == leak  # the raw leak is kept for review
    assert records[4]["response"]["answer_text"] == load_chat_config().replies.withheld
    for record in records[:5]:  # no model answer is displayed, so there is no answer text to measure
        assert record["measurements"]["word_count"] is None


def test_an_unreachable_ollama_aborts_the_run(tmp_path: Path, models_config: ModelsConfig) -> None:
    unreachable = models_config.tags[1]

    def answer(model: str, message: str) -> Any:
        if model == unreachable:
            return OllamaError(OllamaErrorCode.UNAVAILABLE, "cannot reach Ollama (synthetic)")
        return chat_body(ANSWER, model=model)

    server = ModelServer(answer)
    result = run(make_engine(server), load(tmp_path, NAVIGATION_CASE, QUESTION_CASE), tmp_path)
    assert result.status is RunStatus.ABORTED
    assert result.stop_reason is not None and "Ollama could not be reached" in result.stop_reason
    assert result.records_written == 4  # two records for the first model, then the failed request
    manifest, records, rows, review = outputs(result)
    assert manifest["status"] == "aborted" and manifest["stop_reason"] == result.stop_reason
    assert manifest["records_written"] == len(records) == len(rows) == len(review) == 4
    assert records[-1]["response"]["error"]["code"] == "ollama_unavailable"
    assert len(server.chats) == 2


def test_a_knowledge_block_that_changes_during_the_run_aborts_it(tmp_path: Path, knowledge: KnowledgeBlock) -> None:
    changed = KnowledgeBlock(text=knowledge.text + "\n- **Z99.** A synthetic fact added during the run.")
    calls: list[int] = []

    def loader() -> KnowledgeBlock:
        calls.append(1)
        return knowledge if len(calls) <= 3 else changed  # engine check, run start, first request

    server = ModelServer()
    result = run(make_engine(server, knowledge_loader=loader), load(tmp_path, QUESTION_CASE), tmp_path)
    assert result.status is RunStatus.ABORTED
    assert result.stop_reason is not None and "knowledge block changed during the run" in result.stop_reason
    manifest, records, _, _ = outputs(result)
    assert [record["response"]["prompt"]["knowledge_sha256"] for record in records] == [
        knowledge.sha256,
        changed.sha256,
    ]
    assert manifest["knowledge"]["sha256"] == knowledge.sha256
    assert len(manifest["system_messages"]) == 2  # both exact prompts are kept


def test_unavailable_knowledge_aborts_the_run(tmp_path: Path, knowledge: KnowledgeBlock) -> None:
    calls: list[int] = []

    def loader() -> KnowledgeBlock:
        calls.append(1)
        if len(calls) > 2:
            raise KnowledgeError("knowledge/facts.md: missing marker (synthetic)")
        return knowledge

    server = ModelServer()
    result = run(make_engine(server, knowledge_loader=loader), load(tmp_path, QUESTION_CASE), tmp_path)
    assert result.status is RunStatus.ABORTED
    assert result.stop_reason is not None and "could not be loaded" in result.stop_reason
    _, (record,), _, _ = outputs(result)
    assert record["response"]["error"]["code"] == "knowledge_unavailable"
    assert record["request_sent"] is False
    assert server.chats == []


def test_an_interrupted_run_keeps_its_records_and_finishes_its_files(tmp_path: Path, models_config: ModelsConfig) -> None:
    interrupted = models_config.tags[2]

    def answer(model: str, message: str) -> Any:
        return KeyboardInterrupt() if model == interrupted else chat_body(ANSWER, model=model)

    result = run(make_engine(ModelServer(answer)), load(tmp_path, QUESTION_CASE), tmp_path)
    assert result.status is RunStatus.INTERRUPTED
    assert result.records_written == 2
    manifest, records, rows, review = outputs(result)
    assert manifest["status"] == "interrupted"
    assert manifest["finished_at"] == "2026-11-06T12:00:00.000Z"
    assert len(records) == len(rows) == len(review) == manifest["records_written"] == 2
    assert not (result.directory / "manifest.json.tmp").exists()


def test_an_unexpected_error_marks_the_run_failed_and_propagates(tmp_path: Path) -> None:
    server = ModelServer(lambda model, message: RuntimeError("synthetic failure"))
    engine = make_engine(server)
    runner = runner_for(engine, tmp_path)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        runner.run(plan_evaluation(engine, load(tmp_path, QUESTION_CASE)), run_id=RUN_ID)
    manifest = json.loads((tmp_path / "results" / RUN_ID / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "synthetic failure" in manifest["stop_reason"]
    assert manifest["records_written"] == 0


def test_missing_models_stop_the_run_before_anything_is_written(tmp_path: Path) -> None:
    server = ModelServer(missing={"phi4-mini:3.8b-q4_K_M", "granite4.2:3b-q4_K_M"})
    engine = make_engine(server)
    with pytest.raises(ModelsNotInstalledError) as caught:
        runner_for(engine, tmp_path).run(plan_evaluation(engine, load(tmp_path, QUESTION_CASE)))
    assert caught.value.missing == ("phi4-mini:3.8b-q4_K_M", "granite4.2:3b-q4_K_M")
    assert len(server.shows) == 6 and server.chats == []
    assert not (tmp_path / "results").exists()


def test_an_unreachable_ollama_at_the_start_writes_nothing(tmp_path: Path) -> None:
    server = ModelServer()
    server.show_failure = OllamaError(OllamaErrorCode.UNAVAILABLE, "cannot reach Ollama (synthetic)")
    engine = make_engine(server)
    with pytest.raises(OllamaError) as caught:
        runner_for(engine, tmp_path).run(plan_evaluation(engine, load(tmp_path, QUESTION_CASE)))
    assert caught.value.code is OllamaErrorCode.UNAVAILABLE
    assert not (tmp_path / "results").exists()


# ---------------------------------------------------------------- manifest


def test_the_manifest_describes_the_run(
    tmp_path: Path, knowledge: KnowledgeBlock, models_config: ModelsConfig
) -> None:
    server = ModelServer(capabilities=["completion", "tools"], parameters='stop "<end>"\nrepeat_penalty 1')
    engine = make_engine(server)
    case_set = load(tmp_path, NAVIGATION_CASE, QUESTION_CASE)
    plan = plan_evaluation(engine, case_set, repetitions=2)
    result = runner_for(engine, tmp_path).run(plan, run_id=RUN_ID, note="synthetic note")
    manifest, _, _, _ = outputs(result)
    system_message = PromptBuilder(engine.instruction, knowledge).system_message
    assert manifest["schema_version"] == 1
    assert (manifest["run_id"], manifest["status"], manifest["stop_reason"]) == (RUN_ID, "completed", None)
    assert manifest["note"] == "synthetic note"
    assert manifest["started_at"] == manifest["finished_at"] == "2026-11-06T12:00:00.000Z"
    assert set(manifest["software"]) == {"mascot_lab", "python", "platform"}
    assert manifest["ollama"] == {"base_url": "http://127.0.0.1:11434", "timeout_s": 120.0}
    assert manifest["cases"] == {
        "source": case_set.source,
        "sha256": case_set.sha256,
        "count": 2,
        "selected": list(case_set.ids),
    }
    assert manifest["repetitions"] == 2
    assert manifest["configs"] == dict(engine.configs)
    assert manifest["knowledge"] == {"source": "knowledge/facts.md", "sha256": knowledge.sha256}
    assert manifest["system_messages"] == {sha256_hex(system_message): system_message}
    assert manifest["rubric"] == {
        "source": "eval/rubric.md",
        "sha256": sha256_hex(RUBRIC_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")),
    }
    assert (manifest["planned_records"], manifest["planned_model_requests"]) == (24, 12)
    assert manifest["records_written"] == 24
    assert manifest["status_counts"] == {"answered": 12, "routed": 12}
    assert manifest["route_checks"] == {"match": 24, "mismatch": 0}
    assert manifest["files"] == dict(RUN_FILES)
    assert [model["tag"] for model in manifest["models"]] == list(models_config.tags)
    for model in manifest["models"]:
        assert model["think"] is False
        assert model["options"] == {"num_ctx": 4096, "temperature": 0.0, "seed": 42, "top_p": 1.0, "num_predict": 160}
        assert model["override"] is None
        assert model["ollama_show"] == {
            "capabilities": ["completion", "tools"],
            "modelfile_parameters": 'stop "<end>"\nrepeat_penalty 1',
        }


def test_the_manifest_is_written_when_the_run_starts(tmp_path: Path, server: ModelServer) -> None:
    engine = make_engine(server)
    seen: list[dict[str, Any]] = []
    starts: list[RunStart] = []

    def on_start(start: RunStart) -> None:
        starts.append(start)
        seen.append(json.loads((start.directory / "manifest.json").read_text(encoding="utf-8")))

    progress: list[RecordProgress] = []
    plan = plan_evaluation(engine, load(tmp_path, NAVIGATION_CASE, QUESTION_CASE), models=["granite4:micro-h"])
    runner_for(engine, tmp_path).run(plan, run_id=RUN_ID, on_start=on_start, on_record=progress.append)
    ((start,), (early,)) = (starts, seen)
    expected = (RUN_ID, (tmp_path / "results" / RUN_ID).resolve(), True)
    assert (start.run_id, start.directory, start.models_checked) == expected
    assert (early["status"], early["finished_at"], early["records_written"]) == ("running", None, 0)
    assert [(item.sequence, item.total, item.case_id, item.status) for item in progress] == [
        (1, 2, NAVIGATION_CASE["id"], ChatStatus.ROUTED),
        (2, 2, QUESTION_CASE["id"], ChatStatus.ANSWERED),
    ]
    assert progress[0].elapsed_ms is None and progress[1].elapsed_ms is not None


def test_no_model_metadata_is_requested_when_no_case_needs_a_model(tmp_path: Path, server: ModelServer) -> None:
    result = run(make_engine(server), load(tmp_path, NAVIGATION_CASE), tmp_path)
    manifest, _, _, _ = outputs(result)
    assert server.requests == []
    assert all(model["ollama_show"] is None for model in manifest["models"])
    assert manifest["planned_model_requests"] == 0


# ------------------------------------------------------------- CSV and JSONL


def test_records_csv_has_the_documented_columns_and_matches_the_jsonl(tmp_path: Path, server: ModelServer) -> None:
    result = run(make_engine(server), load(tmp_path, NAVIGATION_CASE, QUESTION_CASE, AMBIGUOUS_CASE), tmp_path)
    header, rows = read_csv(result.directory / "records.csv")
    assert tuple(header) == RECORD_COLUMNS
    records = read_jsonl(result.directory / "records.jsonl")
    assert len(rows) == len(records) == 18
    for record, row in zip(records, rows, strict=True):
        assert row == {column: csv_cell(value) for column, value in record_row(record).items()}
    ambiguous = next(row for row in rows if row["case_id"] == AMBIGUOUS_CASE["id"])
    assert ambiguous["route_candidates"] == "project_qmind; project_ingenuity"
    assert ambiguous["actions"] == "suggest:project_qmind; suggest:project_ingenuity"
    assert ambiguous["request_sent"] == "false"


def test_csv_cells_cannot_start_a_formula_but_jsonl_keeps_the_exact_text(tmp_path: Path) -> None:
    content = "=1+1 is not a fact about Rastin."
    result = run(
        make_engine(ModelServer(answer_with(content))), load(tmp_path, QUESTION_CASE), tmp_path, models=["granite4:micro-h"]
    )
    _, (record,), (row,), (review,) = outputs(result)
    assert record["response"]["output"]["content"] == record["response"]["answer_text"] == content
    assert row["raw_output"] == row["answer_text"] == review["answer_text"] == review["raw_output"] == "'" + content


@pytest.mark.parametrize(
    ("value", "cell"),
    [
        (None, ""),
        (True, "true"),
        (False, "false"),
        (0, "0"),
        (1234.567, "1234.567"),
        ("plain", "plain"),
        ("", ""),
        ("=cmd", "'=cmd"),
        ("+1", "'+1"),
        ("-1", "'-1"),
        ("@SUM(A1)", "'@SUM(A1)"),
        ("\tx", "'\tx"),
        ("\rx", "'\rx"),
        (" =not a formula", " =not a formula"),
        ([], ""),
        (["a", "b"], "a; b"),
        (["=a", "b"], "'=a; b"),
    ],
)
def test_csv_cell_forms(value: Any, cell: str) -> None:
    assert csv_cell(value) == cell


def test_csv_cell_rejects_nested_values() -> None:
    with pytest.raises(TypeError):
        csv_cell({"a": 1})


def test_non_ascii_text_is_kept_readable_and_line_separators_are_escaped(tmp_path: Path) -> None:
    content = "Rastin studies at Queen’s University… next line\x85and more"
    result = run(
        make_engine(ModelServer(answer_with(content))), load(tmp_path, QUESTION_CASE), tmp_path, models=["granite4:micro-h"]
    )
    data = (result.directory / "records.jsonl").read_bytes()
    text = data.decode("utf-8")  # valid UTF-8, no byte-order mark
    assert not data.startswith(b"\xef\xbb\xbf")
    assert "Queen’s University…" in text
    assert " " not in text and "\x85" not in text and " " not in text
    assert len(text.splitlines()) == 1  # still one line per record for any line splitter
    (record,) = read_jsonl(result.directory / "records.jsonl")
    assert record["response"]["output"]["content"] == content
    assert (result.directory / "records.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    assert json.loads(json_line({"text": content}))["text"] == content


# ------------------------------------------------------------------ review


def test_the_review_sheet_is_blind_and_its_grades_are_blank(
    tmp_path: Path, server: ModelServer, models_config: ModelsConfig
) -> None:
    case_set = load(tmp_path, NAVIGATION_CASE, QUESTION_CASE)
    result = run(make_engine(server), case_set, tmp_path, repetitions=2)
    header, review = read_csv(result.directory / "review.csv")
    assert tuple(header) == REVIEW_COLUMNS
    _, rows = read_csv(result.directory / "records.csv")
    assert sorted(row["record_id"] for row in review) == sorted(row["record_id"] for row in rows)
    sheet = (result.directory / "review.csv").read_text(encoding="utf-8-sig")
    for model in models_config.models:
        assert model.tag not in sheet and model.name not in sheet
    assert QUESTION_CASE["notes"] not in sheet  # author notes stay out of the review sheet
    for row in review:
        for dimension in DIMENSIONS:
            assert row[dimension.id] == ""
        assert row["reviewer"] == row["notes"] == ""
    # Grouped by case in file order; within a case, by the opaque record ID.
    assert [row["case_id"] for row in review] == [NAVIGATION_CASE["id"]] * 12 + [QUESTION_CASE["id"]] * 12
    for case_id in case_set.ids:
        ids = [row["record_id"] for row in review if row["case_id"] == case_id]
        assert ids == sorted(ids)
    question = next(row for row in review if row["case_id"] == QUESTION_CASE["id"])
    assert question["expected_behavior"] == QUESTION_CASE["expected_behavior"]
    assert (question["status"], question["route_check"]) == ("answered", "match")
    assert question["answer_text"] == question["raw_output"] == ANSWER


def test_the_review_sheet_order_does_not_follow_the_model_order(tmp_path: Path, server: ModelServer) -> None:
    result = run(make_engine(server), load(tmp_path, QUESTION_CASE), tmp_path)
    _, rows = read_csv(result.directory / "records.csv")
    _, review = read_csv(result.directory / "review.csv")
    model_of = {row["record_id"]: row["model_tag"] for row in rows}
    assert [model_of[row["record_id"]] for row in review] != [row["model_tag"] for row in rows]


def test_record_ids_are_unique_deterministic_and_opaque(tmp_path: Path, server: ModelServer) -> None:
    result = run(make_engine(server), load(tmp_path, NAVIGATION_CASE, QUESTION_CASE), tmp_path, repetitions=3)
    records = read_jsonl(result.directory / "records.jsonl")
    ids = [record["record_id"] for record in records]
    assert len(set(ids)) == len(ids) == 36
    for record in records:
        assert re.fullmatch(r"[0-9a-f]{16}", record["record_id"])
        assert record["record_id"] == record_id(
            RUN_ID, record["model"]["tag"], record["case"]["id"], record["repetition"]
        )


# ------------------------------------------------------------ route checks


def test_route_checks_are_recorded_for_every_record(tmp_path: Path, server: ModelServer) -> None:
    wrong = {
        "id": "synthetic-wrong-expectation",
        "category": "synthetic-navigation",
        "message": "take me to Satira",
        "expected_route": {"status": "routed", "route_id": "project_qmind"},
    }
    unchecked = {"id": "synthetic-unchecked", "category": "synthetic", "message": "show me Satira"}
    engine = make_engine(server)
    case_set = load(tmp_path, NAVIGATION_CASE, wrong, unchecked)
    plan = plan_evaluation(engine, case_set)
    assert [planned.route_check for planned in plan.cases] == [RouteCheck.MATCH, RouteCheck.MISMATCH, None]
    result = runner_for(engine, tmp_path).run(plan, run_id=RUN_ID)
    assert result.route_checks == {"match": 6, "mismatch": 6}
    records = read_jsonl(result.directory / "records.jsonl")
    checks = {record["case"]["id"]: record["route_check"] for record in records}
    assert checks == {NAVIGATION_CASE["id"]: "match", wrong["id"]: "mismatch", unchecked["id"]: None}


# ---------------------------------------------------------------- planning


def test_a_plan_counts_records_and_model_requests(tmp_path: Path, server: ModelServer) -> None:
    engine = make_engine(server)
    case_set = load(tmp_path, NAVIGATION_CASE, QUESTION_CASE, AMBIGUOUS_CASE)
    plan = plan_evaluation(engine, case_set, models=["Qwen3.5 2B", "granite4:micro-h", "qwen3.5:2b-q4_K_M"], repetitions=3)
    assert [model.tag for model in plan.models] == ["granite4:micro-h", "qwen3.5:2b-q4_K_M"]  # configuration order
    assert (plan.total_records, plan.model_requests) == (3 * 2 * 3, 1 * 2 * 3)
    assert [planned.needs_model for planned in plan.cases] == [False, True, False]
    assert [planned.application_status for planned in plan.cases] == [
        ChatStatus.ROUTED,
        None,
        ChatStatus.AMBIGUOUS_DESTINATION,
    ]
    assert plan.cases[0].route is not None and plan.cases[0].route.route_id == "project_qmind"
    assert plan.cases[1].route is not None and plan.cases[1].route.status is RouteStatus.NOT_NAVIGATION
    selected = plan_evaluation(engine, case_set, case_ids=[AMBIGUOUS_CASE["id"], NAVIGATION_CASE["id"]])
    assert [planned.case.id for planned in selected.cases] == [NAVIGATION_CASE["id"], AMBIGUOUS_CASE["id"]]
    assert server.requests == []  # planning never contacts Ollama


@pytest.mark.parametrize("repetitions", [0, -1, True, 1.5, "2"])
def test_a_plan_needs_a_positive_whole_number_of_repetitions(
    tmp_path: Path, server: ModelServer, repetitions: Any
) -> None:
    with pytest.raises(ConfigError, match="repetitions"):
        plan_evaluation(make_engine(server), load(tmp_path, QUESTION_CASE), repetitions=repetitions)


def test_a_plan_rejects_unknown_or_empty_model_selections(tmp_path: Path, server: ModelServer) -> None:
    engine = make_engine(server)
    case_set = load(tmp_path, QUESTION_CASE)
    with pytest.raises(ConfigError, match="not a configured model"):
        plan_evaluation(engine, case_set, models=["llama3:8b"])
    with pytest.raises(ConfigError, match="no models selected"):
        plan_evaluation(engine, case_set, models=[])


def test_invalid_input_cases_are_answered_by_the_application(tmp_path: Path, server: ModelServer) -> None:
    blank = {"id": "synthetic-blank", "category": "synthetic", "message": "   "}
    long = {"id": "synthetic-long", "category": "synthetic", "message": "Rastin? " * 200}
    result = run(make_engine(server), load(tmp_path, blank, long), tmp_path, models=["granite4:micro-h"])
    _, records, _, _ = outputs(result)
    assert [(record["status"], record["response"]["error"]["code"]) for record in records] == [
        ("invalid_input", "empty_input"),
        ("invalid_input", "input_too_long"),
    ]
    assert all(record["response"]["route"] is None and not record["request_sent"] for record in records)
    assert server.requests == []


# ------------------------------------------------------------ directories


def test_a_run_never_reuses_a_directory(tmp_path: Path, server: ModelServer) -> None:
    engine = make_engine(server)
    plan = plan_evaluation(engine, load(tmp_path, NAVIGATION_CASE))
    first = runner_for(engine, tmp_path).run(plan, run_id=RUN_ID)
    before = {path.name: path.read_bytes() for path in first.directory.iterdir()}
    with pytest.raises(ConfigError, match="already exists"):
        runner_for(engine, tmp_path).run(plan, run_id=RUN_ID)
    assert {path.name: path.read_bytes() for path in first.directory.iterdir()} == before


@pytest.mark.parametrize("bad", ["", "a/b", "..", "a\\b", ".hidden", "trailing.", "x" * 101, "spaced id"])
def test_an_unsafe_run_id_is_refused_before_anything_happens(tmp_path: Path, server: ModelServer, bad: str) -> None:
    engine = make_engine(server)
    with pytest.raises(ConfigError, match="run ID"):
        runner_for(engine, tmp_path).run(plan_evaluation(engine, load(tmp_path, QUESTION_CASE)), run_id=bad)
    assert server.requests == []
    assert not (tmp_path / "results").exists()


def test_a_blank_note_is_refused(tmp_path: Path, server: ModelServer) -> None:
    engine = make_engine(server)
    with pytest.raises(ConfigError, match="note"):
        runner_for(engine, tmp_path).run(plan_evaluation(engine, load(tmp_path, QUESTION_CASE)), note="  ")
    assert server.requests == []


def test_new_run_ids_name_the_utc_start_and_differ() -> None:
    first, second = new_run_id(FIXED_NOW), new_run_id(FIXED_NOW)
    assert re.fullmatch(r"20261106T120000Z-[0-9a-f]{6}", first)
    assert first != second


def test_results_inside_the_repository_must_be_under_results(tmp_path: Path) -> None:
    assert check_results_dir() == RESULTS_DIR.resolve()
    assert check_results_dir(RESULTS_DIR / "smoke") == (RESULTS_DIR / "smoke").resolve()
    assert check_results_dir(tmp_path) == tmp_path.resolve()
    for inside in (PROJECT_ROOT, PROJECT_ROOT / "docs", PROJECT_ROOT / "eval" / "results", RESULTS_DIR / ".." / "src"):
        with pytest.raises(ConfigError, match="only results/ is ignored by Git"):
            check_results_dir(inside)
    assert not (PROJECT_ROOT / "docs" / "results").exists()


def test_the_results_directory_is_ignored_by_git() -> None:
    ignored = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8-sig").splitlines()
    assert "/results/" in [line.strip() for line in ignored]
