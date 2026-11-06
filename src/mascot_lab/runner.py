"""Evaluation runner: the same cases, through the same engine, for each model.

``plan_evaluation`` selects the cases, models, and repetitions, and runs
steps 1 and 2 of the chat engine (input validation and deterministic
routing) for each case. Neither step depends on the model, so a plan shows,
before anything is sent, which cases the application answers without a
model and whether each case's ``expected_route`` holds.

``EvaluationRunner.run`` then:

1. asks Ollama (``/api/show``, which runs no model) whether every selected
   model is installed, and keeps each model's reported capabilities and
   Modelfile parameters for the manifest. This is skipped when no case
   needs a model;
2. creates a new run directory and writes its manifest
   (``mascot_lab.recorder``);
3. sends every case through ``ChatEngine.respond``, exactly as a visitor
   message would go, for each model in configuration order, each
   repetition, and each case in file order. A model's requests run back to
   back, so Ollama loads each model once per run rather than once per case;
4. records each response as soon as it arrives;
5. writes the blind review sheet (``mascot_lab.review``) and the final
   manifest.

Requests are sequential and never retried, and nothing is sent other than
what ``ChatEngine`` sends. A response that is an error, such as a timeout,
is recorded and the run continues. The run stops early, and its manifest
says why, when continuing could not produce comparable records: Ollama
cannot be reached, or the knowledge block becomes unavailable or changes
during the run, which would give later models different facts. An
interrupted run (Ctrl+C) keeps every record written before it stopped.

The runner measures and records. It does not grade, score, rank, or select
models.
"""

import platform
import re
import secrets
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from importlib import metadata
from pathlib import Path
from typing import Any

from mascot_lab._textfile import read_text_file, sha256_hex
from mascot_lab.cases import CaseSet, EvaluationCase
from mascot_lab.config import ModelConfig
from mascot_lab.engine import ChatEngine, ChatResponse, ChatStatus, ErrorCode
from mascot_lab.errors import ConfigError, MascotLabError
from mascot_lab.knowledge import KnowledgeBlock
from mascot_lab.measurements import measure
from mascot_lab.ollama import ModelDetails, OllamaError, OllamaErrorCode
from mascot_lab.paths import display_path
from mascot_lab.prompt import PromptBuilder
from mascot_lab.recorder import RUN_FILES, SCHEMA_VERSION, RunRecorder, check_results_dir
from mascot_lab.review import RUBRIC_PATH, order_review_rows, review_row
from mascot_lab.router import RouteResult

# A run ID names a directory: no path separators, and it must not end with a
# dot, which Windows drops from directory names.
_RUN_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,98}[A-Za-z0-9])?")
RECORD_ID_LENGTH = 16

RUN_ORDER = "each model in configuration order; for each model, each repetition; for each, each case in file order"


class RunStatus(StrEnum):
    """The state of a run, as its manifest records it."""

    RUNNING = "running"  # started; the manifest is replaced when the run ends
    COMPLETED = "completed"  # every planned record was written
    INTERRUPTED = "interrupted"  # stopped by the operator (Ctrl+C)
    ABORTED = "aborted"  # stopped because later records could not be comparable; see stop_reason
    FAILED = "failed"  # stopped by an unexpected error; see stop_reason


class RouteCheck(StrEnum):
    """Whether deterministic routing met a case's ``expected_route``."""

    MATCH = "match"
    MISMATCH = "mismatch"


@dataclass(frozen=True, slots=True)
class PlannedCase:
    """One selected case, with what steps 1 and 2 do with its message."""

    case: EvaluationCase
    route: RouteResult | None  # None when input validation refused the message
    application_status: ChatStatus | None  # the application's reply; None when a model answers
    route_check: RouteCheck | None  # None when the case sets no expected_route

    @property
    def needs_model(self) -> bool:
        return self.application_status is None


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    """What a run will do: every selected model gets every selected case."""

    case_set: CaseSet
    cases: tuple[PlannedCase, ...]
    models: tuple[ModelConfig, ...]
    repetitions: int

    @property
    def total_records(self) -> int:
        return len(self.cases) * len(self.models) * self.repetitions

    @property
    def model_requests(self) -> int:
        """Chat requests the run will send to Ollama."""
        return sum(planned.needs_model for planned in self.cases) * len(self.models) * self.repetitions


@dataclass(frozen=True, slots=True)
class RunStart:
    """Reported when a run has created its directory and is about to send requests."""

    run_id: str
    directory: Path
    models_checked: bool  # whether /api/show was asked about every model


@dataclass(frozen=True, slots=True)
class RecordProgress:
    """Reported after each record is written."""

    sequence: int
    total: int
    model_tag: str
    case_id: str
    repetition: int
    status: ChatStatus
    error_code: str | None
    elapsed_ms: float | None  # the Ollama request time, when a request was sent


@dataclass(frozen=True, slots=True)
class RunResult:
    """How a run ended. Counts are of records, not grades."""

    run_id: str
    directory: Path
    status: RunStatus
    stop_reason: str | None
    planned_records: int
    records_written: int
    status_counts: Mapping[str, int]  # records by response status
    route_checks: Mapping[str, int]  # records by route check result


class ModelsNotInstalledError(MascotLabError):
    """Ollama does not have every selected model, so the run did not start."""

    def __init__(self, missing: Sequence[str]) -> None:
        super().__init__(f"not installed in Ollama: {', '.join(missing)}")
        self.missing = tuple(missing)


def plan_evaluation(
    engine: ChatEngine,
    case_set: CaseSet,
    *,
    case_ids: Iterable[str] | None = None,
    models: Iterable[str | ModelConfig] | None = None,
    repetitions: int = 1,
) -> EvaluationPlan:
    """Select cases (file order), models (configuration order), and repetitions.

    ``None`` selects every case or every configured model. Runs only input
    validation and routing; nothing is loaded from the knowledge file and
    nothing is sent to Ollama.
    """
    if type(repetitions) is not int or repetitions < 1:
        raise ConfigError(f"repetitions must be a positive integer, found {repetitions!r}")
    planned: list[PlannedCase] = []
    for case in case_set.select(case_ids):
        routing = engine.route(case.message)
        check = None
        if case.expected_route is not None:
            check = RouteCheck.MATCH if case.expected_route.matches(routing.route) else RouteCheck.MISMATCH
        status = None if routing.response is None else routing.response.status
        planned.append(PlannedCase(case=case, route=routing.route, application_status=status, route_check=check))
    return EvaluationPlan(
        case_set=case_set,
        cases=tuple(planned),
        models=_select_models(engine, models),
        repetitions=repetitions,
    )


def check_models(engine: ChatEngine, models: Sequence[ModelConfig]) -> dict[str, ModelDetails]:
    """Return Ollama's metadata for every model (``/api/show``; no model is run).

    Raises ``ModelsNotInstalledError`` naming every missing model, and
    ``OllamaError`` for any other failure.
    """
    details: dict[str, ModelDetails] = {}
    missing: list[str] = []
    for model in models:
        try:
            details[model.tag] = engine.client.show(model.tag)
        except OllamaError as exc:
            if exc.code is not OllamaErrorCode.MODEL_NOT_FOUND:
                raise
            missing.append(model.tag)
    if missing:
        raise ModelsNotInstalledError(missing)
    return details


class EvaluationRunner:
    """Runs evaluation plans through one chat engine and records the results."""

    def __init__(
        self,
        engine: ChatEngine,
        *,
        results_dir: str | Path | None = None,
        clock: Callable[[], float] = time.perf_counter,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._engine = engine
        self._results_dir = check_results_dir(results_dir)
        self._clock = clock
        self._now = now if now is not None else _utc_now

    @property
    def results_dir(self) -> Path:
        return self._results_dir

    def run(
        self,
        plan: EvaluationPlan,
        *,
        run_id: str | None = None,
        note: str | None = None,
        on_start: Callable[[RunStart], None] | None = None,
        on_record: Callable[[RecordProgress], None] | None = None,
    ) -> RunResult:
        """Run ``plan`` and record every response in a new run directory.

        Nothing is written when the selected models are not all installed
        (``ModelsNotInstalledError``) or Ollama cannot be asked
        (``OllamaError``).
        """
        if note is not None and (not isinstance(note, str) or not note.strip()):
            raise ConfigError("a run note must be a non-empty string")
        if run_id is not None:
            _check_run_id(run_id)
        details = check_models(self._engine, plan.models) if plan.model_requests else None
        knowledge = self._engine.load_knowledge()
        system_message = PromptBuilder(self._engine.instruction, knowledge).system_message
        started = self._now()
        run_id = new_run_id(started) if run_id is None else run_id
        manifest = self._manifest(plan, run_id, started, note, details, knowledge, system_message)
        system_messages = manifest["system_messages"]
        recorder = RunRecorder(self._results_dir / run_id)

        status_counts: Counter[str] = Counter()
        route_checks: Counter[str] = Counter()
        rows: list[dict[str, Any]] = []
        status, reason = RunStatus.COMPLETED, None
        try:
            recorder.write_manifest(manifest)
            if on_start is not None:
                on_start(RunStart(run_id, recorder.directory, models_checked=details is not None))
            for sequence, (model, repetition, planned) in enumerate(_schedule(plan), start=1):
                record, response = self._record(run_id, sequence, model, repetition, planned.case)
                row = review_row(record)
                recorder.write_record(record)
                status_counts[response.status.value] += 1
                rows.append(row)
                if record["route_check"] is not None:
                    route_checks[record["route_check"]] += 1
                if response.prompt is not None:
                    system_messages.setdefault(response.prompt.system_message_sha256, response.prompt.system_message)
                if on_record is not None:
                    on_record(_progress(record, response, plan.total_records))
                reason = _stop_reason(response, knowledge.sha256)
                if reason is not None:
                    status = RunStatus.ABORTED
                    break
        except KeyboardInterrupt:
            status, reason = RunStatus.INTERRUPTED, "interrupted by the operator"
        except BaseException as exc:
            status, reason = RunStatus.FAILED, f"unexpected error: {exc!r}"
            raise
        finally:
            manifest.update(
                status=status.value,
                stop_reason=reason,
                records_written=sum(status_counts.values()),
                status_counts={s.value: status_counts[s.value] for s in ChatStatus if status_counts[s.value]},
                route_checks={check.value: route_checks[check.value] for check in RouteCheck},
            )
            self._finish(recorder, manifest, rows, [planned.case.id for planned in plan.cases])
        return RunResult(
            run_id=run_id,
            directory=recorder.directory,
            status=status,
            stop_reason=reason,
            planned_records=plan.total_records,
            records_written=manifest["records_written"],
            status_counts=dict(manifest["status_counts"]),
            route_checks=dict(manifest["route_checks"]),
        )

    def _record(
        self, run_id: str, sequence: int, model: ModelConfig, repetition: int, case: EvaluationCase
    ) -> tuple[dict[str, Any], ChatResponse]:
        started_at = _timestamp(self._now())
        start = self._clock()
        response = self._engine.respond(case.message, model)
        pipeline_ms = round((self._clock() - start) * 1000, 3)
        route_check = None
        if case.expected_route is not None:
            matched = case.expected_route.matches(response.route)
            route_check = (RouteCheck.MATCH if matched else RouteCheck.MISMATCH).value
        record = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "record_id": record_id(run_id, model.tag, case.id, repetition),
            "sequence": sequence,
            "started_at": started_at,
            "repetition": repetition,
            "case": {**case.to_dict(), "sha256": case.sha256},
            "model": {"name": model.name, "tag": model.tag},
            "status": response.status.value,
            "request_sent": response.request is not None,
            "route_check": route_check,
            "measurements": measure(
                response,
                pipeline_ms=pipeline_ms,
                line_characters=self._engine.chat_config.cloud_line_characters,
            ),
            "response": response.to_dict(),
        }
        return record, response

    def _manifest(
        self,
        plan: EvaluationPlan,
        run_id: str,
        started: datetime,
        note: str | None,
        details: Mapping[str, ModelDetails] | None,
        knowledge: KnowledgeBlock,
        system_message: str,
    ) -> dict[str, Any]:
        client = self._engine.client
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "status": RunStatus.RUNNING.value,
            "stop_reason": None,
            "note": note,
            "started_at": _timestamp(started),
            "finished_at": None,
            "software": _software(),
            "ollama": {"base_url": client.base_url, "timeout_s": client.timeout_s},
            "order": RUN_ORDER,
            "cases": {
                "source": plan.case_set.source,
                "sha256": plan.case_set.sha256,
                "count": len(plan.case_set.cases),
                "selected": [planned.case.id for planned in plan.cases],
            },
            "repetitions": plan.repetitions,
            "models": [_model_entry(model, details) for model in plan.models],
            "configs": dict(self._engine.configs),
            "knowledge": {"source": knowledge.source, "sha256": knowledge.sha256},
            # The exact system message of every request, keyed by its SHA-256.
            "system_messages": {sha256_hex(system_message): system_message},
            "rubric": _rubric(),
            "planned_records": plan.total_records,
            "planned_model_requests": plan.model_requests,
            "records_written": 0,
            "status_counts": {},
            "route_checks": {},
            "files": dict(RUN_FILES),
        }

    def _finish(
        self,
        recorder: RunRecorder,
        manifest: dict[str, Any],
        rows: Sequence[Mapping[str, Any]],
        case_ids: Sequence[str],
    ) -> None:
        try:
            recorder.write_review_sheet(order_review_rows(rows, case_ids))
        finally:
            manifest["finished_at"] = _timestamp(self._now())
            try:
                recorder.write_manifest(manifest)
            finally:
                recorder.close()


def record_id(run_id: str, model_tag: str, case_id: str, repetition: int) -> str:
    """A deterministic record ID that does not reveal the model to a reviewer."""
    return sha256_hex("\n".join((run_id, model_tag, case_id, str(repetition))))[:RECORD_ID_LENGTH]


def new_run_id(started: datetime) -> str:
    """A run ID such as ``20261106T150000Z-3fa91c``: the UTC start time and a random suffix."""
    return f"{started.astimezone(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"


def _check_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ConfigError(
            f"run ID {run_id!r} must be letters, digits, '.', '_', or '-' (at most 100), "
            "starting and ending with a letter or digit"
        )


def _select_models(engine: ChatEngine, models: Iterable[str | ModelConfig] | None) -> tuple[ModelConfig, ...]:
    if models is None:
        return engine.models.models
    wanted = {engine.model(key).tag for key in models}
    if not wanted:
        raise ConfigError("no models selected")
    return tuple(model for model in engine.models.models if model.tag in wanted)


def _schedule(plan: EvaluationPlan) -> Iterator[tuple[ModelConfig, int, PlannedCase]]:
    for model in plan.models:
        for repetition in range(1, plan.repetitions + 1):
            for planned in plan.cases:
                yield model, repetition, planned


def _stop_reason(response: ChatResponse, knowledge_sha256: str) -> str | None:
    """Why the run cannot go on after ``response``, or None."""
    error = response.error
    if error is not None and error.code is ErrorCode.OLLAMA_UNAVAILABLE:
        return f"Ollama could not be reached, so no later response could be recorded: {error.message}"
    if error is not None and error.code is ErrorCode.KNOWLEDGE_UNAVAILABLE:
        return f"the knowledge block could not be loaded, so later records would not be comparable: {error.message}"
    if response.prompt is not None and response.prompt.knowledge_sha256 != knowledge_sha256:
        return (
            f"the knowledge block changed during the run (sha256 {knowledge_sha256} at the start, "
            f"{response.prompt.knowledge_sha256} now), so later records would not be comparable"
        )
    return None


def _progress(record: Mapping[str, Any], response: ChatResponse, total: int) -> RecordProgress:
    return RecordProgress(
        sequence=record["sequence"],
        total=total,
        model_tag=record["model"]["tag"],
        case_id=record["case"]["id"],
        repetition=record["repetition"],
        status=response.status,
        error_code=None if response.error is None else response.error.code.value,
        elapsed_ms=record["measurements"]["elapsed_ms"],
    )


def _model_entry(model: ModelConfig, details: Mapping[str, ModelDetails] | None) -> dict[str, Any]:
    override = None
    if model.override is not None:
        override = {"reason": model.override.reason, "options": _options(model.override.options)}
    ollama = None
    if details is not None:
        reported = details[model.tag]
        ollama = {
            "capabilities": None if reported.capabilities is None else list(reported.capabilities),
            "modelfile_parameters": reported.parameters,
        }
    return {
        "name": model.name,
        "tag": model.tag,
        "think": model.think,
        "options": _options(model.options),
        "override": override,
        "ollama_show": ollama,  # None when no case needed a model, so Ollama was not asked
    }


def _options(options: Mapping[str, Any]) -> dict[str, Any]:
    return {name: list(value) if isinstance(value, tuple) else value for name, value in options.items()}


def _software() -> dict[str, str | None]:
    try:
        version: str | None = metadata.version("mascot-lab")
    except metadata.PackageNotFoundError:
        version = None
    return {"mascot_lab": version, "python": platform.python_version(), "platform": platform.platform()}


def _rubric() -> dict[str, str | None]:
    """The rubric file in effect, by SHA-256; None when it cannot be read."""
    try:
        text = read_text_file(RUBRIC_PATH, error=ConfigError, label="review rubric")
    except ConfigError:
        return {"source": display_path(RUBRIC_PATH), "sha256": None}
    return {"source": display_path(RUBRIC_PATH), "sha256": sha256_hex(text)}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
