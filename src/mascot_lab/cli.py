"""Command line for the shared chat engine and the evaluation runner.

    python -m mascot_lab validate [--ollama]
    python -m mascot_lab route MESSAGE
    python -m mascot_lab chat --model TAG_OR_NAME [--dry-run] [--show-prompt] MESSAGE
    python -m mascot_lab evaluate [--cases FILE] [--model TAG_OR_NAME]... [--case ID]...
                                  [--repeat N] [--results-dir DIR] [--note TEXT] [--dry-run]

``validate`` checks every configuration file and the knowledge block without
contacting Ollama. With ``--ollama`` it also reads each configured model's
metadata from the local server (``/api/show``), which runs no model.
``route`` shows how a message is handled before any model is called.
``chat`` sends one message to one configured model through the full
pipeline; with ``--dry-run`` it prints the exact request instead of sending
it. ``route`` and ``chat`` print JSON. ``evaluate`` runs the evaluation cases
against the configured models and records the results under ``results/``;
with ``--dry-run`` it prints the plan and contacts nothing.

Exit status: 0 on success, 1 when a request fails, a configured model is not
installed, or an evaluation run stops before writing every planned record,
and 2 for invalid configuration, cases, or arguments.
"""

import argparse
import copy
import json
import sys
from collections import Counter
from collections.abc import Sequence
from typing import Any

from mascot_lab.cases import load_cases
from mascot_lab.config import ModelsConfig, OptionValue
from mascot_lab.engine import ChatEngine, ChatResponse, ChatStatus, route_to_dict
from mascot_lab.errors import ConfigError, MascotLabError
from mascot_lab.ollama import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT_S,
    OllamaClient,
    OllamaError,
    OllamaErrorCode,
    Transport,
)
from mascot_lab.paths import display_path
from mascot_lab.prompt import PromptBuilder
from mascot_lab.recorder import RUN_FILES, check_results_dir
from mascot_lab.review import RUBRIC_PATH
from mascot_lab.runner import (
    EvaluationPlan,
    EvaluationRunner,
    ModelsNotInstalledError,
    PlannedCase,
    RecordProgress,
    RouteCheck,
    RunResult,
    RunStart,
    RunStatus,
    plan_evaluation,
)

PROGRAM = "python -m mascot_lab"


def main(argv: Sequence[str] | None = None, *, transport: Transport | None = None) -> int:
    """Run the command line. ``transport`` replaces HTTP in tests."""
    args = _parser().parse_args(argv)
    _tolerate_unencodable_output()
    try:
        return args.handler(args, transport)
    except MascotLabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Shared chat engine for the candidate mascot models.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    validate = commands.add_parser(
        "validate",
        help="check every configuration file",
        description="Check the configuration files, the shared instruction, and the knowledge "
        "block. Ollama is not contacted unless --ollama is given.",
    )
    validate.add_argument(
        "--ollama",
        action="store_true",
        help="also read each configured model's metadata from the local Ollama server "
        "(installed or not, capabilities, Modelfile parameters); no model is run",
    )
    _add_ollama_arguments(validate)
    validate.set_defaults(handler=_validate)

    route = commands.add_parser(
        "route",
        help="show how a message is handled before any model is called",
        description="Run input validation and deterministic routing only. Nothing is sent to "
        "Ollama.",
    )
    route.add_argument("message")
    route.set_defaults(handler=_route)

    chat = commands.add_parser(
        "chat",
        help="send one message to one configured model",
        description="Handle one message with one configured model: input validation, routing, "
        "knowledge loading, prompt construction, the Ollama request, and output validation.",
    )
    chat.add_argument("--model", required=True, help="exact Ollama tag or configured name")
    chat.add_argument(
        "--dry-run", action="store_true", help="print the request instead of sending it"
    )
    chat.add_argument(
        "--show-prompt", action="store_true", help="include the full system message in the output"
    )
    _add_ollama_arguments(chat)
    chat.add_argument("message")
    chat.set_defaults(handler=_chat)

    evaluate = commands.add_parser(
        "evaluate",
        help="run the evaluation cases against the configured models",
        description="Send every selected case through the shared chat engine for every selected "
        "model and record the machine results in a new directory under results/: manifest.json, "
        "records.jsonl, records.csv, and a blank review sheet, review.csv. Nothing is graded, "
        "scored, or ranked. Ollama is checked first with /api/show, which runs no model.",
    )
    evaluate.add_argument(
        "--cases", metavar="FILE", help="JSON Lines case file (default: eval/cases.jsonl)"
    )
    evaluate.add_argument(
        "--model",
        action="append",
        metavar="TAG_OR_NAME",
        help="evaluate only this configured model; repeat for more (default: all six, in "
        "configuration order)",
    )
    evaluate.add_argument(
        "--case",
        action="append",
        metavar="ID",
        help="run only this case; repeat for more (default: every case, in file order)",
    )
    evaluate.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="run every selected case N times per model (default: 1)",
    )
    evaluate.add_argument(
        "--results-dir",
        metavar="DIR",
        help="where run directories are created (default: results/); inside this repository "
        "only results/ is accepted, because only it is ignored by Git",
    )
    evaluate.add_argument("--note", metavar="TEXT", help="a short note stored in the run's manifest")
    evaluate.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan, with each case's routing and route check, without contacting "
        "Ollama or writing files",
    )
    _add_ollama_arguments(evaluate)
    evaluate.set_defaults(handler=_evaluate)
    return parser


def _add_ollama_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_BASE_URL,
        help=f"local Ollama server; loopback addresses only (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"seconds to wait for a complete response (default: {DEFAULT_TIMEOUT_S:g})",
    )


def _validate(args: argparse.Namespace, transport: Transport | None) -> int:
    client = OllamaClient(args.ollama_url, timeout_s=args.timeout, transport=transport)
    engine = ChatEngine.from_files(client=client)
    knowledge = engine.load_knowledge()
    system_message = PromptBuilder(engine.instruction, knowledge).system_message
    models, routes, chat = engine.models, engine.router.config, engine.chat_config

    print("Configuration is valid.")
    print(f"models     {models.source}  sha256 {models.sha256}")
    print(f"           {_baseline(models)}")
    overrides = [(model.tag, model.override) for model in models.models if model.override]
    print(f"           documented exceptions: {len(overrides) or 'none'}")
    for tag, override in overrides:
        print(f"             {tag}: {_options(override.options)}: {override.reason}")
    for model in models.models:
        print(f"           - {model.tag} ({model.name})")
    terms = ", ".join(term.term for term in routes.ambiguous_terms) or "none"
    print(f"routes     {routes.source}  sha256 {routes.sha256}")
    print(f"           routes: {', '.join(routes.route_ids)}; ambiguous terms: {terms}")
    print(f"chat       {chat.source}  sha256 {chat.sha256}")
    print(
        f"           input up to {chat.max_input_characters} characters; cloud preview "
        f"{chat.cloud_max_lines} lines of {chat.cloud_line_characters} characters; "
        f"continuation route {chat.continuation_route}"
    )
    print(f"prompt     {engine.instruction.source}  sha256 {engine.instruction.sha256}")
    print(f"knowledge  {knowledge.source}  sha256 {knowledge.sha256}")
    print(f"system message: {len(system_message)} characters")
    if not args.ollama:
        print("Ollama was not contacted.")
        return 0
    return _check_ollama(engine, client)


def _check_ollama(engine: ChatEngine, client: OllamaClient) -> int:
    print(f"Ollama at {client.base_url} (model metadata from /api/show; no model is run):")
    missing: list[str] = []
    thinking: list[str] = []
    for model in engine.models.models:
        try:
            details = client.show(model.tag)
        except OllamaError as exc:
            if exc.code is OllamaErrorCode.MODEL_NOT_FOUND:
                print(f"  {model.tag}: NOT FOUND")
                missing.append(model.tag)
                continue
            print(f"error: {exc.message}", file=sys.stderr)
            return 1
        if details.capabilities is None:
            capabilities = "not reported by this Ollama version"
        else:
            capabilities = ", ".join(details.capabilities) or "none"
            if "thinking" in details.capabilities:
                thinking.append(model.tag)
        print(f"  {model.tag}: installed; capabilities: {capabilities}")
        if details.parameters:
            lines = [line.strip() for line in details.parameters.splitlines() if line.strip()]
            print(f"    Modelfile parameters: {' | '.join(lines)}")
    print(f"thinking capability reported by: {', '.join(thinking) or 'none'}")
    if missing:
        print(f"not installed: {', '.join(missing)}")
        return 1
    return 0


def _route(args: argparse.Namespace, transport: Transport | None) -> int:
    engine = ChatEngine.from_files(client=OllamaClient(transport=transport))
    routing = engine.route(args.message)
    if routing.response is not None:
        _print_json(routing.response.to_dict())
        return 0
    if routing.route is None:  # unreachable: only invalid input has no route, and it has a response
        raise ValueError("routing produced neither a route nor a response")
    _print_json(
        {
            "status": routing.route.status.value,
            "message": routing.message,
            "route": route_to_dict(routing.route),
            "next": "not a navigation command: knowledge loading, prompt construction, "
            "the model, and output validation would handle it",
        }
    )
    return 0


def _chat(args: argparse.Namespace, transport: Transport | None) -> int:
    client = OllamaClient(args.ollama_url, timeout_s=args.timeout, transport=transport)
    engine = ChatEngine.from_files(client=client)
    prepared = engine.prepare(args.message, args.model)
    if isinstance(prepared, ChatResponse):  # answered without a model
        _print_json(prepared.to_dict(include_system_message=args.show_prompt))
        return _exit_status(prepared)
    if args.dry_run:
        payload = copy.deepcopy(dict(prepared.payload))
        if not args.show_prompt:
            system = payload["messages"][0]
            system["content"] = (
                f"<system message: {len(system['content'])} characters, sha256 "
                f"{prepared.prompt.system_message_sha256}; --show-prompt prints it>"
            )
        _print_json(
            {
                "dry_run": "nothing was sent",
                "url": client.chat_url,
                "timeout_s": client.timeout_s,
                "payload": payload,
                "prompt": {
                    "system_instruction_sha256": prepared.prompt.system_instruction_sha256,
                    "knowledge_sha256": prepared.prompt.knowledge_sha256,
                    "system_message_sha256": prepared.prompt.system_message_sha256,
                },
                "configs": dict(engine.configs),
            }
        )
        return 0
    response = engine.send(prepared)
    _print_json(response.to_dict(include_system_message=args.show_prompt))
    return _exit_status(response)


def _exit_status(response: ChatResponse) -> int:
    return 1 if response.status is ChatStatus.ERROR else 0


def _evaluate(args: argparse.Namespace, transport: Transport | None) -> int:
    if args.repeat < 1:
        raise ConfigError(f"--repeat must be at least 1, found {args.repeat}")
    client = OllamaClient(args.ollama_url, timeout_s=args.timeout, transport=transport)
    engine = ChatEngine.from_files(client=client)
    runner = EvaluationRunner(engine, results_dir=args.results_dir)
    case_set = load_cases(args.cases, route_ids=engine.router.config.route_ids)
    plan = plan_evaluation(
        engine, case_set, case_ids=args.case, models=args.model, repetitions=args.repeat
    )
    if args.dry_run:
        print("Evaluation plan (dry run: nothing is sent to Ollama and no files are written).")
        _print_plan(plan, runner)
        _print_cases(plan)
        return 0

    print("Evaluation run.")
    _print_plan(plan, runner)
    try:
        result = runner.run(plan, note=args.note, on_start=_print_start, on_record=_print_record)
    except ModelsNotInstalledError as exc:
        print(f"error: {exc}; nothing was run or written", file=sys.stderr)
        return 1
    except OllamaError as exc:
        print(f"error: {exc.message}; nothing was run or written", file=sys.stderr)
        return 1
    _print_result(result)
    return 0 if result.status is RunStatus.COMPLETED else 1


def _print_plan(plan: EvaluationPlan, runner: EvaluationRunner) -> None:
    case_set = plan.case_set
    categories = Counter(planned.case.category for planned in plan.cases)
    print(f"cases      {case_set.source}  sha256 {case_set.sha256}")
    print(
        f"           {len(plan.cases)} of {len(case_set.cases)} selected; categories: "
        + ", ".join(f"{name} {count}" for name, count in categories.items())
    )
    print(f"models     {', '.join(model.tag for model in plan.models)}")
    print(
        f"records    {plan.total_records} ({_count(len(plan.cases), 'case')} x "
        f"{_count(len(plan.models), 'model')} x {_count(plan.repetitions, 'repetition')}); "
        f"Ollama chat requests: {plan.model_requests}"
    )
    print(f"results    {display_path(runner.results_dir / '<run id>')}")


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


def _print_cases(plan: EvaluationPlan) -> None:
    width = max(len(planned.case.id) for planned in plan.cases)
    category_width = max(len(planned.case.category) for planned in plan.cases)
    for planned in plan.cases:
        check = "" if planned.route_check is None else f"  expected route: {planned.route_check.value}"
        print(
            f"  {planned.case.id:<{width}}  {planned.case.category:<{category_width}}  "
            f"{_handled_by(planned)}{check}"
        )
    checked = [planned.route_check for planned in plan.cases if planned.route_check is not None]
    mismatched = checked.count(RouteCheck.MISMATCH)
    print(f"route expectations: {len(checked)} checked, {mismatched} mismatched")


def _handled_by(planned: PlannedCase) -> str:
    status, route = planned.application_status, planned.route
    if status is None:
        return "model"
    detail = ""
    if route is not None and route.route_id is not None:
        detail = f" {route.route_id}"
    elif route is not None and route.candidates:
        detail = f" {', '.join(route.candidates)}"
    return f"application: {status.value}{detail}"


def _print_start(start: RunStart) -> None:
    if start.models_checked:
        print("Every selected model is installed (Ollama /api/show; no model was run).")
    print(f"run        {start.run_id}  ->  {display_path(start.directory)}")


def _print_record(progress: RecordProgress) -> None:
    width = len(str(progress.total))
    line = (
        f"[{progress.sequence:>{width}}/{progress.total}] {progress.model_tag}  {progress.case_id}  "
        f"r{progress.repetition}  {progress.status.value}"
    )
    if progress.error_code is not None:
        line += f" ({progress.error_code})"
    if progress.elapsed_ms is not None:
        line += f"  {progress.elapsed_ms:.0f} ms"
    print(line, flush=True)


def _print_result(result: RunResult) -> None:
    outcome = f"Run {result.status.value}: {result.records_written} of {result.planned_records} planned records written."
    print(outcome)
    if result.stop_reason is not None:
        print(f"stopped: {result.stop_reason}")
    print("statuses: " + (", ".join(f"{name} {count}" for name, count in result.status_counts.items()) or "none"))
    print(
        f"route checks (records): {result.route_checks.get(RouteCheck.MATCH, 0)} match, "
        f"{result.route_checks.get(RouteCheck.MISMATCH, 0)} mismatch"
    )
    print(f"results: {display_path(result.directory)}  ({', '.join(RUN_FILES.values())})")
    print(
        f"Grade in a copy of review.csv using {display_path(RUBRIC_PATH)}; the machine records are "
        "not edited."
    )


def _baseline(models: ModelsConfig) -> str:
    think = "not sent" if models.baseline_think is None else json.dumps(models.baseline_think)
    return f"baseline: think={think}; {_options(models.baseline_options)}"


def _options(options: Any) -> str:
    items: dict[str, OptionValue] = dict(options)
    return ", ".join(f"{name}={json.dumps(value, ensure_ascii=False)}" for name, value in items.items())


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _tolerate_unencodable_output() -> None:
    """Escape characters the console cannot encode instead of crashing.

    A redirected Windows console often uses a legacy code page such as cp1252.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")
