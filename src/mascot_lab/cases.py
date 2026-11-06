"""Evaluation cases: one version-controlled JSON Lines file for every model.

Every configured model receives the same cases, from one file (default
``eval/cases.jsonl``). The file holds one JSON object per line, in a stable
order; blank lines are ignored. Each object is one visitor message::

    {"id": "projects-satira-detail", "category": "projects",
     "message": "What does Satira do?",
     "expected_behavior": "Says that no description of Satira is available.",
     "expected_route": {"status": "not_navigation"},
     "notes": "facts.md P04"}

* ``id`` (required): unique within the file; lowercase letters and digits,
  optionally joined by single ``.``, ``_``, or ``-``; at most 64 characters.
* ``category`` (required): the same form; groups cases, for example by the
  areas of the handoff's test protocol (section 5).
* ``message`` (required): the exact text sent through the chat engine as the
  visitor's message. It is not trimmed or changed here: the engine's input
  validation handles it as it would any message, so a whitespace-only
  message tests the empty-input reply. It must not be the empty string.
* ``expected_behavior`` (optional): guidance for human reviewers, shown on
  the review sheet. It is never sent to a model.
* ``expected_route`` (optional): what deterministic routing must return for
  the message, checked by the runner without any model. ``status`` is a
  router status; ``route_id`` is required with ``routed`` and not allowed
  otherwise; ``candidates`` (two or more route IDs, compared as a set) is
  allowed only with ``ambiguous_destination``. Route IDs must be in the
  navigation allowlist.
* ``notes`` (optional): notes for case authors, such as the facts a case
  covers. Recorded with the results, never shown on the review sheet, and
  never sent to a model.

Unknown keys, duplicate keys, duplicate IDs, non-finite numbers, and values
of the wrong type are rejected with their line number: loading fails closed
with ``CaseError``. The evaluation set itself is written and frozen by the
owner; none ships with the code.
"""

import dataclasses
import json
import os
import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mascot_lab._textfile import read_text_file, sha256_hex
from mascot_lab.errors import CaseError
from mascot_lab.paths import PROJECT_ROOT, display_path
from mascot_lab.router import RouteResult, RouteStatus

CASES_PATH = PROJECT_ROOT / "eval" / "cases.jsonl"
MAX_ID_LENGTH = 64

_KEYS = frozenset({"id", "category", "message", "expected_behavior", "expected_route", "notes"})
_REQUIRED_KEYS = frozenset({"id", "category", "message"})
_ROUTE_KEYS = frozenset({"status", "route_id", "candidates"})
_SLUG = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


@dataclass(frozen=True, slots=True)
class ExpectedRoute:
    """What deterministic routing must return for a case's message."""

    status: RouteStatus
    route_id: str | None = None  # set exactly when ``status`` is ROUTED
    candidates: frozenset[str] | None = None  # None: the candidates are not checked

    def matches(self, route: RouteResult | None) -> bool:
        """Whether ``route`` (None when input validation refused the message) meets this."""
        if route is None or route.status is not self.status:
            return False
        if self.route_id is not None and route.route_id != self.route_id:
            return False
        return self.candidates is None or frozenset(route.candidates) == self.candidates

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"status": self.status.value}
        if self.route_id is not None:
            data["route_id"] = self.route_id
        if self.candidates is not None:
            data["candidates"] = sorted(self.candidates)
        return data


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One validated case."""

    id: str
    category: str
    message: str
    expected_behavior: str | None
    expected_route: ExpectedRoute | None
    notes: str | None
    line: int  # where the case is in its file, for messages
    sha256: str  # of the case's canonical JSON form, so layout does not affect it

    def to_dict(self) -> dict[str, Any]:
        """Return the case's canonical form: its keys, without absent optional ones."""
        data: dict[str, Any] = {"id": self.id, "category": self.category, "message": self.message}
        if self.expected_behavior is not None:
            data["expected_behavior"] = self.expected_behavior
        if self.expected_route is not None:
            data["expected_route"] = self.expected_route.to_dict()
        if self.notes is not None:
            data["notes"] = self.notes
        return data


@dataclass(frozen=True, slots=True)
class CaseSet:
    """The validated cases of one file, in file order."""

    cases: tuple[EvaluationCase, ...]
    source: str
    sha256: str  # of the canonical JSON forms of all cases, in file order

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(case.id for case in self.cases)

    def select(self, ids: Iterable[str] | None = None) -> tuple[EvaluationCase, ...]:
        """Return the cases named by ``ids`` in file order (all when ``ids`` is None).

        A name that is not a case ID raises ``CaseError``; repeated names count once.
        """
        if ids is None:
            return self.cases
        wanted = list(dict.fromkeys(ids))
        unknown = [case_id for case_id in wanted if case_id not in self.ids]
        if unknown:
            raise CaseError(f"{self.source}: no case with ID {', '.join(map(repr, unknown))}")
        if not wanted:
            raise CaseError(f"{self.source}: no cases selected")
        return tuple(case for case in self.cases if case.id in wanted)


def load_cases(
    path: str | os.PathLike[str] | None = None, *, route_ids: Collection[str]
) -> CaseSet:
    """Load and validate a case file (default: ``eval/cases.jsonl``).

    ``route_ids`` is the navigation allowlist that expected routes may name.
    The file is only read, never written.
    """
    file_path = CASES_PATH if path is None else Path(path)
    text = read_text_file(file_path, error=CaseError, label="evaluation case file")
    return parse_cases(text, source=display_path(file_path), route_ids=route_ids)


def parse_cases(text: str, *, source: str = "<string>", route_ids: Collection[str]) -> CaseSet:
    """Validate JSON Lines ``text``; raise ``CaseError`` on any problem."""
    allowed_routes = frozenset(route_ids)
    cases: list[EvaluationCase] = []
    lines_by_id: dict[str, int] = {}
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        where = f"{source}:{number}"
        case = _case(_json_object(line, where), where, number, allowed_routes)
        if case.id in lines_by_id:
            raise CaseError(f"{where}: duplicate case ID {case.id!r} (first on line {lines_by_id[case.id]})")
        lines_by_id[case.id] = number
        cases.append(case)
    if not cases:
        raise CaseError(f"{source}: the file has no cases")
    return CaseSet(
        cases=tuple(cases),
        source=source,
        sha256=sha256_hex(_canonical([case.to_dict() for case in cases])),
    )


class _Rejected(Exception):
    """A JSON construct the case format does not allow."""


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key, value in pairs:
        if key in data:
            raise _Rejected(f"duplicate key {key!r}")
        data[key] = value
    return data


def _no_constants(name: str) -> Any:
    raise _Rejected(f"{name} is not a valid value")


def _json_object(line: str, where: str) -> dict[str, Any]:
    try:
        data = json.loads(line, object_pairs_hook=_unique_keys, parse_constant=_no_constants)
    except _Rejected as exc:
        raise CaseError(f"{where}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise CaseError(f"{where}: invalid JSON: {exc.msg} (column {exc.colno})") from None
    if not isinstance(data, dict):
        raise CaseError(f"{where}: each line must be a JSON object")
    return data


def _case(data: dict[str, Any], where: str, line: int, route_ids: frozenset[str]) -> EvaluationCase:
    _check_keys(data, where, _KEYS, _REQUIRED_KEYS)
    case_id = _slug(data["id"], f"{where}: id")
    category = _slug(data["category"], f"{where}: category")
    message = data["message"]
    if not isinstance(message, str) or not message:
        raise CaseError(f"{where}: message must be a non-empty string")
    _check_unicode(message, f"{where}: message")
    expected_behavior = _optional_text(data, "expected_behavior", where)
    notes = _optional_text(data, "notes", where)
    expected_route = None
    if "expected_route" in data:
        expected_route = _expected_route(data["expected_route"], f"{where}: expected_route", route_ids)
    case = EvaluationCase(
        id=case_id,
        category=category,
        message=message,
        expected_behavior=expected_behavior,
        expected_route=expected_route,
        notes=notes,
        line=line,
        sha256="",
    )
    return dataclasses.replace(case, sha256=sha256_hex(_canonical(case.to_dict())))


def _expected_route(value: object, where: str, route_ids: frozenset[str]) -> ExpectedRoute:
    if not isinstance(value, dict):
        raise CaseError(f"{where} must be an object")
    _check_keys(value, where, _ROUTE_KEYS, frozenset({"status"}))
    raw_status = value["status"]
    statuses = [status.value for status in RouteStatus]
    if not isinstance(raw_status, str) or raw_status not in statuses:
        raise CaseError(f"{where}.status must be one of {', '.join(statuses)}, found {raw_status!r}")
    status = RouteStatus(raw_status)

    route_id = None
    if status is RouteStatus.ROUTED:
        if "route_id" not in value:
            raise CaseError(f"{where}: route_id is required when status is routed")
        route_id = _allowlisted(value["route_id"], f"{where}.route_id", route_ids)
    elif "route_id" in value:
        raise CaseError(f"{where}: route_id is allowed only when status is routed")

    candidates = None
    if "candidates" in value:
        if status is not RouteStatus.AMBIGUOUS_DESTINATION:
            raise CaseError(f"{where}: candidates are allowed only when status is ambiguous_destination")
        raw = value["candidates"]
        if not isinstance(raw, list) or len(raw) < 2:
            raise CaseError(f"{where}.candidates must be a list of at least two route IDs")
        names = [_allowlisted(item, f"{where}.candidates", route_ids) for item in raw]
        if len(set(names)) != len(names):
            raise CaseError(f"{where}.candidates must not repeat a route ID")
        candidates = frozenset(names)
    return ExpectedRoute(status=status, route_id=route_id, candidates=candidates)


def _allowlisted(value: object, where: str, route_ids: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in route_ids:
        raise CaseError(
            f"{where}: {value!r} is not an allowlisted route ID; allowlisted: "
            f"{', '.join(sorted(route_ids)) or 'none'}"
        )
    return value


def _check_keys(data: dict[str, Any], where: str, allowed: frozenset[str], required: frozenset[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise CaseError(f"{where}: unknown key(s) {', '.join(unknown)}; allowed: {', '.join(sorted(allowed))}")
    missing = sorted(required - set(data))
    if missing:
        raise CaseError(f"{where}: missing required key(s) {', '.join(missing)}")


def _slug(value: object, where: str) -> str:
    if not isinstance(value, str) or not _SLUG.fullmatch(value) or len(value) > MAX_ID_LENGTH:
        raise CaseError(
            f"{where} must be lowercase letters and digits, optionally joined by single '.', '_', "
            f"or '-', with at most {MAX_ID_LENGTH} characters; found {value!r}"
        )
    return value


def _optional_text(data: dict[str, Any], key: str, where: str) -> str | None:
    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise CaseError(f"{where}: {key} must be a non-empty string")
    if value != value.strip():
        raise CaseError(f"{where}: {key} must not have leading or trailing whitespace")
    _check_unicode(value, f"{where}: {key}")
    return value


def _check_unicode(value: str, where: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise CaseError(f"{where} is not valid Unicode text (it contains a lone surrogate)") from None


def _canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
