"""Results recorder: one directory of machine output per evaluation run.

::

    results/<run_id>/
        manifest.json   the run: settings, hashes, model metadata, the exact
                        system message, and the final status
        records.jsonl   one complete machine record per response, exact
        records.csv     the same records as flat columns, one row each
        review.csv      the blank, blind human review sheet (eval/rubric.md)

``results/`` is ignored by Git. The records hold raw model output and the
manifest holds the system message with the private knowledge block, so
inside this repository the recorder writes only under ``results/``. A
directory outside the repository is accepted, for tests and scratch runs.
A run never reuses or overwrites a directory or a file.

Records are appended and flushed one at a time, so an interrupted run keeps
everything recorded before it stopped. ``manifest.json`` is written when
the run starts, with status ``running``, and replaced when it ends.

A value that was not measured or not reported is ``null`` in the JSON files
and an empty cell in the CSV files; nothing is estimated or filled in. CSV
files are UTF-8 with a byte-order mark, so spreadsheet programs detect the
encoding; lists are joined with ``"; "`` and booleans are ``true`` or
``false``. A text cell that begins with ``=``, ``+``, ``-``, ``@``, a tab,
or a carriage return gets a leading apostrophe, so a spreadsheet program
does not run model output as a formula. ``records.jsonl`` keeps the exact
text.
"""

import csv
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import IO, Any

from mascot_lab.errors import ConfigError
from mascot_lab.measurements import MEASUREMENT_FIELDS
from mascot_lab.paths import PROJECT_ROOT, display_path
from mascot_lab.review import REVIEW_COLUMNS, action_labels, continuation_reasons

SCHEMA_VERSION = 1
RESULTS_DIR = PROJECT_ROOT / "results"

MANIFEST_FILE = "manifest.json"
RECORDS_JSONL_FILE = "records.jsonl"
RECORDS_CSV_FILE = "records.csv"
REVIEW_FILE = "review.csv"
RUN_FILES: Mapping[str, str] = MappingProxyType(
    {
        "manifest": MANIFEST_FILE,
        "records_jsonl": RECORDS_JSONL_FILE,
        "records_csv": RECORDS_CSV_FILE,
        "review_sheet": REVIEW_FILE,
    }
)

# The columns of records.csv, in order.
RECORD_COLUMNS: tuple[str, ...] = (
    "run_id",
    "sequence",
    "record_id",
    "started_at",
    "case_id",
    "category",
    "repetition",
    "model_name",
    "model_tag",
    "status",
    "request_sent",
    "route_status",
    "route_id",
    "route_candidates",
    "route_check",
    "actions",
    "message",
    "answer_text",
    "continuation",
    "output_issues",
    "raw_output",
    "raw_thinking",
    "done_reason",
    "error_code",
    "error_message",
    "http_status",
    *MEASUREMENT_FIELDS,
    "models_sha256",
    "routes_sha256",
    "chat_sha256",
    "system_instruction_sha256",
    "knowledge_sha256",
    "system_message_sha256",
    "case_sha256",
)

_FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")
_LINE_BREAK_ESCAPES = str.maketrans({"\x85": "\\u0085", " ": "\\u2028", " ": "\\u2029"})


def check_results_dir(path: str | os.PathLike[str] | None = None) -> Path:
    """Return the absolute results directory for ``path`` (default: ``results/``).

    Inside this repository, only ``results/`` and directories under it are
    accepted, because only ``results/`` is ignored by Git.
    """
    target = (RESULTS_DIR if path is None else Path(path)).resolve()
    if target.is_relative_to(PROJECT_ROOT) and not target.is_relative_to(RESULTS_DIR):
        raise ConfigError(
            f"results directory {display_path(target)!r} is inside the repository but not under "
            "results/: evaluation output holds raw model output and the private knowledge block, "
            "and only results/ is ignored by Git"
        )
    return target


class RunRecorder:
    """Writes the files of one new run directory."""

    def __init__(self, directory: Path) -> None:
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise ConfigError(
                f"run directory {display_path(directory)} already exists; a run never reuses one"
            ) from None
        self._directory = directory
        self._jsonl = _create(directory / RECORDS_JSONL_FILE, csv_file=False)
        self._csv_file = _create(directory / RECORDS_CSV_FILE, csv_file=True)
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(RECORD_COLUMNS)
        self._csv_file.flush()

    @property
    def directory(self) -> Path:
        return self._directory

    def write_manifest(self, manifest: Mapping[str, Any]) -> None:
        """Write ``manifest.json``, replacing the previous version in one step."""
        path = self._directory / MANIFEST_FILE
        temporary = path.with_name(path.name + ".tmp")
        with _create(temporary, csv_file=False, replace=True) as handle:
            handle.write(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, path)

    def write_record(self, record: Mapping[str, Any]) -> None:
        """Append one record to ``records.jsonl`` and ``records.csv`` and flush both."""
        self._jsonl.write(json_line(record) + "\n")
        self._jsonl.flush()
        self._csv.writerow(csv_cells(record_row(record), RECORD_COLUMNS))
        self._csv_file.flush()

    def write_review_sheet(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Write ``review.csv`` with ``rows`` (from ``mascot_lab.review``) and blank grades."""
        with _create(self._directory / REVIEW_FILE, csv_file=True) as handle:
            writer = csv.writer(handle)
            writer.writerow(REVIEW_COLUMNS)
            for row in rows:
                writer.writerow(csv_cells(row, REVIEW_COLUMNS))

    def close(self) -> None:
        self._jsonl.close()
        self._csv_file.close()


def json_line(record: Mapping[str, Any]) -> str:
    """Return ``record`` as one line of JSON.

    Non-ASCII text stays readable, except the three characters that some
    line splitters (Python's ``str.splitlines``, for one) treat as line
    breaks but JSON does not escape: they are written as ``\\uXXXX`` escapes,
    which decode to the same text.
    """
    text = json.dumps(record, ensure_ascii=False, allow_nan=False)
    return text.translate(_LINE_BREAK_ESCAPES)


def record_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one machine record into the ``records.csv`` columns."""
    response = record["response"]
    route = response["route"] or {}
    output = response["output"] or {}
    error = response["error"] or {}
    prompt = response["prompt"] or {}
    configs = response["configs"]
    return {
        "run_id": record["run_id"],
        "sequence": record["sequence"],
        "record_id": record["record_id"],
        "started_at": record["started_at"],
        "case_id": record["case"]["id"],
        "category": record["case"]["category"],
        "repetition": record["repetition"],
        "model_name": record["model"]["name"],
        "model_tag": record["model"]["tag"],
        "status": record["status"],
        "request_sent": record["request_sent"],
        "route_status": route.get("status"),
        "route_id": route.get("route_id"),
        "route_candidates": route.get("candidates"),
        "route_check": record["route_check"],
        "actions": action_labels(response["actions"]),
        "message": record["case"]["message"],
        "answer_text": response["answer_text"],
        "continuation": continuation_reasons(response["continuation"]),
        "output_issues": response["output_issues"],
        "raw_output": output.get("content"),
        "raw_thinking": output.get("thinking"),
        "done_reason": output.get("done_reason"),
        "error_code": error.get("code"),
        "error_message": error.get("message"),
        "http_status": error.get("http_status"),
        **{name: record["measurements"][name] for name in MEASUREMENT_FIELDS},
        "models_sha256": configs["models"],
        "routes_sha256": configs["routes"],
        "chat_sha256": configs["chat"],
        "system_instruction_sha256": configs["system_instruction"],
        "knowledge_sha256": prompt.get("knowledge_sha256"),
        "system_message_sha256": prompt.get("system_message_sha256"),
        "case_sha256": record["case"]["sha256"],
    }


def csv_cells(row: Mapping[str, Any], columns: Sequence[str]) -> list[str]:
    """Return the CSV cells of ``row`` for ``columns``.

    ``row`` must hold a value (None for an empty cell) for every column and no other key.
    """
    unknown = sorted(set(row) - set(columns))
    if unknown:
        raise ValueError(f"values for unknown columns: {', '.join(unknown)}")
    return [csv_cell(row[column]) for column in columns]


def csv_cell(value: Any) -> str:
    """Return one CSV cell: empty for None, ``true``/``false``, numbers, or safe text."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list | tuple):
        value = "; ".join(str(item) for item in value)
    if not isinstance(value, str):
        raise TypeError(f"no CSV form for {type(value).__name__}")
    return "'" + value if value.startswith(_FORMULA_STARTS) else value


def _create(path: Path, *, csv_file: bool, replace: bool = False) -> IO[str]:
    """Open a new text file: UTF-8 (with a byte-order mark for CSV), LF lines for JSON.

    A lone surrogate that cannot be encoded is written as a ``\\uXXXX``
    escape instead of failing, which in JSON is the same character.
    """
    return open(  # the caller closes it
        path,
        "w" if replace else "x",
        encoding="utf-8-sig" if csv_file else "utf-8",
        errors="backslashreplace",
        newline="" if csv_file else "\n",
    )
