"""Human review: the provisional rubric's dimensions and the blank review sheet.

Grades are human judgments, kept apart from machine measurements. Each run
writes ``review.csv`` beside its machine records: one row per record, with
the context a reviewer needs and one blank column per dimension of the
provisional rubric in ``eval/rubric.md``. The code never fills in a grade
and never computes a score, a weight, a total, or a ranking.

The sheet is blind. It omits the model's name and tag and every timing and
token measurement, and it lists each case's records in ``record_id`` order.
A record ID is an opaque hash, so that order does not follow the model
order. ``record_id`` links a graded row back to ``records.csv`` once
grading is done.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mascot_lab.paths import PROJECT_ROOT

RUBRIC_PATH = PROJECT_ROOT / "eval" / "rubric.md"


@dataclass(frozen=True, slots=True)
class Dimension:
    """One rubric dimension: its review sheet column and its heading in the rubric."""

    id: str
    title: str


# PROVISIONAL, like eval/rubric.md, which defines each dimension.
DIMENSIONS: tuple[Dimension, ...] = (
    Dimension("factual_accuracy", "Factual accuracy"),
    Dimension("scope", "Scope"),
    Dimension("unknown_handling", "Unknown handling"),
    Dimension("disclosure_boundaries", "Disclosure boundaries"),
    Dimension("safety", "Safety"),
    Dimension("navigation", "Navigation"),
    Dimension("brevity", "Brevity"),
)

# PROVISIONAL grade values, defined in eval/rubric.md.
GRADES: tuple[str, ...] = ("pass", "partial", "fail", "n/a")

# What a reviewer sees for each record: no model identity and no measurements.
# ``route_check`` comes from deterministic routing, which does not depend on the model.
CONTEXT_COLUMNS: tuple[str, ...] = (
    "record_id",
    "case_id",
    "category",
    "message",
    "expected_behavior",
    "status",
    "actions",
    "route_check",
    "answer_text",
    "continuation",
    "raw_output",
    "output_issues",
)
REVIEWER_COLUMNS: tuple[str, ...] = ("reviewer", "notes")
REVIEW_COLUMNS: tuple[str, ...] = (
    CONTEXT_COLUMNS + tuple(dimension.id for dimension in DIMENSIONS) + REVIEWER_COLUMNS
)


def review_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the review sheet row for one machine record; every grade is blank (None)."""
    case = record["case"]
    response = record["response"]
    output = response["output"]
    row: dict[str, Any] = dict.fromkeys(REVIEW_COLUMNS)
    row.update(
        record_id=record["record_id"],
        case_id=case["id"],
        category=case["category"],
        message=case["message"],
        expected_behavior=case.get("expected_behavior"),
        status=record["status"],
        actions=action_labels(response["actions"]),
        route_check=record["route_check"],
        answer_text=response["answer_text"],
        continuation=continuation_reasons(response["continuation"]),
        raw_output=None if output is None else output["content"],
        output_issues=list(response["output_issues"]),
    )
    return row


def order_review_rows(rows: Sequence[Mapping[str, Any]], case_ids: Sequence[str]) -> list[Mapping[str, Any]]:
    """Group rows by case, in ``case_ids`` order; within a case, order them by record ID."""
    position = {case_id: index for index, case_id in enumerate(case_ids)}
    return sorted(rows, key=lambda row: (position[row["case_id"]], row["record_id"]))


def action_labels(actions: Sequence[Mapping[str, str]]) -> list[str]:
    """Navigation actions as ``type:route_id`` labels, such as ``navigate:project_qmind``."""
    return [f"{action['type']}:{action['route_id']}" for action in actions]


def continuation_reasons(continuation: Mapping[str, Any] | None) -> list[str]:
    """Why an answer continues on the mascot's page; empty when it fits the cloud."""
    return [] if continuation is None else list(continuation["reasons"])
