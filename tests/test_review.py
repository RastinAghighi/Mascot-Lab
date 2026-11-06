"""Tests for the human review sheet and its provisional rubric (``eval/rubric.md``)."""

import re

from mascot_lab.measurements import MEASUREMENT_FIELDS
from mascot_lab.paths import PROJECT_ROOT
from mascot_lab.recorder import RECORD_COLUMNS
from mascot_lab.review import (
    CONTEXT_COLUMNS,
    DIMENSIONS,
    GRADES,
    REVIEW_COLUMNS,
    REVIEWER_COLUMNS,
    RUBRIC_PATH,
    order_review_rows,
)

RUBRIC = RUBRIC_PATH.read_text(encoding="utf-8")


def test_the_rubric_is_eval_rubric_md_and_marked_provisional() -> None:
    assert RUBRIC_PATH == PROJECT_ROOT / "eval" / "rubric.md"
    title = RUBRIC.splitlines()[0]
    assert title.startswith("# ") and "PROVISIONAL" in title
    assert "No grades, scores, or results exist yet" in RUBRIC


def test_the_rubric_defines_every_dimension_in_order() -> None:
    headings = re.findall(r"^### (\d+)\. (.+) \(`([a-z_]+)`\)$", RUBRIC, flags=re.MULTILINE)
    assert [(int(number), title, key) for number, title, key in headings] == [
        (index, dimension.title, dimension.id) for index, dimension in enumerate(DIMENSIONS, start=1)
    ]
    sections = re.split(r"^### ", RUBRIC, flags=re.MULTILINE)[1:]
    for section in sections:
        for grade in GRADES:
            assert f"- **{grade}:**" in section, (section.splitlines()[0], grade)


def test_the_rubric_defines_every_grade_value() -> None:
    table = re.findall(r"^\| `([a-z/]+)` \|", RUBRIC, flags=re.MULTILINE)
    assert tuple(table) == GRADES


def test_the_rubric_names_the_review_sheet_columns_it_relies_on() -> None:
    for column in ("message", "answer_text", "raw_output", "expected_behavior", "status", "record_id", "route_check"):
        assert column in CONTEXT_COLUMNS
        assert f"`{column}`" in RUBRIC, column
    assert "`reviewer`" in RUBRIC and "notes" in RUBRIC


def test_the_rubric_defines_no_weights_totals_or_winner() -> None:
    assert "defines no weights, totals, pass thresholds, or decision rule" in RUBRIC
    assert not re.search(r"\b(weight|score)s?\s*[:=]\s*\d", RUBRIC, flags=re.IGNORECASE)


def test_review_columns_hold_context_grades_and_reviewer_fields_only() -> None:
    assert REVIEW_COLUMNS == CONTEXT_COLUMNS + tuple(dimension.id for dimension in DIMENSIONS) + REVIEWER_COLUMNS
    assert len(set(REVIEW_COLUMNS)) == len(REVIEW_COLUMNS)


def test_review_columns_reveal_no_model_and_no_measurement() -> None:
    forbidden = {"model_name", "model_tag", "sequence", "started_at", "repetition", *MEASUREMENT_FIELDS}
    assert not forbidden & set(REVIEW_COLUMNS)
    assert all("sha256" not in column for column in REVIEW_COLUMNS)
    assert forbidden <= set(RECORD_COLUMNS)  # the machine record keeps them


def test_review_rows_are_grouped_by_case_then_ordered_by_record_id() -> None:
    rows = [
        {"case_id": "b", "record_id": "9"},
        {"case_id": "a", "record_id": "5"},
        {"case_id": "b", "record_id": "1"},
        {"case_id": "a", "record_id": "0"},
    ]
    ordered = order_review_rows(rows, ["b", "a"])
    assert [(row["case_id"], row["record_id"]) for row in ordered] == [("b", "1"), ("b", "9"), ("a", "0"), ("a", "5")]
