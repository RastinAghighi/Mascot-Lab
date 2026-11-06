"""Tests for the evaluation case loader (``mascot_lab.cases``).

Every case is synthetic; case files are written under ``tmp_path``.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from evaluation_fakes import AMBIGUOUS_CASE, NAVIGATION_CASE, QUESTION_CASE, write_cases

from mascot_lab.cases import CASES_PATH, ExpectedRoute, load_cases, parse_cases
from mascot_lab.errors import CaseError, MascotLabError
from mascot_lab.paths import PROJECT_ROOT
from mascot_lab.router import Router, RouteResult, RouteStatus

ROUTE_IDS = Router.from_file().config.route_ids


def parse(*lines: str, source: str = "cases.jsonl") -> Any:
    return parse_cases("\n".join(lines) + "\n", source=source, route_ids=ROUTE_IDS)


def line(**fields: Any) -> str:
    case = {"id": "synthetic-1", "category": "synthetic", "message": "Where does Rastin study?"}
    case.update(fields)
    return json.dumps({key: value for key, value in case.items() if value is not ...})


# ------------------------------------------------------------- valid files


def test_a_case_file_loads_in_file_order(tmp_path: Path) -> None:
    path = write_cases(tmp_path / "cases.jsonl", NAVIGATION_CASE, QUESTION_CASE, AMBIGUOUS_CASE)
    case_set = load_cases(path, route_ids=ROUTE_IDS)
    assert case_set.ids == ("synthetic-nav-qmind", "synthetic-question", "synthetic-ambiguous")
    assert case_set.source == str(path)  # outside the project: shown as given
    navigation, question, ambiguous = case_set.cases
    assert navigation.expected_route == ExpectedRoute(RouteStatus.ROUTED, "project_qmind")
    assert (navigation.expected_behavior, navigation.notes, navigation.line) == (None, None, 1)
    assert question.message == "Where does Rastin study?"
    assert question.expected_behavior == "SYNTHETIC-REVIEWER-GUIDANCE"
    assert question.notes == "SYNTHETIC-AUTHOR-NOTE"
    assert question.expected_route == ExpectedRoute(RouteStatus.NOT_NAVIGATION)
    assert ambiguous.expected_route == ExpectedRoute(
        RouteStatus.AMBIGUOUS_DESTINATION, candidates=frozenset({"project_qmind", "project_ingenuity"})
    )
    assert question.to_dict() == QUESTION_CASE
    assert all(len(case.sha256) == 64 for case in case_set.cases)
    assert len({case.sha256 for case in case_set.cases}) == 3


def test_the_default_case_file_is_eval_cases_jsonl() -> None:
    assert CASES_PATH == PROJECT_ROOT / "eval" / "cases.jsonl"


def test_blank_lines_a_bom_and_crlf_change_neither_cases_nor_hashes(tmp_path: Path) -> None:
    plain = write_cases(tmp_path / "plain.jsonl", NAVIGATION_CASE, QUESTION_CASE)
    messy = tmp_path / "messy.jsonl"
    text = "\r\n\r\n" + json.dumps(NAVIGATION_CASE) + "\r\n   \r\n" + json.dumps(QUESTION_CASE)
    messy.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    first, second = (load_cases(path, route_ids=ROUTE_IDS) for path in (plain, messy))
    assert first.sha256 == second.sha256
    assert [case.sha256 for case in first.cases] == [case.sha256 for case in second.cases]
    assert [case.line for case in second.cases] == [3, 5]


def test_hashes_follow_content_not_layout() -> None:
    compact = json.dumps(QUESTION_CASE, separators=(",", ":"))
    reordered = json.dumps(dict(reversed(list(QUESTION_CASE.items()))), indent=None)
    assert parse(compact).sha256 == parse(reordered).sha256
    edited = dict(QUESTION_CASE, message="Where does Rastin study? ")
    assert parse(json.dumps(edited)).cases[0].sha256 != parse(compact).cases[0].sha256
    assert parse(compact, json.dumps(NAVIGATION_CASE)).sha256 != parse(json.dumps(NAVIGATION_CASE), compact).sha256


def test_the_message_is_kept_exactly() -> None:
    message = "  take me to\tQMIND  ​"
    assert parse(line(message=message)).cases[0].message == message


def test_a_whitespace_only_message_is_allowed_to_test_the_empty_input_reply() -> None:
    assert parse(line(message="   ")).cases[0].message == "   "


# ----------------------------------------------------------- invalid files


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("{not json", "invalid JSON"),
        ('["a", "list"]', "must be a JSON object"),
        ('"text"', "must be a JSON object"),
        (line(id=...), "missing required key(s) id"),
        (line(category=...), "missing required key(s) category"),
        (line(message=...), "missing required key(s) message"),
        (line(prompt="extra"), "unknown key(s) prompt"),
        (line(id=7), "id must be lowercase"),
        (line(id="Upper-Case"), "id must be lowercase"),
        (line(id="two--dashes"), "id must be lowercase"),
        (line(id="trailing-"), "id must be lowercase"),
        (line(id="has space"), "id must be lowercase"),
        (line(id="x" * 65), "at most 64"),
        (line(category=""), "category must be lowercase"),
        (line(category="Navigation"), "category must be lowercase"),
        (line(message=""), "message must be a non-empty string"),
        (line(message=["take me to QMIND"]), "message must be a non-empty string"),
        (line(expected_behavior=""), "expected_behavior must be a non-empty string"),
        (line(expected_behavior=" padded"), "leading or trailing whitespace"),
        (line(notes=3), "notes must be a non-empty string"),
        ('{"id": "a", "id": "b", "category": "c", "message": "m"}', "duplicate key 'id'"),
        ('{"id": "a", "category": "c", "message": "m", "x": NaN}', "NaN is not a valid value"),
        ('{"id": "a", "category": "c", "message": "\\ud800"}', "lone surrogate"),
        (line(expected_behavior="\\ud800".encode().decode("unicode_escape")), "lone surrogate"),
        (line(expected_route="routed"), "expected_route must be an object"),
        (line(expected_route={}), "missing required key(s) status"),
        (line(expected_route={"status": "routed", "url": "x"}), "unknown key(s) url"),
        (line(expected_route={"status": "navigated"}), "status must be one of"),
        (line(expected_route={"status": "routed"}), "route_id is required"),
        (line(expected_route={"status": "routed", "route_id": "project_papers"}), "not an allowlisted route ID"),
        (line(expected_route={"status": "routed", "route_id": "mascot_chat"}), "not an allowlisted route ID"),
        (line(expected_route={"status": "not_navigation", "route_id": "project_qmind"}), "only when status is routed"),
        (line(expected_route={"status": "routed", "route_id": "project_qmind", "candidates": []}), "only when status"),
        (line(expected_route={"status": "ambiguous_destination", "candidates": ["project_qmind"]}), "at least two"),
        (line(expected_route={"status": "ambiguous_destination", "candidates": "project_qmind"}), "at least two"),
        (
            line(expected_route={"status": "ambiguous_destination", "candidates": ["project_qmind"] * 2}),
            "must not repeat",
        ),
        (
            line(expected_route={"status": "ambiguous_destination", "candidates": ["project_qmind", "/x"]}),
            "not an allowlisted route ID",
        ),
    ],
)
def test_invalid_cases_are_rejected_with_their_line_number(text: str, error: str) -> None:
    with pytest.raises(CaseError) as caught:
        parse(line(id="synthetic-ok"), text)
    message = str(caught.value)
    assert message.startswith("cases.jsonl:2: ")
    assert error in message


def test_duplicate_ids_name_both_lines() -> None:
    with pytest.raises(CaseError, match=r"cases.jsonl:3: duplicate case ID 'synthetic-1' \(first on line 1\)"):
        parse(line(), line(id="synthetic-2"), line())


@pytest.mark.parametrize("text", ["", "\n\n", "   \n\t\n"])
def test_a_file_without_cases_is_rejected(text: str) -> None:
    with pytest.raises(CaseError, match="has no cases"):
        parse_cases(text, source="cases.jsonl", route_ids=ROUTE_IDS)


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(CaseError, match="evaluation case file not found"):
        load_cases(tmp_path / "absent.jsonl", route_ids=ROUTE_IDS)


def test_a_file_that_is_not_utf8_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "latin1.jsonl"
    path.write_bytes(line(message="café").encode("utf-8").replace(b"\\u00e9", b"\xe9"))
    with pytest.raises(CaseError, match="not valid UTF-8"):
        load_cases(path, route_ids=ROUTE_IDS)


def test_case_errors_are_mascot_lab_errors() -> None:
    assert issubclass(CaseError, MascotLabError)


# --------------------------------------------------------------- selection


def test_select_returns_the_named_cases_in_file_order() -> None:
    case_set = parse(line(id="a"), line(id="b"), line(id="c"))
    assert [case.id for case in case_set.select(["c", "a", "c"])] == ["a", "c"]
    assert case_set.select(None) == case_set.cases


def test_select_rejects_unknown_ids_and_empty_selections() -> None:
    case_set = parse(line(id="a"))
    with pytest.raises(CaseError, match="no case with ID 'z'"):
        case_set.select(["a", "z"])
    with pytest.raises(CaseError, match="no cases selected"):
        case_set.select([])


# ---------------------------------------------------------- expected route


@pytest.mark.parametrize(
    ("expected", "route", "matches"),
    [
        (ExpectedRoute(RouteStatus.ROUTED, "project_qmind"), RouteResult(RouteStatus.ROUTED, "project_qmind", (), ("QMIND",)), True),
        (ExpectedRoute(RouteStatus.ROUTED, "project_qmind"), RouteResult(RouteStatus.ROUTED, "project_satira", (), ("Satira",)), False),
        (ExpectedRoute(RouteStatus.NOT_NAVIGATION), RouteResult(RouteStatus.NOT_NAVIGATION), True),
        (ExpectedRoute(RouteStatus.NOT_NAVIGATION), RouteResult(RouteStatus.URL_REJECTED), False),
        (ExpectedRoute(RouteStatus.NOT_NAVIGATION), None, False),
        (
            ExpectedRoute(RouteStatus.AMBIGUOUS_DESTINATION, candidates=frozenset({"project_ingenuity", "project_qmind"})),
            RouteResult(RouteStatus.AMBIGUOUS_DESTINATION, None, ("project_qmind", "project_ingenuity"), ("research",)),
            True,
        ),
        (
            ExpectedRoute(RouteStatus.AMBIGUOUS_DESTINATION, candidates=frozenset({"project_qmind", "project_satira"})),
            RouteResult(RouteStatus.AMBIGUOUS_DESTINATION, None, ("project_qmind", "project_ingenuity"), ("research",)),
            False,
        ),
        (
            ExpectedRoute(RouteStatus.AMBIGUOUS_DESTINATION),
            RouteResult(RouteStatus.AMBIGUOUS_DESTINATION, None, ("project_qmind", "project_ingenuity"), ("research",)),
            True,
        ),
    ],
)
def test_expected_routes_compare_status_route_and_candidates(
    expected: ExpectedRoute, route: RouteResult | None, matches: bool
) -> None:
    assert expected.matches(route) is matches


def test_expected_routes_hold_for_the_synthetic_cases() -> None:
    router = Router.from_file()
    case_set = parse(json.dumps(NAVIGATION_CASE), json.dumps(QUESTION_CASE), json.dumps(AMBIGUOUS_CASE))
    for case in case_set.cases:
        assert case.expected_route is not None
        assert case.expected_route.matches(router.route(case.message)), case.id
