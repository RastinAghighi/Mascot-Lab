"""Tests for ``mascot_lab.origin_request``: the origin's request checks before the engine.

The checks follow the public chat API v1 contract (rastinaghighi.dev,
``docs/api/chat-v1.md``, sections 4 to 6, PROPOSED). The request schema is
the byte-for-byte copy in ``tests/contracts/``, validated with the standard
``jsonschema`` Draft 2020-12 validator; the website repository is never
read. No test opens a socket, starts a listener or calls a model, and the
randomized tests use fixed seeds.
"""

import ast
import dataclasses
import json
import random
import socket
import subprocess
import sys
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

import mascot_lab.origin_request as origin_request
from mascot_lab.chat_config import load_chat_config
from mascot_lab.origin_request import (
    ECMA_262_WHITESPACE,
    MAX_BODY_BYTES,
    REJECTION_OUTCOMES,
    ChatRequest,
    RejectionReason,
    RequestErrorCode,
    RequestRejected,
    check_chat_request,
)

R = RejectionReason
MODULE_PATH = Path(origin_request.__file__)
SRC = MODULE_PATH.parents[1]
REQUEST_SCHEMA_PATH = Path(__file__).resolve().parent / "contracts" / "chat-request-v1.schema.json"

# The engine's configured limit, which the HTTP adapter will pass in.
LIMIT = load_chat_config().max_input_characters


def load_request_validator() -> Draft202012Validator:
    schema = json.loads(REQUEST_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)
    return Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)


REQUEST_VALIDATOR = load_request_validator()

# The six characters that Python's \s (str.isspace) and ECMA-262's \s classify
# differently. The schema's "pattern": "\\S" is evaluated by jsonschema with
# Python's re module, while the contract (JSON Schema) means ECMA-262.
#   code point: (whitespace to Python, whitespace to ECMA-262)
WHITESPACE_DIFFERENCES: dict[str, tuple[bool, bool]] = {
    "\u001c": (True, False),  # INFORMATION SEPARATOR FOUR (FS), Cc, bidi B
    "\u001d": (True, False),  # INFORMATION SEPARATOR THREE (GS), Cc, bidi B
    "\u001e": (True, False),  # INFORMATION SEPARATOR TWO (RS), Cc, bidi B
    "\u001f": (True, False),  # INFORMATION SEPARATOR ONE (US), Cc, bidi S
    "\u0085": (True, False),  # NEXT LINE (NEL), Cc, bidi B
    "﻿": (False, True),  # ZERO WIDTH NO-BREAK SPACE (BOM), Cf; ECMA-262 <ZWNBSP>
}


def body_for(message: Any, *, ascii_only: bool = False) -> bytes:
    return json.dumps({"message": message}, ensure_ascii=ascii_only).encode("utf-8")


VALID_BODY = body_for("What kinds of projects are featured on this site?")


def check(body: bytes = VALID_BODY, **overrides: Any) -> ChatRequest | RequestRejected:
    arguments: dict[str, Any] = {
        "method": "POST",
        "query_string": "",
        "content_type": "application/json",
        "content_encoding": None,
        "content_length": len(body) if isinstance(body, bytes) else None,
        "body": body,
        "max_characters": LIMIT,
    }
    arguments.update(overrides)
    return check_chat_request(**arguments)


def accepted(result: ChatRequest | RequestRejected) -> str:
    assert isinstance(result, ChatRequest), result
    return result.message


def reason(result: ChatRequest | RequestRejected) -> RejectionReason:
    assert isinstance(result, RequestRejected), result
    return result.reason


def assert_rejected(
    result: ChatRequest | RequestRejected, expected: RejectionReason, status: int, code: str
) -> None:
    assert isinstance(result, RequestRejected), result
    assert (result.reason, result.http_status, result.error_code) == (expected, status, code)


# ------------------------------------------------------------- the table

# The contract's table, written out here rather than read from the module.
EXPECTED_OUTCOMES = {
    R.METHOD_NOT_ALLOWED: (1, 405, "invalid_request"),
    R.QUERY_STRING: (2, 400, "invalid_request"),
    R.UNSUPPORTED_CONTENT_TYPE: (3, 415, "invalid_request"),
    R.UNSUPPORTED_CONTENT_ENCODING: (3, 415, "invalid_request"),
    R.DECLARED_BODY_TOO_LARGE: (4, 413, "invalid_request"),
    R.BODY_TOO_LARGE: (4, 413, "invalid_request"),
    R.BYTE_ORDER_MARK: (5, 400, "invalid_request"),
    R.INVALID_UTF8: (5, 400, "invalid_request"),
    R.INVALID_JSON: (6, 400, "invalid_request"),
    R.DUPLICATE_MEMBER: (6, 400, "invalid_request"),
    R.NONSTANDARD_CONSTANT: (6, 400, "invalid_request"),
    R.NESTING_TOO_DEEP: (6, 400, "invalid_request"),
    R.NOT_AN_OBJECT: (7, 400, "invalid_request"),
    R.UNEXPECTED_MEMBER: (7, 400, "invalid_request"),
    R.MISSING_MESSAGE: (7, 400, "invalid_request"),
    R.MESSAGE_NOT_A_STRING: (7, 400, "invalid_request"),
    R.UNPAIRED_SURROGATE: (8, 400, "invalid_request"),
    R.EMPTY_MESSAGE: (9, 400, "empty_input"),
    R.MESSAGE_TOO_LONG: (10, 400, "input_too_long"),
}

# One request per reason, each failing only the step it names.
STEP_CASES: list[tuple[str, dict[str, Any], RejectionReason]] = [
    ("1-get", {"method": "GET"}, R.METHOD_NOT_ALLOWED),
    ("2-query", {"query_string": "a=1"}, R.QUERY_STRING),
    ("3-content-type", {"content_type": "text/plain"}, R.UNSUPPORTED_CONTENT_TYPE),
    ("3-content-encoding", {"content_encoding": "gzip"}, R.UNSUPPORTED_CONTENT_ENCODING),
    ("4-declared", {"content_length": MAX_BODY_BYTES + 1}, R.DECLARED_BODY_TOO_LARGE),
    ("4-actual", {"body": VALID_BODY + b" " * MAX_BODY_BYTES, "content_length": None}, R.BODY_TOO_LARGE),
    ("5-bom", {"body": b"\xef\xbb\xbf" + VALID_BODY}, R.BYTE_ORDER_MARK),
    ("5-utf8", {"body": b'{"message": "caf\xe9"}'}, R.INVALID_UTF8),
    ("6-json", {"body": b'{"message": "a"'}, R.INVALID_JSON),
    ("6-duplicate", {"body": b'{"message": "a", "message": "a"}'}, R.DUPLICATE_MEMBER),
    ("6-nan", {"body": b'{"message": NaN}'}, R.NONSTANDARD_CONSTANT),
    ("6-nesting", {"body": b"[" * 4000 + b"]" * 4000}, R.NESTING_TOO_DEEP),
    ("7-array", {"body": b'["a"]'}, R.NOT_AN_OBJECT),
    ("7-extra", {"body": b'{"message": "a", "history": []}'}, R.UNEXPECTED_MEMBER),
    ("7-missing", {"body": b"{}"}, R.MISSING_MESSAGE),
    ("7-type", {"body": b'{"message": null}'}, R.MESSAGE_NOT_A_STRING),
    ("8-surrogate", {"body": b'{"message": "a\\ud800b"}'}, R.UNPAIRED_SURROGATE),
    ("9-empty", {"body": b'{"message": ""}'}, R.EMPTY_MESSAGE),
    ("9-spaces", {"body": b'{"message": " \\t\\n "}'}, R.EMPTY_MESSAGE),
    ("10-long", {"body": body_for("a" * (LIMIT + 1))}, R.MESSAGE_TOO_LONG),
]


def test_the_outcome_table_matches_the_contract() -> None:
    assert dict(REJECTION_OUTCOMES) == EXPECTED_OUTCOMES
    assert set(REJECTION_OUTCOMES) == set(RejectionReason)
    assert {code.value for code in RequestErrorCode} == {"invalid_request", "empty_input", "input_too_long"}
    assert sorted({outcome.step for outcome in REJECTION_OUTCOMES.values()}) == list(range(1, 11))


def test_every_reason_has_a_step_case() -> None:
    assert {expected for _, _, expected in STEP_CASES} == set(RejectionReason)


@pytest.mark.parametrize(("overrides", "expected"), [c[1:] for c in STEP_CASES], ids=[c[0] for c in STEP_CASES])
def test_each_step_rejects_with_its_status_and_code(overrides: dict[str, Any], expected: RejectionReason) -> None:
    _, status, code = EXPECTED_OUTCOMES[expected]
    assert_rejected(check(**overrides), expected, status, code)


def test_a_valid_request_is_accepted() -> None:
    assert check() == ChatRequest("What kinds of projects are featured on this site?")
    assert check(content_length=None) == check()  # no Content-Length, for example a chunked body


def test_the_contract_example_request_is_accepted() -> None:
    body = (REQUEST_SCHEMA_PATH.parent / "examples" / "request.json").read_bytes()
    expected = json.loads(body.decode("utf-8"))["message"]
    assert accepted(check(body)) == expected


@pytest.mark.parametrize("method", ["GET", "HEAD", "PUT", "PATCH", "DELETE", "OPTIONS", "post", "Post", " POST", "POST ", ""])
def test_only_post_is_allowed(method: str) -> None:
    assert_rejected(check(method=method), R.METHOD_NOT_ALLOWED, 405, "invalid_request")


@pytest.mark.parametrize("query", ["a=1", "message=hello", "?", " ", "=", "&"])
def test_any_query_string_is_rejected(query: str) -> None:
    assert_rejected(check(query_string=query), R.QUERY_STRING, 400, "invalid_request")


# ------------------------------------------------------------ precedence

PRECEDENCE_CASES: list[tuple[str, dict[str, Any], RejectionReason]] = [
    (
        "method before everything",
        {"method": "GET", "query_string": "x", "content_type": None, "content_encoding": "gzip", "body": b"\xff" * 9000},
        R.METHOD_NOT_ALLOWED,
    ),
    ("query before content type", {"query_string": "x", "content_type": "text/plain"}, R.QUERY_STRING),
    ("content type before encoding", {"content_type": None, "content_encoding": "br"}, R.UNSUPPORTED_CONTENT_TYPE),
    ("encoding before size", {"content_encoding": "gzip", "body": b"x" * 9000}, R.UNSUPPORTED_CONTENT_ENCODING),
    ("oversized invalid UTF-8", {"body": b"\xff" * (MAX_BODY_BYTES + 1), "content_length": None}, R.BODY_TOO_LARGE),
    ("oversized BOM", {"body": b"\xef\xbb\xbf" + b" " * MAX_BODY_BYTES, "content_length": None}, R.BODY_TOO_LARGE),
    ("oversized declared invalid UTF-8", {"body": b"\xff" * (MAX_BODY_BYTES + 1)}, R.DECLARED_BODY_TOO_LARGE),
    ("declared size before invalid UTF-8", {"body": b"\xff", "content_length": MAX_BODY_BYTES + 1}, R.DECLARED_BODY_TOO_LARGE),
    ("BOM before invalid UTF-8", {"body": b"\xef\xbb\xbf\xff"}, R.BYTE_ORDER_MARK),
    ("invalid UTF-8 before invalid JSON", {"body": b"{\xff"}, R.INVALID_UTF8),
    ("duplicate before extra member", {"body": b'{"message": "a", "x": {"k": 1, "k": 2}}'}, R.DUPLICATE_MEMBER),
    ("constant before wrong type", {"body": b'{"message": [Infinity]}'}, R.NONSTANDARD_CONSTANT),
    ("extra member before missing message", {"body": b'{"Message": "a"}'}, R.UNEXPECTED_MEMBER),
    ("extra member before wrong type", {"body": b'{"message": 1, "stream": true}'}, R.UNEXPECTED_MEMBER),
    ("extra member before a surrogate", {"body": b'{"message": "\\ud800", "\\udc00": 1}'}, R.UNEXPECTED_MEMBER),
    ("surrogate before empty", {"body": b'{"message": "\\udfff"}'}, R.UNPAIRED_SURROGATE),
    ("surrogate in otherwise blank text", {"body": b'{"message": "   \\ud800   "}'}, R.UNPAIRED_SURROGATE),
    ("surrogate before too long", {"body": body_for("a" * (LIMIT + 5) + "\ud800", ascii_only=True)}, R.UNPAIRED_SURROGATE),
    ("1,001 spaces are empty, not too long", {"body": body_for(" " * (LIMIT + 1))}, R.EMPTY_MESSAGE),
    ("1,001 U+FEFF are empty, not too long", {"body": body_for("﻿" * (LIMIT + 1))}, R.EMPTY_MESSAGE),
]


@pytest.mark.parametrize(
    ("overrides", "expected"), [c[1:] for c in PRECEDENCE_CASES], ids=[c[0] for c in PRECEDENCE_CASES]
)
def test_the_first_failure_wins(overrides: dict[str, Any], expected: RejectionReason) -> None:
    _, status, code = EXPECTED_OUTCOMES[expected]
    assert_rejected(check(**overrides), expected, status, code)


# ------------------------------------------------------------- body size


def padded_body(size: int, message: str = "Synthetic question") -> bytes:
    """A valid request body of exactly ``size`` bytes, padded with JSON whitespace."""
    body = body_for(message)
    assert len(body) <= size
    return body + b" " * (size - len(body))


def test_the_body_limit_is_the_contract_value() -> None:
    assert MAX_BODY_BYTES == 8192


def test_a_body_of_exactly_the_limit_is_accepted() -> None:
    body = padded_body(MAX_BODY_BYTES)
    assert len(body) == 8192
    assert accepted(check(body)) == "Synthetic question"
    assert accepted(check(body, content_length=None)) == "Synthetic question"


def test_a_body_one_byte_over_the_limit_is_rejected() -> None:
    body = padded_body(MAX_BODY_BYTES + 1)
    assert len(body) == 8193
    assert_rejected(check(body), R.DECLARED_BODY_TOO_LARGE, 413, "invalid_request")
    assert_rejected(check(body, content_length=None), R.BODY_TOO_LARGE, 413, "invalid_request")
    assert_rejected(check(body, content_length=10), R.BODY_TOO_LARGE, 413, "invalid_request")


def test_the_declared_length_is_checked_at_the_boundary() -> None:
    assert accepted(check(content_length=MAX_BODY_BYTES))
    assert reason(check(content_length=MAX_BODY_BYTES + 1)) is R.DECLARED_BODY_TOO_LARGE
    assert reason(check(content_length=10**30)) is R.DECLARED_BODY_TOO_LARGE


def test_a_declared_length_that_differs_from_the_body_is_left_to_the_transport() -> None:
    # Framing belongs to the transport; the checks always apply to the bytes passed in.
    assert accepted(check(content_length=0)) == accepted(check(content_length=MAX_BODY_BYTES))
    assert reason(check(b'{"message": ""}', content_length=MAX_BODY_BYTES)) is R.EMPTY_MESSAGE


def test_the_largest_standard_serialization_fits() -> None:
    # Contract section 5: 1,000 code points, each written in at most 6 bytes, plus 14 bytes of structure.
    message = "\u0001" * LIMIT  # JSON.stringify writes a control character as a 6-byte escape
    body = json.dumps({"message": message}, separators=(",", ":")).encode("utf-8")
    assert len(body) == 14 + 6 * LIMIT <= MAX_BODY_BYTES
    assert accepted(check(body)) == message


def test_a_valid_message_written_entirely_as_surrogate_pair_escapes_exceeds_the_limit() -> None:
    message = "\U0001f600" * LIMIT
    body = body_for(message, ascii_only=True)  # 12 bytes per emoji
    assert len(body) > MAX_BODY_BYTES
    assert reason(check(body)) is R.DECLARED_BODY_TOO_LARGE
    assert accepted(check(body_for(message))) == message  # 4 bytes per emoji as UTF-8


# -------------------------------------------------------- message length


def test_the_configured_limit_is_the_schema_limit() -> None:
    assert LIMIT == REQUEST_VALIDATOR.schema["properties"]["message"]["maxLength"] == 1000


@pytest.mark.parametrize(
    "unit",
    [
        "a",
        "é",  # 2 bytes in UTF-8
        "中",  # 3 bytes
        "\U0001f600",  # 4 bytes, outside the BMP: 2 UTF-16 code units, 1 code point
        "\U0001d11e",  # MUSICAL SYMBOL G CLEF, outside the BMP
    ],
)
def test_length_is_counted_in_code_points(unit: str) -> None:
    at_limit = unit * LIMIT
    over = unit * (LIMIT + 1)
    assert accepted(check(body_for(at_limit))) == at_limit
    assert_rejected(check(body_for(over)), R.MESSAGE_TOO_LONG, 400, "input_too_long")


def test_an_emoji_sequence_counts_each_code_point() -> None:
    family = "\U0001f468‍\U0001f469‍\U0001f467"  # 5 code points, one visible character
    assert len(family) == 5
    at_limit = family * (LIMIT // 5)
    assert accepted(check(body_for(at_limit))) == at_limit
    assert reason(check(body_for(at_limit + "x"))) is R.MESSAGE_TOO_LONG


def test_surrounding_whitespace_counts_toward_the_length() -> None:
    message = " " + "a" * (LIMIT - 1)
    assert accepted(check(body_for(message))) == message
    assert reason(check(body_for(message + " "))) is R.MESSAGE_TOO_LONG


@pytest.mark.parametrize("limit", [1, 5, 1500, 2000])
def test_the_limit_comes_from_the_caller(limit: int) -> None:
    assert accepted(check(body_for("x" * limit), max_characters=limit)) == "x" * limit
    assert reason(check(body_for("x" * (limit + 1)), max_characters=limit)) is R.MESSAGE_TOO_LONG


# ------------------------------------------------ Content-Type and Encoding

ACCEPTED_CONTENT_TYPES = [
    "application/json",
    "APPLICATION/JSON",
    "Application/Json",
    "application/json; charset=utf-8",
    "application/json;charset=utf-8",
    "application/json; charset=UTF-8",
    "application/json; CHARSET=Utf-8",
    "application/json ; charset=utf-8",
    "application/json\t;\tcharset=utf-8",
    "application/json  ;  charset=utf-8",
    " application/json",  # surrounding SP and HTAB are not part of a field value (RFC 9110, 5.5)
    "application/json\t",
    "\t application/json; charset=utf-8 \t",
]

REJECTED_CONTENT_TYPES = [
    None,
    "",
    " ",
    "text/json",
    "text/plain",
    "text/plain; charset=utf-8",
    "application/x-www-form-urlencoded",
    "multipart/form-data; boundary=x",
    "application/problem+json",
    "application/json-seq",
    "application/jsonx",
    "application/json5",
    "application/*",
    "*/*",
    "json",
    "application/json; charset=utf8",
    "application/json; charset=utf-16",
    "application/json; charset=utf-32",
    "application/json; charset=iso-8859-1",
    "application/json; charset=us-ascii",
    'application/json; charset="utf-8"',  # quoted value: equal to the token in RFC 9110, refused here
    "application/json; charset='utf-8'",
    "application/json; charset = utf-8",
    "application/json; charset= utf-8",
    "application/json; charset =utf-8",
    "application/json; charset=utf-8;",
    "application/json;",
    "application/json; ",
    "application/json;;charset=utf-8",
    "application/json; charset=utf-8; charset=utf-8",
    "application/json; charset=utf-8; boundary=x",
    "application/json; boundary=x",
    "application/json; version=1",
    "application/json; q=1",
    "application/json; charset",
    "application/json; charset=",
    "application/json, application/json",
    "application/json,text/plain",
    "application /json",
    "application/ json",
    "application/json\r\n",
    "application/json\n",
    "application/json\x00",
    "application/json\x0b",
    " application/json",  # NO-BREAK SPACE is not HTTP whitespace
    "application/json　",
    "application/jſon",  # LONG S, which Unicode case folding equates with "s"
    "application/json; charſet=utf-8",
    "ａpplication/json",  # FULLWIDTH LATIN SMALL LETTER A
    "application/json; charset=utf‑8",  # NON-BREAKING HYPHEN
]


@pytest.mark.parametrize("content_type", ACCEPTED_CONTENT_TYPES)
def test_permitted_content_types(content_type: str) -> None:
    assert accepted(check(content_type=content_type))


@pytest.mark.parametrize("content_type", REJECTED_CONTENT_TYPES)
def test_other_content_types_are_rejected(content_type: str | None) -> None:
    assert_rejected(check(content_type=content_type), R.UNSUPPORTED_CONTENT_TYPE, 415, "invalid_request")


@pytest.mark.parametrize("content_encoding", [None, "identity", "IDENTITY", "Identity", " identity", "identity\t"])
def test_permitted_content_encodings(content_encoding: str | None) -> None:
    assert accepted(check(content_encoding=content_encoding))


@pytest.mark.parametrize(
    "content_encoding",
    [
        "",
        " ",
        "gzip",
        "x-gzip",
        "deflate",
        "br",
        "zstd",
        "compress",
        "dcz",
        "*",
        "identity, gzip",
        "gzip, identity",
        "identity, identity",
        "identity,",
        "identity;q=1",
        "identit",
        "identityy",
        "ıdentity",  # DOTLESS I, which upper-cases to "I"
        "İdentity",  # CAPITAL I WITH DOT ABOVE
        "identity\x00",
        "identity\r\n",
        " identity",
    ],
)
def test_other_content_encodings_are_rejected(content_encoding: str) -> None:
    assert_rejected(check(content_encoding=content_encoding), R.UNSUPPORTED_CONTENT_ENCODING, 415, "invalid_request")


def test_unicode_case_folding_would_accept_the_look_alikes() -> None:
    # Documents why only ASCII is folded: Python's own case mappings equate these with the ASCII tokens.
    assert "application/jſon".upper() == "APPLICATION/JSON"
    assert "ıdentity".upper() == "IDENTITY"


# ------------------------------------------------------------------- JSON

MALFORMED_JSON = [
    b"",
    b" ",
    b"{",
    b"}",
    b"{]",
    b'{"message": "a"',
    b'{"message": "a",}',
    b'{"message": "a"} {}',
    b'{"message": "a"} x',
    b'{"message": "a"}\x00',
    b"{'message': 'a'}",
    b'{message: "a"}',
    b'{"message" "a"}',
    b'{"message": "a" // comment\n}',
    b'/* c */ {"message": "a"}',
    b'{"message": "a\tb"}',  # a raw control character inside a string
    b'{"message": "a\nb"}',
    b'{"message": "a\x00b"}',
    b'{"message": "\\x41"}',
    b'{"message": "\\u12"}',
    b'{"message": "\\u12G4"}',
    b'{"message": "\\U0001F600"}',
    b'{"message": "\\\'"}',
    b'{"message": "a\\"}',
    b'{"message": "a", "n": 01}',
    b'{"message": "a", "n": 0x10}',
    b'{"message": "a", "n": +1}',
    b'{"message": "a", "n": .5}',
    b'{"message": "a", "n": 1.}',
    b'{"message": "a", "n": -}',
    b'{"message": "a", "n": 1e}',
    b'{"message": "a", "n": undefined}',
    b'{"message": "a", "n": True}',
    b'{"message": "a", "n": nan}',
    b'{"message": "a", "n": infinity}',
    b'\x0c{"message": "a"}',  # form feed is not JSON whitespace
    b'\x0b{"message": "a"}',
    b'{"message":\xc2\xa0"a"}',  # NO-BREAK SPACE is not JSON whitespace
    b'{"message": "a"}\xe2\x80\xa8',  # LINE SEPARATOR is not JSON whitespace
    b' \xef\xbb\xbf{"message": "a"}',  # a BOM after leading whitespace is an ordinary character
    b'{"message": "a"}\xef\xbb\xbf',
    # UTF-16 and UTF-32 without a BOM are valid UTF-8 (ASCII and NUL bytes) but not JSON.
    '{"message": "a"}'.encode("utf-16-le"),
    '{"message": "a"}'.encode("utf-16-be"),
    '{"message": "a"}'.encode("utf-32-le"),
    '{"message": "a"}'.encode("utf-32-be"),
]


@pytest.mark.parametrize("body", MALFORMED_JSON, ids=[repr(b)[:40] for b in MALFORMED_JSON])
def test_malformed_json_is_rejected(body: bytes) -> None:
    assert_rejected(check(body), R.INVALID_JSON, 400, "invalid_request")


@pytest.mark.parametrize(
    "body",
    [
        b'{"message": "a", "message": "a"}',
        b'{"message": "a", "message": "b"}',
        b'{"message": "a", "\\u006dessage": "b"}',  # the same name, written with an escape
        b'{"message": "a", "x": {"k": 1, "k": 2}}',
        b'{"message": "a", "x": [{"k": 1}, {"k": 1, "k": 1}]}',
        b'{"message": {"a": {"b": {"c": 1, "c": 1}}}}',
        b'[{"message": "a", "message": "a"}]',
        b'{"": 1, "": 2}',
    ],
)
def test_duplicate_member_names_are_rejected_at_every_level(body: bytes) -> None:
    assert_rejected(check(body), R.DUPLICATE_MEMBER, 400, "invalid_request")


@pytest.mark.parametrize(
    "body",
    [
        b'{"message": NaN}',
        b'{"message": Infinity}',
        b'{"message": -Infinity}',
        b'{"message": "a", "x": [1, NaN]}',
        b'{"message": "a", "x": {"y": -Infinity}}',
        b"NaN",
        b"Infinity",
    ],
)
def test_nonstandard_constants_are_rejected(body: bytes) -> None:
    assert_rejected(check(body), R.NONSTANDARD_CONSTANT, 400, "invalid_request")


@pytest.mark.parametrize(
    "body",
    [
        b"[" * 4000 + b"]" * 4000,
        b"[" * (MAX_BODY_BYTES // 2) + b"]" * (MAX_BODY_BYTES // 2),
        b'{"message": ' + b"[" * 4000 + b"]" * 4000 + b"}",
        b"[" * MAX_BODY_BYTES,  # unbalanced: the parser recurses before it finds the error
    ],
    ids=["arrays", "arrays-at-limit", "inside-message", "unbalanced"],
)
def test_excessive_nesting_is_rejected(body: bytes) -> None:
    assert len(body) <= MAX_BODY_BYTES
    assert_rejected(check(body), R.NESTING_TOO_DEEP, 400, "invalid_request")


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (b'{"message": ' + b"[" * 50 + b"]" * 50 + b"}", R.MESSAGE_NOT_A_STRING),
        # The deepest object nesting that fits the limit parses without RecursionError in CPython 3.12.
        (b'{"":' * 1638 + b"1" + b"}" * 1638, R.UNEXPECTED_MEMBER),
        (b'{"message":' * 682 + b'"a"' + b"}" * 682, R.MESSAGE_NOT_A_STRING),
        (b'[{"":' * 1170 + b"1" + b"}]" * 1170, R.NOT_AN_OBJECT),
    ],
    ids=["arrays-in-message", "objects-at-limit", "messages-at-limit", "mixed-at-limit"],
)
def test_nesting_the_parser_accepts_fails_the_structure_check(body: bytes, expected: RejectionReason) -> None:
    assert len(body) <= MAX_BODY_BYTES
    assert reason(check(body)) is expected


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        # More digits than Python's int() accepts by default (4,300): never converted.
        (b'{"message": ' + b"9" * 5000 + b"}", R.MESSAGE_NOT_A_STRING),
        (b'{"message": "a", "n": ' + b"9" * 8000 + b"}", R.UNEXPECTED_MEMBER),
        (b"9" * 8192, R.NOT_AN_OBJECT),
        (b"-" + b"9" * 8191, R.NOT_AN_OBJECT),
        (b'{"message": 1e999999999}', R.MESSAGE_NOT_A_STRING),
        (b'{"message": -1e-999999999}', R.MESSAGE_NOT_A_STRING),
        (b'{"message": 0.' + b"1" * 6000 + b"}", R.MESSAGE_NOT_A_STRING),
    ],
    ids=["int-in-message", "int-in-extra-member", "top-level-int", "top-level-negative", "huge-exp", "tiny-exp", "long-fraction"],
)
def test_huge_numeric_literals_are_controlled(body: bytes, expected: RejectionReason) -> None:
    assert len(body) <= MAX_BODY_BYTES
    assert reason(check(body)) is expected


@pytest.mark.parametrize(
    "body", [b"[]", b'["a"]', b'"a"', b"1", b"-0", b"1.5", b"null", b"true", b"false", b'[{"message": "a"}]']
)
def test_a_top_level_value_that_is_not_an_object_is_rejected(body: bytes) -> None:
    assert_rejected(check(body), R.NOT_AN_OBJECT, 400, "invalid_request")


REJECTED_MEMBERS = [
    "history", "messages", "context", "conversation_id", "session_id", "model", "options", "temperature",
    "system", "prompt", "stream", "url", "route", "route_id", "action", "request_id", "client_request_id",
    "user", "metadata", "Message", "MESSAGE", "message ", " message", "", "message\u0000", "méssage",
]


@pytest.mark.parametrize("member", REJECTED_MEMBERS)
def test_every_other_member_is_rejected(member: str) -> None:
    body = json.dumps({"message": "a", member: "x"}).encode("utf-8")
    assert_rejected(check(body), R.UNEXPECTED_MEMBER, 400, "invalid_request")
    alone = json.dumps({member: "a"}).encode("utf-8")
    assert reason(check(alone)) is R.UNEXPECTED_MEMBER


def test_a_missing_message_is_rejected() -> None:
    assert_rejected(check(b"{}"), R.MISSING_MESSAGE, 400, "invalid_request")
    assert_rejected(check(b" { } "), R.MISSING_MESSAGE, 400, "invalid_request")


@pytest.mark.parametrize("value", [b"null", b"1", b"0", b"-1.5", b"true", b"false", b"[]", b'["a"]', b"{}", b'{"text": "a"}'])
def test_a_message_that_is_not_a_string_is_rejected(value: bytes) -> None:
    assert_rejected(check(b'{"message": ' + value + b"}"), R.MESSAGE_NOT_A_STRING, 400, "invalid_request")


def test_json_whitespace_around_the_value_is_allowed() -> None:
    assert accepted(check(b' \t\r\n{ "message" : "a" }\r\n\t ')) == "a"


# ------------------------------------------------------------------ UTF-8


def test_a_leading_byte_order_mark_is_rejected() -> None:
    assert_rejected(check(b"\xef\xbb\xbf" + VALID_BODY), R.BYTE_ORDER_MARK, 400, "invalid_request")
    assert_rejected(check(b"\xef\xbb\xbf"), R.BYTE_ORDER_MARK, 400, "invalid_request")


@pytest.mark.parametrize(
    "body",
    [
        '{"message": "a"}'.encode("utf-16"),  # with BOM FF FE
        b"\xfe\xff" + '{"message": "a"}'.encode("utf-16-be"),  # with BOM FE FF
        '{"message": "a"}'.encode("utf-32"),  # with BOM FF FE 00 00
        b'{"message": "\xc0\xaf"}',  # overlong "/"
        b'{"message": "\xc0\x80"}',  # overlong NUL (modified UTF-8)
        b'{"message": "\xe0\x80\xaf"}',  # 3-byte overlong "/"
        b'{"message": "\xf0\x80\x80\xaf"}',  # 4-byte overlong "/"
        b'{"message": "\xed\xa0\x80"}',  # an encoded surrogate, U+D800
        b'{"message": "\xed\xa0\xbd\xed\xb8\x80"}',  # CESU-8 surrogate pair for U+1F600
        b'{"message": "\xf4\x90\x80\x80"}',  # above U+10FFFF
        b'{"message": "\xf8\x88\x80\x80\x80"}',  # 5-byte form
        b'{"message": "\xfc\x84\x80\x80\x80\x80"}',  # 6-byte form
        b'{"message": "\x80"}',  # a lone continuation byte
        b'{"message": "\xe2\x82"}',  # a truncated sequence
        b'{"message": "a"}\xe2\x82',  # truncated at the end of the body
        b'{"message": "caf\xe9"}',  # Latin-1
        b'{"message": "\xff"}',
        b'{"message": "\xfe"}',
    ],
)
def test_bodies_that_are_not_strict_utf8_are_rejected(body: bytes) -> None:
    assert_rejected(check(body), R.INVALID_UTF8, 400, "invalid_request")


def test_utf8_bodies_python_would_detect_as_utf16_are_not_decoded_as_utf16() -> None:
    body = '{"message": "a"}'.encode("utf-16-le")
    assert json.loads(body) == {"message": "a"}  # Python's own bytes detection accepts it
    assert reason(check(body)) is R.INVALID_JSON


# ---------------------------------------------------------- surrogates


@pytest.mark.parametrize(
    "escaped",
    ["\\ud800", "\\udbff", "\\udc00", "\\udfff", "a\\ud83d", "\\ude00b", "\\ude00\\ud83d", "\\ud83d\\ud83d", "\\ud83d \\ude00"],
)
def test_unpaired_surrogate_escapes_are_rejected(escaped: str) -> None:
    body = ('{"message": "' + escaped + '"}').encode("ascii")
    assert_rejected(check(body), R.UNPAIRED_SURROGATE, 400, "invalid_request")


def test_a_paired_surrogate_escape_is_one_code_point() -> None:
    assert accepted(check(b'{"message": "\\ud83d\\ude00"}')) == "\U0001f600"
    assert accepted(check(b'{"message": "\\uD83D\\uDE00"}')) == "\U0001f600"


# ---------------------------------------------------------------- Unicode


def test_the_whitespace_constant_is_ecma_262_whitespace() -> None:
    zs = {chr(c) for c in range(sys.maxunicode + 1) if unicodedata.category(chr(c)) == "Zs"}
    assert ECMA_262_WHITESPACE == zs | set("\t\v\f﻿") | set("\n\r  ")
    assert len(ECMA_262_WHITESPACE) == 25
    assert "᠎" not in ECMA_262_WHITESPACE  # MONGOLIAN VOWEL SEPARATOR left Zs in Unicode 6.3


def test_python_and_ecma_262_whitespace_differ_in_exactly_six_characters() -> None:
    python = {chr(c) for c in range(sys.maxunicode + 1) if chr(c).isspace()}
    differences = python ^ ECMA_262_WHITESPACE
    assert differences == set(WHITESPACE_DIFFERENCES)
    for char, (to_python, to_ecma) in WHITESPACE_DIFFERENCES.items():
        assert (char.isspace(), char in ECMA_262_WHITESPACE) == (to_python, to_ecma)


@pytest.mark.parametrize("char", sorted(ECMA_262_WHITESPACE), ids=lambda c: f"U+{ord(c):04X}")
def test_a_message_of_only_ecma_262_whitespace_is_empty(char: str) -> None:
    assert_rejected(check(body_for(char)), R.EMPTY_MESSAGE, 400, "empty_input")
    assert_rejected(check(body_for(char * 3, ascii_only=True)), R.EMPTY_MESSAGE, 400, "empty_input")
    assert accepted(check(body_for(char + "a" + char))) == char + "a" + char


@pytest.mark.parametrize("char", ["\u001c", "\u001d", "\u001e", "\u001f", "\u0085"], ids=lambda c: f"U+{ord(c):04X}")
def test_python_only_whitespace_is_not_empty(char: str) -> None:
    # str.isspace() is True, but ECMA-262 does not treat it as whitespace: accepted.
    assert char.isspace()
    assert accepted(check(body_for(char))) == char
    assert accepted(check(body_for(" " + char + " "))) == " " + char + " "


def test_u_feff_alone_is_empty() -> None:
    # str.isspace() is False, but ECMA-262 treats ZWNBSP as whitespace: empty_input.
    assert not "﻿".isspace()
    assert_rejected(check(body_for("﻿")), R.EMPTY_MESSAGE, 400, "empty_input")
    assert_rejected(check(b'{"message": "\\ufeff \\ufeff"}'), R.EMPTY_MESSAGE, 400, "empty_input")


@pytest.mark.parametrize(
    "char",
    ["​", "‌", "‍", "⁠", "᠎", "­", "\u0000", "\u0001", "\u007f", "‎", "ㅤ"],
    ids=lambda c: f"U+{ord(c):04X}",
)
def test_invisible_characters_that_are_not_whitespace_are_accepted(char: str) -> None:
    # Neither Python nor ECMA-262 whitespace. The engine removes the format and
    # control characters afterwards and may still answer empty_input (contract 6.2, O3).
    assert not char.isspace() and char not in ECMA_262_WHITESPACE
    assert accepted(check(body_for(char))) == char


PRESERVED_MESSAGES = [
    "  surrounding spaces  ",
    "\ttabs\t",
    "line\r\nbreaks\rand\nmore\n",
    "é and é",  # decomposed and precomposed: not normalized to NFC
    "ﬁ ① Ａ ²",  # ligature, circled digit, fullwidth A, superscript: not NFKC
    "İSTANBUL Straße Σσς K",  # no case folding
    "zero​width‍joiner⁠",
    "‮right-to-left override‬ and ‏ marks",
    "\x01\x02 controls \x7f \x9f \x85",
    " separators ",
    "nul \x00 inside",
    "a﻿b",
    "\U0001f468‍\U0001f469‍\U0001f467 \U0001f44d\U0001f3fd \U0001f1ee\U0001f1f7",
    "\U00010348 \U0001d11e \U0010fffd",
    "中文 العربية हिन्दी",
    "x" * LIMIT,
]


@pytest.mark.parametrize("message", PRESERVED_MESSAGES, ids=range(len(PRESERVED_MESSAGES)))
@pytest.mark.parametrize("ascii_only", [False, True], ids=["utf8", "escaped"])
def test_an_accepted_message_is_returned_exactly(message: str, ascii_only: bool) -> None:
    body = body_for(message, ascii_only=ascii_only)
    if len(body) > MAX_BODY_BYTES:
        pytest.fail("test message too large for its body")
    result = accepted(check(body))
    assert result == message
    assert [ord(c) for c in result] == [ord(c) for c in message]
    assert type(result) is str


def test_escaped_and_literal_characters_decode_to_the_same_message() -> None:
    literal = check('{"message": "café \U0001f600 /"}'.encode("utf-8"))
    escaped = check(b'{"message": "caf\\u00e9 \\ud83d\\ude00 \\/"}')
    assert accepted(literal) == accepted(escaped) == "café \U0001f600 /"


# ------------------------------------------------------ schema agreement


def schema_code(instance: Any) -> str | None:
    """The public error code the schema's failures map to (schema ``$comment``), or None if valid."""
    keywords = {error.validator for error in REQUEST_VALIDATOR.iter_errors(instance)}
    if not keywords:
        return None
    if keywords & {"type", "required", "additionalProperties"}:
        return "invalid_request"
    if keywords & {"minLength", "pattern"}:
        return "empty_input"
    assert keywords == {"maxLength"}, keywords
    return "input_too_long"


def our_code(result: ChatRequest | RequestRejected) -> str | None:
    return None if isinstance(result, ChatRequest) else result.error_code.value


def expected_disagreement(message: str) -> tuple[str | None, str | None] | None:
    """(schema, ours) for a message where the six documented characters change the outcome."""
    python_blank = all(c.isspace() for c in message)
    ecma_blank = all(c in ECMA_262_WHITESPACE for c in message)
    if python_blank == ecma_blank:
        return None
    schema = "empty_input" if python_blank else schema_code({"message": message})
    ours = "empty_input" if ecma_blank else ("input_too_long" if len(message) > LIMIT else None)
    return schema, ours


def assert_agrees_with_schema(message: str) -> None:
    instance = {"message": message}
    result = check(body_for(message))
    disagreement = expected_disagreement(message)
    if disagreement is None:
        assert our_code(result) == schema_code(instance), ascii(message)
    else:
        assert (schema_code(instance), our_code(result)) == disagreement, ascii(message)
        assert set(message) & set(WHITESPACE_DIFFERENCES), ascii(message)


def sweep_code_points() -> Iterator[int]:
    yield from range(0xD800)  # the whole BMP below the surrogates
    yield from range(0xE000, 0x10000)
    yield from range(0x10000, sys.maxunicode + 1, 257)  # a sample of the supplementary planes
    yield from (0x1FFFF, 0x10FFFF, 0xE0001, 0xE0020)


def test_single_code_points_agree_with_the_schema_except_the_six() -> None:
    disagreements = {}
    for code_point in sweep_code_points():
        char = chr(code_point)
        assert_agrees_with_schema(char)
        if expected_disagreement(char) is not None:
            disagreements[char] = (schema_code({"message": char}), our_code(check(body_for(char))))
    assert disagreements == {
        "\u001c": ("empty_input", None),
        "\u001d": ("empty_input", None),
        "\u001e": ("empty_input", None),
        "\u001f": ("empty_input", None),
        "\u0085": ("empty_input", None),
        "﻿": (None, "empty_input"),
    }


@pytest.mark.parametrize(
    "message",
    [
        "a", " a ", "", " ", "\t\n", "　 ", "a" * LIMIT, "a" * (LIMIT + 1), " " * (LIMIT + 1),
        "\U0001f600" * LIMIT, "\U0001f600" * (LIMIT + 1), "\u001f" * (LIMIT + 1), "﻿" * (LIMIT + 1),
        "\u001c\u001d\u001e\u001f\u0085", " \u001f ", "﻿ ", "﻿a", "\u0085" * 3 + "a",
        "\u001c" + " " * LIMIT, "﻿" + "a" * LIMIT,
    ],
    ids=lambda m: ascii(m)[:24],
)
def test_messages_agree_with_the_schema_except_the_six(message: str) -> None:
    assert_agrees_with_schema(message)


STRUCTURE_CASES: list[Any] = [
    {"message": "a"},
    {},
    {"message": None},
    {"message": 1},
    {"message": 1.5},
    {"message": True},
    {"message": []},
    {"message": {}},
    {"message": ["a"]},
    {"message": "a", "history": []},
    {"message": "a", "stream": False},
    {"history": []},
    {"Message": "a"},
    {"message": "", "x": 1},
    {"message": "a" * (LIMIT + 1), "x": 1},
    [],
    ["a"],
    "a",
    1,
    None,
    True,
    [{"message": "a"}],
]


@pytest.mark.parametrize("instance", STRUCTURE_CASES, ids=lambda i: json.dumps(i)[:30])
def test_structures_agree_with_the_schema(instance: Any) -> None:
    assert our_code(check(json.dumps(instance).encode("utf-8"))) == schema_code(instance)


def test_unpaired_surrogates_are_a_runtime_check_the_schema_cannot_make() -> None:
    # Documented in the schema's description and contract section 18.1; not one of the six.
    assert schema_code({"message": "\ud800"}) is None
    assert reason(check(b'{"message": "\\ud800"}')) is R.UNPAIRED_SURROGATE


def test_the_schema_pattern_uses_python_whitespace_under_jsonschema() -> None:
    # Why the six differ: jsonschema evaluates "pattern" with Python's re, where \S excludes U+001F.
    assert REQUEST_VALIDATOR.schema["properties"]["message"]["pattern"] == "\\S"
    assert schema_code({"message": "\u001f"}) == "empty_input"
    assert schema_code({"message": "﻿"}) is None


# ----------------------------------------------------- robustness (fuzz)

FUZZ_SEED = 20261107
JSON_ALPHABET = b'{}[]":,\\ \t\r\nmessagetrufalsn0123456789.eE+-u\xef\xbb\xbf\xc0\xed\xa0\x80\xff'


def valid_bodies() -> list[bytes]:
    return [body_for(m) for m in PRESERVED_MESSAGES] + [VALID_BODY, padded_body(MAX_BODY_BYTES)]


def mutate(rng: random.Random, body: bytes) -> bytes:
    data = bytearray(body)
    for _ in range(rng.randint(1, 4)):
        operation = rng.randrange(7)
        position = rng.randint(0, len(data))
        if operation == 0 and data:
            data[min(position, len(data) - 1)] ^= 1 << rng.randrange(8)
        elif operation == 1:
            data[position:position] = bytes([rng.randrange(256)])
        elif operation == 2 and data:
            del data[min(position, len(data) - 1)]
        elif operation == 3:
            del data[position:]
        elif operation == 4:
            data[position:position] = data[position : position + rng.randint(1, 40)]
        elif operation == 5:
            data[position:position] = rng.choice(
                [b'"', b"\\", b"\\ud800", b"\\u", b"{", b"}", b"[", b"]", b",", b":", b"NaN", b'"message":', b"\xef\xbb\xbf"]
            )
        else:
            data[position:position] = bytes(rng.choice(JSON_ALPHABET) for _ in range(rng.randint(1, 8)))
    return bytes(data)


def fuzz_bodies(count: int) -> Iterator[bytes]:
    rng = random.Random(FUZZ_SEED)
    bases = valid_bodies()
    for index in range(count):
        kind = index % 3
        if kind == 0:
            yield rng.randbytes(rng.choice([0, 1, 2, 8, 32, 200, 1000, MAX_BODY_BYTES, MAX_BODY_BYTES + 1]))
        elif kind == 1:
            yield bytes(rng.choice(JSON_ALPHABET) for _ in range(rng.randint(0, 120)))
        else:
            yield mutate(rng, rng.choice(bases))


def assert_consistent(body: bytes, result: ChatRequest | RequestRejected) -> None:
    if isinstance(result, ChatRequest):
        assert json.loads(body.decode("utf-8")) == {"message": result.message}
        assert len(body) <= MAX_BODY_BYTES and not body.startswith(b"\xef\xbb\xbf")
        result.message.encode("utf-8")  # well formed: no unpaired surrogate
        assert len(result.message) <= LIMIT
        assert any(c not in ECMA_262_WHITESPACE for c in result.message)
    else:
        assert isinstance(result, RequestRejected)
        outcome = REJECTION_OUTCOMES[result.reason]
        assert (result.http_status, result.error_code) == (outcome.http_status, outcome.error_code)


def test_random_and_mutated_bodies_never_raise() -> None:
    seen: set[RejectionReason | None] = set()
    for body in fuzz_bodies(6000):
        result = check(body, content_length=None)
        assert_consistent(body, result)
        seen.add(result.reason if isinstance(result, RequestRejected) else None)
    # The corpus reaches the accepted path and most rejection reasons.
    assert None in seen
    assert {R.BODY_TOO_LARGE, R.INVALID_UTF8, R.INVALID_JSON, R.UNPAIRED_SURROGATE} <= seen


def test_random_transport_metadata_never_raises() -> None:
    rng = random.Random(FUZZ_SEED + 1)
    pieces = ["application", "/", "json", ";", " ", "\t", "charset", "=", "utf-8", '"', "identity", "gzip", ",", "ſ", "\x00", "POST"]
    for _ in range(3000):
        def field() -> str | None:
            return None if rng.random() < 0.2 else "".join(rng.choice(pieces) for _ in range(rng.randint(0, 7)))

        result = check(
            method=rng.choice(["POST", "GET", "post", "", "POST\x00"]),
            query_string=rng.choice(["", "", "", "a", "?"]),
            content_type=field(),
            content_encoding=field(),
            content_length=rng.choice([None, 0, len(VALID_BODY), MAX_BODY_BYTES, MAX_BODY_BYTES + 1]),
        )
        assert_consistent(VALID_BODY, result)


# ------------------------------------------------- no visitor content

SENTINEL = "zq7Sentinel"


def sentinel_requests() -> Iterator[tuple[str, dict[str, Any]]]:
    s = SENTINEL
    yield "method", {"method": s}
    yield "query", {"query_string": f"message={s}"}
    yield "content-type", {"content_type": f"application/json; {s}=1"}
    yield "content-encoding", {"content_encoding": s}
    yield "oversized", {"body": body_for(s * 900), "content_length": None}
    yield "bom", {"body": b"\xef\xbb\xbf" + body_for(s)}
    yield "utf8", {"body": body_for(s) + b"\xff"}
    yield "json", {"body": body_for(s)[:-1]}
    yield "duplicate", {"body": f'{{"message": "{s}", "message": "{s}"}}'.encode()}
    yield "constant", {"body": f'{{"message": "{s}", "x": NaN}}'.encode()}
    yield "nesting", {"body": b"[" * 4000 + s.encode() + b"]" * 4000}
    yield "not-object", {"body": json.dumps([s]).encode()}
    yield "extra", {"body": json.dumps({"message": s, s: s}).encode()}
    # A missing or blank message cannot carry visible text; {s: s} is an unexpected member (step 7 order).
    yield "type", {"body": json.dumps({"message": [s]}).encode()}
    yield "surrogate", {"body": f'{{"message": "{s}\\ud800"}}'.encode()}
    yield "too-long", {"body": body_for(s * 100)}


def test_each_sentinel_request_fails_a_different_check() -> None:
    reasons = {name: reason(check(**overrides)) for name, overrides in sentinel_requests()}
    assert len(set(reasons.values())) == len(reasons)


@pytest.mark.parametrize(("name", "overrides"), list(sentinel_requests()), ids=[n for n, _ in sentinel_requests()])
def test_a_rejection_contains_no_visitor_content(name: str, overrides: dict[str, Any]) -> None:
    result = check(**overrides)
    assert isinstance(result, RequestRejected)
    assert [field.name for field in dataclasses.fields(result)] == ["reason", "http_status", "error_code"]
    assert type(result.reason) is RejectionReason and type(result.error_code) is RequestErrorCode
    assert type(result.http_status) is int
    for text in (repr(result), str(result), result.reason.value, result.error_code.value, repr(dataclasses.astuple(result))):
        assert SENTINEL.lower() not in text.lower(), name
    assert not hasattr(result, "__dict__")  # slots: nothing else can be attached


def test_argument_errors_contain_no_request_content() -> None:
    with pytest.raises(TypeError) as raised:
        check(body=SENTINEL)
    assert SENTINEL not in str(raised.value)
    with pytest.raises(TypeError) as raised:
        check(method=SENTINEL.encode())
    assert SENTINEL not in str(raised.value)


# ------------------------------------------------ trusted configuration


@pytest.mark.parametrize(
    ("max_characters", "error"),
    [(0, ValueError), (-1, ValueError), (True, TypeError), (False, TypeError), (1.0, TypeError), ("1000", TypeError), (None, TypeError)],
)
def test_an_invalid_limit_fails_explicitly(max_characters: object, error: type[Exception]) -> None:
    with pytest.raises(error, match="max_characters"):
        check(max_characters=max_characters)
    with pytest.raises(error, match="max_characters"):
        check(method="GET", max_characters=max_characters)  # before any request check


@pytest.mark.parametrize(
    ("overrides", "error", "name"),
    [
        ({"method": b"POST"}, TypeError, "method"),
        ({"method": None}, TypeError, "method"),
        ({"query_string": None}, TypeError, "query_string"),
        ({"query_string": b""}, TypeError, "query_string"),
        ({"content_type": b"application/json"}, TypeError, "content_type"),
        ({"content_encoding": b"identity"}, TypeError, "content_encoding"),
        ({"content_length": "10"}, TypeError, "content_length"),
        ({"content_length": 10.0}, TypeError, "content_length"),
        ({"content_length": True}, TypeError, "content_length"),
        ({"content_length": -1}, ValueError, "content_length"),
        ({"body": "{}"}, TypeError, "body"),
        ({"body": bytearray(VALID_BODY)}, TypeError, "body"),
        ({"body": memoryview(VALID_BODY)}, TypeError, "body"),
        ({"body": None}, TypeError, "body"),
    ],
)
def test_caller_defects_raise(overrides: dict[str, Any], error: type[Exception], name: str) -> None:
    with pytest.raises(error, match=name):
        check(**overrides)


def test_the_arguments_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        check_chat_request("POST", "", "application/json", None, None, VALID_BODY, LIMIT)  # type: ignore[misc]


def test_the_value_objects_are_immutable() -> None:
    request = ChatRequest("a")
    rejected = RequestRejected(R.QUERY_STRING, 400, RequestErrorCode.INVALID_REQUEST)
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.message = "b"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        rejected.http_status = 200  # type: ignore[misc]


@pytest.mark.parametrize(
    ("status", "code"),
    [(200, RequestErrorCode.INVALID_REQUEST), (405, RequestErrorCode.INVALID_REQUEST), (400, RequestErrorCode.EMPTY_INPUT)],
)
def test_a_rejection_must_match_its_reason(status: int, code: RequestErrorCode) -> None:
    with pytest.raises(ValueError):
        RequestRejected(R.QUERY_STRING, status, code)


def test_the_result_is_deterministic() -> None:
    cases = [overrides for _, overrides, _ in STEP_CASES] + [{}]
    first = [check(**overrides) for overrides in cases]
    assert [check(**overrides) for overrides in cases] == first


# --------------------------------------------- dependencies and isolation

ALLOWED_IMPORTS = {"collections.abc", "dataclasses", "enum", "json", "re", "types", "typing"}


def test_the_module_imports_only_the_standard_library_it_needs() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module is not None
            imported.add(node.module)
    assert imported <= ALLOWED_IMPORTS, imported
    calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not calls & {"__import__", "open", "exec", "eval", "print"}


def test_the_module_does_not_use_python_whitespace_or_normalization() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not attributes & {"isspace", "casefold", "normalize", "upper", "splitlines"}
    for value in vars(origin_request).values():
        if hasattr(value, "pattern") and isinstance(value.pattern, str):
            assert "\\s" not in value.pattern and "\\S" not in value.pattern


FORBIDDEN_MODULES = [
    "mascot_lab.engine", "mascot_lab.ollama", "mascot_lab.router", "mascot_lab.public_response",
    "mascot_lab.public_preview", "mascot_lab.output", "mascot_lab.chat_config", "mascot_lab.runner",
    "socket", "ssl", "selectors", "asyncio", "http", "http.client", "http.server", "urllib.request",
    "socketserver", "wsgiref", "logging", "subprocess",
    "starlette", "fastapi", "uvicorn", "h11", "httpx", "requests", "aiohttp", "flask", "ollama",
]

ISOLATION_SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
import mascot_lab
before = set(sys.modules)
import mascot_lab.origin_request
added = sorted(set(sys.modules) - before)
print(json.dumps({"file": mascot_lab.origin_request.__file__, "added": added, "loaded": sorted(sys.modules)}))
"""


def test_importing_the_module_loads_no_engine_model_or_network_code() -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-c", ISOLATION_SCRIPT, str(SRC)],
        capture_output=True, text=True, timeout=60, check=True,
    )
    report = json.loads(completed.stdout)
    assert Path(report["file"]).resolve() == MODULE_PATH.resolve()
    assert [name for name in report["added"] if name.startswith("mascot_lab")] == ["mascot_lab.origin_request"]
    assert not set(report["loaded"]) & set(FORBIDDEN_MODULES)


# ------------------------------------------------------------- no network

_AUDITING = False
_EVENTS: list[str] = []


def _audit(event: str, args: tuple[object, ...]) -> None:
    if _AUDITING and (event.startswith(("socket.", "subprocess.", "os.system", "os.exec", "os.posix_spawn")) or event == "open"):
        _EVENTS.append(event)


sys.addaudithook(_audit)  # inert unless _AUDITING is set


@contextmanager
def audited() -> Iterator[list[str]]:
    global _AUDITING
    _EVENTS.clear()
    _AUDITING = True
    try:
        yield _EVENTS
    finally:
        _AUDITING = False


def test_validation_performs_no_network_or_file_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("check_chat_request used the network")

    for name in ("socket", "create_connection", "create_server", "getaddrinfo", "socketpair", "fromfd"):
        monkeypatch.setattr(socket, name, refuse)
    requests: list[Callable[[], ChatRequest | RequestRejected]] = [
        lambda overrides=overrides: check(**overrides) for _, overrides, _ in STEP_CASES + PRECEDENCE_CASES
    ]
    requests += [lambda body=body: check(body, content_length=None) for body in fuzz_bodies(600)]
    requests += [lambda: check()]
    with audited() as events:
        for request in requests:
            request()
    assert events == []
