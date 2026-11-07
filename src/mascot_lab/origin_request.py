"""Validation of a chat request at the inference origin, before the engine sees it.

The public chat API v1 (rastinaghighi.dev, ``docs/api/chat-v1.md``, PROPOSED)
requires the origin to repeat the gateway's request checks (section 6.2, O2).
``check_chat_request`` applies them to the request's transport metadata and
raw body bytes and returns either a ``ChatRequest`` holding the message,
exactly as decoded, or a ``RequestRejected`` naming the reason, the HTTP
status and the public error code. It is pure and deterministic: it opens no
socket, reads no file, logs nothing, and imports neither the engine, the
Ollama adapter nor any HTTP framework. Nothing in the engine or the command
line uses it; the HTTP adapter, which does not exist yet, will call it after
authenticating the gateway (O1) and before running the engine.

The checks run in this order, and the first failure wins:

==== ================================================ ==== =================
Step Check                                            HTTP ``error.code``
==== ================================================ ==== =================
1    The method is ``POST``                           405  invalid_request
2    The query string is empty                        400  invalid_request
3    ``Content-Type`` and ``Content-Encoding``        415  invalid_request
4    Declared and actual body size <= 8,192 bytes     413  invalid_request
5    Strict UTF-8, no leading byte-order mark         400  invalid_request
6    Strict JSON (RFC 8259)                           400  invalid_request
7    Exactly one member, ``message``, a string        400  invalid_request
8    No unpaired surrogate in ``message``             400  invalid_request
9    At least one ECMA-262 non-whitespace character   400  empty_input
10   At most ``max_characters`` code points           400  input_too_long
==== ================================================ ==== =================

These are the gateway's steps 1, 2 and 4 to 11 (section 6.1). The gateway's
step 3 (the ``Origin`` header) and step 12 (admission control) are not
repeated at the origin. ``REJECTION_OUTCOMES`` lists the internal reasons.

Details of each step:

* Step 1: methods are case-sensitive (RFC 9110, section 9.1), so ``post`` is
  refused.
* Step 3: ``Content-Type`` must be ``application/json``, optionally followed
  by the single parameter ``charset=utf-8``; ``Content-Encoding`` must be
  absent or ``identity``. The media type, the parameter name, the charset
  value and the coding are compared without regard to ASCII case (RFC 9110,
  sections 8.3.1, 8.3.2 and 8.4.1). Only ASCII is folded, so no non-ASCII
  look-alike such as U+017F (long s) or U+0131 (dotless i) can match.
  Surrounding spaces and tabs of a field value are not part of it (RFC 9110,
  section 5.5) and are ignored, and optional spaces and tabs may surround
  the ``;`` (section 5.6.6). Everything else is refused: a quoted value such
  as ``charset="utf-8"``, spaces around ``=``, an empty parameter (a trailing
  ``;``), a repeated parameter, an empty ``Content-Encoding`` value and a
  list of codings. RFC 9110 treats a quoted value as equal to the token, so
  refusing it is stricter than HTTP; the contract does not say.
* Step 4: ``content_length`` is the ``Content-Length`` the transport parsed,
  or ``None`` when there was none. A declared length above the limit is
  refused even if the body is short, and the body itself is counted, never
  the header alone. Framing, that is reading the body and detecting a body
  that does not match its declared length, belongs to the transport: every
  check here applies to the bytes actually passed in.
* Steps 5 and 6: the bytes are decoded as strict UTF-8 before parsing, so
  UTF-16 and UTF-32 bodies, overlong forms, encoded surrogates (CESU-8) and
  code points above U+10FFFF are refused, and Python's own encoding
  detection for ``bytes`` never runs. The parser refuses duplicate member
  names at every level, ``NaN``, ``Infinity`` and ``-Infinity``, and
  nesting too deep for the parser (``RecursionError``). Numbers are never
  converted, so a huge numeric literal is valid JSON that step 7 refuses.
* Step 8: a parser accepts an escaped unpaired surrogate such as
  ``"\\ud800"``; the engine could not encode it for the model runtime.
* Step 9 uses ECMA-262 whitespace (``ECMA_262_WHITESPACE``), as JSON Schema's
  ``pattern`` is defined to, never ``str.isspace`` or Python's ``\\s``. They
  differ in six characters: U+001C to U+001F and U+0085 are whitespace only
  to Python, and U+FEFF only to ECMA-262. So a message of only U+001F is
  accepted here, and one of only U+FEFF is refused with ``empty_input``.
* Step 10 counts code points, as JSON Schema's ``maxLength`` does. The limit
  is the caller's ``ChatConfig.max_input_characters``; this module has no
  limit of its own.

The accepted message is returned unchanged: no Unicode normalization, case
folding, newline conversion or trimming. The engine normalizes it afterwards
and checks the limits again (O3); this module does not repeat that.

A rejection carries only the reason, the HTTP status and the public error
code: never the message, the body, a header or any other part of the
request. Arguments of the wrong type, and a negative ``content_length`` or a
``max_characters`` below 1, are defects of the caller or its configuration
and raise ``TypeError`` or ``ValueError`` before any check runs. Malformed
request content never raises.

Not done here (later tasks): authentication (O1), the request ID (O2,
section 11), public response bodies and texts, streaming reads and framing,
deadlines and concurrency.
"""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import NamedTuple

# Request body limit in bytes (contract section 5, PROPOSED, API-06).
MAX_BODY_BYTES = 8192

# ECMA-262 WhiteSpace and LineTerminator, which ECMAScript's \s matches and
# JSON Schema's "pattern" therefore treats as whitespace: TAB, VT, FF, ZWNBSP
# (U+FEFF), the space separators (category Zs, unchanged since Unicode 6.3),
# LF, CR, LS and PS.
ECMA_262_WHITESPACE: frozenset[str] = frozenset(
    "\u0009\u000b\u000c﻿"  # TAB, VT, FF, ZWNBSP
    "          "  # Zs
    "      　"  # Zs
    "\u000a\u000d  "  # LF, CR, LS, PS
)

_UTF8_BOM = b"\xef\xbb\xbf"
_FIELD_WHITESPACE = " \t"  # HTTP's SP and HTAB
_CONTENT_TYPE = re.compile(r"application/json(?:[ \t]*;[ \t]*charset=utf-8)?", re.ASCII | re.IGNORECASE)
_SURROGATE = re.compile("[\ud800-\udfff]")


class RequestErrorCode(StrEnum):
    """The public ``error.code`` of a rejected request (contract section 9)."""

    INVALID_REQUEST = "invalid_request"
    EMPTY_INPUT = "empty_input"
    INPUT_TOO_LONG = "input_too_long"


class RejectionReason(StrEnum):
    """Why a request was rejected; internal, never part of the public body."""

    METHOD_NOT_ALLOWED = "method_not_allowed"
    QUERY_STRING = "query_string"
    UNSUPPORTED_CONTENT_TYPE = "unsupported_content_type"
    UNSUPPORTED_CONTENT_ENCODING = "unsupported_content_encoding"
    DECLARED_BODY_TOO_LARGE = "declared_body_too_large"
    BODY_TOO_LARGE = "body_too_large"
    BYTE_ORDER_MARK = "byte_order_mark"
    INVALID_UTF8 = "invalid_utf8"
    INVALID_JSON = "invalid_json"
    DUPLICATE_MEMBER = "duplicate_member"
    NONSTANDARD_CONSTANT = "nonstandard_constant"
    NESTING_TOO_DEEP = "nesting_too_deep"
    NOT_AN_OBJECT = "not_an_object"
    UNEXPECTED_MEMBER = "unexpected_member"
    MISSING_MESSAGE = "missing_message"
    MESSAGE_NOT_A_STRING = "message_not_a_string"
    UNPAIRED_SURROGATE = "unpaired_surrogate"
    EMPTY_MESSAGE = "empty_message"
    MESSAGE_TOO_LONG = "message_too_long"


class RejectionOutcome(NamedTuple):
    step: int
    http_status: int
    error_code: RequestErrorCode


_INVALID = RequestErrorCode.INVALID_REQUEST
_R = RejectionReason

# Every reason, with its validation step, HTTP status and public error code.
REJECTION_OUTCOMES: Mapping[RejectionReason, RejectionOutcome] = MappingProxyType(
    {
        _R.METHOD_NOT_ALLOWED: RejectionOutcome(1, 405, _INVALID),
        _R.QUERY_STRING: RejectionOutcome(2, 400, _INVALID),
        _R.UNSUPPORTED_CONTENT_TYPE: RejectionOutcome(3, 415, _INVALID),
        _R.UNSUPPORTED_CONTENT_ENCODING: RejectionOutcome(3, 415, _INVALID),
        _R.DECLARED_BODY_TOO_LARGE: RejectionOutcome(4, 413, _INVALID),
        _R.BODY_TOO_LARGE: RejectionOutcome(4, 413, _INVALID),
        _R.BYTE_ORDER_MARK: RejectionOutcome(5, 400, _INVALID),
        _R.INVALID_UTF8: RejectionOutcome(5, 400, _INVALID),
        _R.INVALID_JSON: RejectionOutcome(6, 400, _INVALID),
        _R.DUPLICATE_MEMBER: RejectionOutcome(6, 400, _INVALID),
        _R.NONSTANDARD_CONSTANT: RejectionOutcome(6, 400, _INVALID),
        _R.NESTING_TOO_DEEP: RejectionOutcome(6, 400, _INVALID),
        _R.NOT_AN_OBJECT: RejectionOutcome(7, 400, _INVALID),
        _R.UNEXPECTED_MEMBER: RejectionOutcome(7, 400, _INVALID),
        _R.MISSING_MESSAGE: RejectionOutcome(7, 400, _INVALID),
        _R.MESSAGE_NOT_A_STRING: RejectionOutcome(7, 400, _INVALID),
        _R.UNPAIRED_SURROGATE: RejectionOutcome(8, 400, _INVALID),
        _R.EMPTY_MESSAGE: RejectionOutcome(9, 400, RequestErrorCode.EMPTY_INPUT),
        _R.MESSAGE_TOO_LONG: RejectionOutcome(10, 400, RequestErrorCode.INPUT_TOO_LONG),
    }
)


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """An accepted request: the message exactly as decoded from the body."""

    message: str


@dataclass(frozen=True, slots=True)
class RequestRejected:
    """A rejected request. It holds no part of the request."""

    reason: RejectionReason
    http_status: int
    error_code: RequestErrorCode

    def __post_init__(self) -> None:
        outcome = REJECTION_OUTCOMES.get(self.reason)
        if outcome is None or (self.http_status, self.error_code) != outcome[1:]:
            raise ValueError("the HTTP status and error code do not match the rejection reason")


def check_chat_request(
    *,
    method: str,
    query_string: str,
    content_type: str | None,
    content_encoding: str | None,
    content_length: int | None,
    body: bytes,
    max_characters: int,
) -> ChatRequest | RequestRejected:
    """Validate one request; see the module documentation for the steps.

    ``max_characters`` is ``ChatConfig.max_input_characters``.
    """
    _check_arguments(method, query_string, content_type, content_encoding, content_length, body, max_characters)

    if method != "POST":
        return _reject(_R.METHOD_NOT_ALLOWED)
    if query_string:
        return _reject(_R.QUERY_STRING)
    if content_type is None or not _is_json_media_type(content_type):
        return _reject(_R.UNSUPPORTED_CONTENT_TYPE)
    if content_encoding is not None and not _is_identity(content_encoding):
        return _reject(_R.UNSUPPORTED_CONTENT_ENCODING)
    if content_length is not None and content_length > MAX_BODY_BYTES:
        return _reject(_R.DECLARED_BODY_TOO_LARGE)
    if len(body) > MAX_BODY_BYTES:
        return _reject(_R.BODY_TOO_LARGE)

    if body.startswith(_UTF8_BOM):
        return _reject(_R.BYTE_ORDER_MARK)
    try:
        text = body.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return _reject(_R.INVALID_UTF8)

    try:
        value = _strict_decoder().decode(text)
    except _StrictJsonError as exc:
        return _reject(exc.reason)
    except json.JSONDecodeError:
        return _reject(_R.INVALID_JSON)
    except RecursionError:
        return _reject(_R.NESTING_TOO_DEEP)

    if type(value) is not dict:
        return _reject(_R.NOT_AN_OBJECT)
    if any(key != "message" for key in value):
        return _reject(_R.UNEXPECTED_MEMBER)
    if "message" not in value:
        return _reject(_R.MISSING_MESSAGE)
    message = value["message"]
    if type(message) is not str:
        return _reject(_R.MESSAGE_NOT_A_STRING)

    if _SURROGATE.search(message):
        return _reject(_R.UNPAIRED_SURROGATE)
    if all(char in ECMA_262_WHITESPACE for char in message):
        return _reject(_R.EMPTY_MESSAGE)
    if len(message) > max_characters:
        return _reject(_R.MESSAGE_TOO_LONG)
    return ChatRequest(message)


def _reject(reason: RejectionReason) -> RequestRejected:
    outcome = REJECTION_OUTCOMES[reason]
    return RequestRejected(reason, outcome.http_status, outcome.error_code)


def _check_arguments(
    method: object,
    query_string: object,
    content_type: object,
    content_encoding: object,
    content_length: object,
    body: object,
    max_characters: object,
) -> None:
    # The messages name only the argument and its type, never a value from the request.
    for name, value in (("method", method), ("query_string", query_string)):
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a str, not {type(value).__name__}")
    for name, value in (("content_type", content_type), ("content_encoding", content_encoding)):
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{name} must be a str or None, not {type(value).__name__}")
    if content_length is not None:
        if type(content_length) is not int:
            raise TypeError(f"content_length must be an int or None, not {type(content_length).__name__}")
        if content_length < 0:
            raise ValueError("content_length must not be negative")
    if not isinstance(body, bytes):
        raise TypeError(f"body must be bytes, not {type(body).__name__}")
    if type(max_characters) is not int:
        raise TypeError(f"max_characters must be an int, not {type(max_characters).__name__}")
    if max_characters < 1:
        raise ValueError("max_characters must be at least 1")


def _field_value(value: str) -> str | None:
    """``value`` without surrounding SP and HTAB, or None unless it is ASCII."""
    return value.strip(_FIELD_WHITESPACE) if value.isascii() else None


def _is_json_media_type(content_type: str) -> bool:
    value = _field_value(content_type)
    return value is not None and _CONTENT_TYPE.fullmatch(value) is not None


def _is_identity(content_encoding: str) -> bool:
    value = _field_value(content_encoding)
    return value is not None and value.lower() == "identity"


class _StrictJsonError(Exception):
    def __init__(self, reason: RejectionReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


class _Number:
    """Stands in for every JSON number, which is never converted."""

    __slots__ = ()


_NUMBER = _Number()


def _unique_members(pairs: list[tuple[str, object]]) -> dict[str, object]:
    members = dict(pairs)
    if len(members) != len(pairs):
        raise _StrictJsonError(_R.DUPLICATE_MEMBER)
    return members


def _nonstandard_constant(name: str) -> object:
    raise _StrictJsonError(_R.NONSTANDARD_CONSTANT)


def _number(literal: str) -> _Number:
    return _NUMBER


def _strict_decoder() -> json.JSONDecoder:
    # A new decoder for every request, so no parser state is shared between requests.
    return json.JSONDecoder(
        object_pairs_hook=_unique_members,
        parse_constant=_nonstandard_constant,
        parse_int=_number,
        parse_float=_number,
        strict=True,  # no raw control characters inside strings
    )
