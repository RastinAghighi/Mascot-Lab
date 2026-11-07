"""The public chat API response: an engine result mapped to an HTTP status and body.

The public chat API v1 (rastinaghighi.dev, ``docs/api/chat-v1.md``) gives
every response the same seven members: ``request_id``, ``status``,
``answer_text``, ``cloud_preview``, ``continuation``, ``actions`` and
``error``. ``PublicResponseMapper.map`` turns one ``ChatResponse`` into that
body and the HTTP status that goes with it. It is pure: it opens no socket,
serves nothing, generates no identifier and leaves the engine's response
unchanged. The HTTP adapter, which does not exist yet, will authenticate the
gateway, validate the request, run the engine, call ``map`` and serialize the
result. Nothing in the engine or the command line uses this module.

The mapping is closed (contract sections 8.2 and 9). Every other engine
outcome or error code, including one added by a later engine change, gives
the fixed 502 ``unavailable`` response:

======================== ===================== ============ ====== ===============
Engine status            Engine error code     Public       HTTP   ``error.code``
======================== ===================== ============ ====== ===============
answered                 (none)                answered     200    (null)
routed                   (none)                navigation   200    (null)
ambiguous_destination    (none)                navigation   200    (null)
unknown_destination      (none)                refused      200    (null)
url_rejected             (none)                refused      200    (null)
withheld                 (none)                refused      200    (null)
invalid_input            empty_input           error        400    empty_input
invalid_input            input_too_long        error        400    input_too_long
error                    knowledge_unavailable unavailable  503    unavailable
error                    model_not_found       unavailable  503    unavailable
error                    ollama_unavailable    unavailable  503    unavailable
error                    timeout               unavailable  504    timeout
error                    ollama_error          unavailable  502    unavailable
error                    invalid_response      unavailable  502    unavailable
error                    empty_output          unavailable  502    unavailable
======================== ===================== ============ ====== ===============

The body is built only from allowlisted values, never from the engine's
record: for ``answered`` the validated display text, never the raw model
output; otherwise the configured reply for the outcome, which ``map`` checks
the engine's text against (R6). The preview and continuation come from
``public_preview``, never from the engine's own preview. ``output_limit_reached``
is passed on only for ``answered``, where the engine's continuation, its
output issues and Ollama's ``done_reason`` must agree; any other outcome that
claims it fails closed. The request ID is the gateway's, validated and copied
unchanged (R8). Every body is then checked (``check_public_response``)
against the response schema and the runtime checks R1, R4, R5 and R7, and any
failure gives the fixed 502 response instead (R10). Its text is the
configured ``replies.unavailable``; this module defines no wording of its own,
and the gateway's texts are separate (contract section 17; API-07 is open).

Gaps that this module does not close (contract section 18.2), so the origin
is not ready to serve visitors:

* R1. Output validation keeps unpaired surrogates (category Cs) that a model
  reply can contain. This module refuses them with the 502 response and never
  repairs them. A visitor message with one makes the engine raise while it
  encodes the Ollama request, so the adapter must reject it first (O2).
* R2. Output validation keeps single-underscore emphasis, strikethrough,
  Markdown tables and tag-like text whose name contains an underscore. This
  module refuses only control, format, separator and surrogate code points
  (Cc, Cf, Zl, Zp, Cs); it is not a sanitizer.
* R3. Output validation withholds only verbatim copies of the instruction. It
  keeps the facts' bracketed source tags and entry codes, the knowledge
  delimiter tags and the template placeholder. This module detects none of
  them.
* There is no configured reply for ``invalid_request`` (HTTP 400, 405, 413
  and 415), which the adapter will need, and what the origin returns for a
  malformed request ID is open (API-02): ``map`` raises ``RequestIdError``
  and builds no body.
"""

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from mascot_lab.chat_config import ChatConfig
from mascot_lab.engine import ActionType, ChatResponse, ChatStatus, ErrorCode, NavigationAction, join_alternatives
from mascot_lab.errors import ConfigError
from mascot_lab.output import ELLIPSIS, OutputIssue
from mascot_lab.public_preview import (
    PUBLIC_CONTINUATION_ROUTE,
    PUBLIC_PREVIEW_MAX_CHARACTERS,
    public_preview,
    reached_output_limit,
)

# The response schema's members, statuses, action routes and bounds (chat-response-v1.schema.json).
PUBLIC_FIELDS = ("request_id", "status", "answer_text", "cloud_preview", "continuation", "actions", "error")
PUBLIC_STATUSES = ("answered", "navigation", "refused", "error", "unavailable")
PUBLIC_ACTION_ROUTES = ("project_qmind", "project_ingenuity", "project_satira", "project_deeplense", "project_website")
MAX_ANSWER_CHARACTERS = 4000
MAX_FIXED_TEXT_CHARACTERS = PUBLIC_PREVIEW_MAX_CHARACTERS  # error and unavailable texts fit the cloud

_REQUEST_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
_REASON_LISTS = (["exceeds_cloud"], ["output_limit_reached"], ["exceeds_cloud", "output_limit_reached"])
_ACTION_TYPES = MappingProxyType({ActionType.NAVIGATE: "navigate", ActionType.SUGGEST: "suggest"})
_FAILURE_CODES = MappingProxyType(
    {
        "error": ("invalid_request", "empty_input", "input_too_long"),
        "unavailable": ("rate_limited", "timeout", "unavailable"),
    }
)
# R7: the HTTP statuses the origin may send with each status and error code (contract section 9).
_HTTP_STATUSES = MappingProxyType(
    {
        ("answered", None): (200,),
        ("navigation", None): (200,),
        ("refused", None): (200,),
        ("error", "invalid_request"): (400, 405, 413, 415),
        ("error", "empty_input"): (400,),
        ("error", "input_too_long"): (400,),
        ("unavailable", "rate_limited"): (429,),
        ("unavailable", "unavailable"): (500, 502, 503),
        ("unavailable", "timeout"): (504,),
    }
)
# Never published: controls, format characters, line and paragraph separators, and surrogates.
# This covers every character the schema's display-text pattern rejects (R1, R2).
_UNPUBLISHABLE = frozenset({"Cc", "Cf", "Zl", "Zp", "Cs"})


@dataclass(frozen=True, slots=True)
class _Outcome:
    status: str
    http_status: int
    error_code: str | None
    reply: str | None  # the ChatConfig.replies field that is the text; None: the validated answer


_OUTCOMES: Mapping[tuple[ChatStatus, ErrorCode | None], _Outcome] = MappingProxyType(
    {
        (ChatStatus.ANSWERED, None): _Outcome("answered", 200, None, None),
        (ChatStatus.ROUTED, None): _Outcome("navigation", 200, None, "routed"),
        (ChatStatus.AMBIGUOUS_DESTINATION, None): _Outcome("navigation", 200, None, "ambiguous_destination"),
        (ChatStatus.UNKNOWN_DESTINATION, None): _Outcome("refused", 200, None, "unknown_destination"),
        (ChatStatus.URL_REJECTED, None): _Outcome("refused", 200, None, "url_rejected"),
        (ChatStatus.WITHHELD, None): _Outcome("refused", 200, None, "withheld"),
        (ChatStatus.INVALID_INPUT, ErrorCode.EMPTY_INPUT): _Outcome("error", 400, "empty_input", "empty_input"),
        (ChatStatus.INVALID_INPUT, ErrorCode.INPUT_TOO_LONG): _Outcome("error", 400, "input_too_long", "input_too_long"),
        # 503: the knowledge block, the configured model or the model runtime is not available at the origin.
        (ChatStatus.ERROR, ErrorCode.KNOWLEDGE_UNAVAILABLE): _Outcome("unavailable", 503, "unavailable", "unavailable"),
        (ChatStatus.ERROR, ErrorCode.MODEL_NOT_FOUND): _Outcome("unavailable", 503, "unavailable", "unavailable"),
        (ChatStatus.ERROR, ErrorCode.OLLAMA_UNAVAILABLE): _Outcome("unavailable", 503, "unavailable", "unavailable"),
        (ChatStatus.ERROR, ErrorCode.TIMEOUT): _Outcome("unavailable", 504, "timeout", "unavailable"),
        # 502: the model runtime failed, answered invalidly, or left no displayable text.
        (ChatStatus.ERROR, ErrorCode.OLLAMA_ERROR): _Outcome("unavailable", 502, "unavailable", "unavailable"),
        (ChatStatus.ERROR, ErrorCode.INVALID_RESPONSE): _Outcome("unavailable", 502, "unavailable", "unavailable"),
        (ChatStatus.ERROR, ErrorCode.EMPTY_OUTPUT): _Outcome("unavailable", 502, "unavailable", "unavailable"),
    }
)


class RequestIdError(ValueError):
    """The gateway's request ID is missing or malformed. The message never contains it."""


@dataclass(frozen=True, slots=True)
class PublicResponse:
    """One public chat API response: the HTTP status and the seven-member body.

    Construction checks both with ``check_public_response`` and raises
    ``ValueError`` on any failure, so every instance meets the response
    schema and the runtime checks R1, R4, R5 and R7. ``body`` is read-only;
    ``to_dict`` returns a new copy, ready for JSON.
    """

    http_status: int
    body: Mapping[str, Any]

    def __post_init__(self) -> None:
        plain = _plain(self.body)
        check_public_response(self.http_status, plain)
        object.__setattr__(self, "body", _read_only(plain))

    def to_dict(self) -> dict[str, Any]:
        return _plain(self.body)


class PublicResponseMapper:
    """Maps engine responses to public responses, with the engine's chat configuration."""

    def __init__(self, chat: ChatConfig) -> None:
        if not isinstance(chat, ChatConfig):
            raise TypeError(f"chat must be a ChatConfig, not {type(chat).__name__}")
        text = chat.replies.unavailable  # the fixed 502 response must always be valid
        if (
            _text_problem(text, MAX_FIXED_TEXT_CHARACTERS) is not None
            or public_preview(text, output_limit_reached=False, chat=chat).continuation is not None
        ):
            raise ConfigError(f"{chat.source}: replies.unavailable cannot be a public error text")
        self._chat = chat

    def map(self, response: ChatResponse, *, request_id: str) -> PublicResponse:
        """Return the public response to ``response``, for the gateway's ``request_id``.

        Raises ``RequestIdError``, and builds nothing, when ``request_id`` is
        malformed. Otherwise it never raises: an unmapped outcome, an
        inconsistent response or a body that fails a check gives
        ``fail_closed(request_id)``.
        """
        check_request_id(request_id)
        try:
            return self._map(response, request_id)
        except Exception:  # R10: never a partial, repaired or unchecked body
            return self.fail_closed(request_id)

    def fail_closed(self, request_id: str) -> PublicResponse:
        """The fixed 502 ``unavailable`` response (contract section 8.2, rule 2, and R10)."""
        return self._unavailable(request_id, 502)

    def engine_exception(self, request_id: str) -> PublicResponse:
        """The fixed 500 response for an engine that raised instead of responding (section 8.2, rule 3)."""
        return self._unavailable(request_id, 500)

    def _unavailable(self, request_id: str, http_status: int) -> PublicResponse:
        check_request_id(request_id)
        outcome = _Outcome("unavailable", http_status, "unavailable", "unavailable")
        return self._build(request_id, outcome, self._chat.replies.unavailable, [], False)

    def _map(self, response: ChatResponse, request_id: str) -> PublicResponse:
        if not isinstance(response, ChatResponse):
            raise TypeError("not a ChatResponse")
        code = None if response.error is None else response.error.code
        if type(response.status) is not ChatStatus or not (code is None or type(code) is ErrorCode):
            raise ValueError("unmapped engine outcome")
        outcome = _OUTCOMES.get((response.status, code))
        if outcome is None:
            raise ValueError("unmapped engine outcome")
        if response.configs.get("chat") != self._chat.sha256:
            raise ValueError("the response comes from another chat configuration")

        limit = reached_output_limit(response)
        if outcome.reply is None:
            _check_answer_metadata(response, limit)
            text = response.answer_text  # the validated display text, never the raw output
        else:
            if limit:
                raise ValueError("only an answered message can reach the output limit")
            text = self._configured_reply(outcome.reply, response.actions)
            if response.answer_text != text:
                raise ValueError("the text is not the configured reply for the outcome")
        actions = [_public_action(action) for action in response.actions]
        return self._build(request_id, outcome, text, actions, limit)

    def _configured_reply(self, reply: str, actions: tuple[NavigationAction, ...]) -> str:
        template: str = getattr(self._chat.replies, reply)
        labels = [self._chat.route_labels[action.route_id] for action in actions]
        if reply == "routed":
            (label,) = labels
            return template.replace("{label}", label)
        if reply == "ambiguous_destination":
            return template.replace("{labels}", join_alternatives(labels))
        return template

    def _build(
        self, request_id: str, outcome: _Outcome, text: str, actions: list[dict[str, str]], limit: bool
    ) -> PublicResponse:
        preview = public_preview(text, output_limit_reached=limit, chat=self._chat)
        body = {
            "request_id": request_id,
            "status": outcome.status,
            "answer_text": text,
            "cloud_preview": preview.cloud_preview,
            "continuation": preview.continuation_json(),
            "actions": actions,
            "error": None if outcome.error_code is None else {"code": outcome.error_code},
        }
        return PublicResponse(outcome.http_status, body)


def check_request_id(request_id: object) -> str:
    """Return ``request_id`` unchanged if it has the contract's form (section 11).

    That is a version 4 UUID in lowercase canonical form, matched exactly:
    no surrounding whitespace, braces, prefix or uppercase letters. Raises
    ``RequestIdError`` otherwise; no ID is ever generated or repaired.
    """
    if type(request_id) is not str or _REQUEST_ID.fullmatch(request_id) is None:
        raise RequestIdError("the request ID is not a lowercase canonical version 4 UUID")
    return request_id


def check_public_response(http_status: int, body: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` unless the origin may send ``body`` with ``http_status``.

    ``body`` holds plain JSON values (dict, list, str, int, None). The checks
    are the response schema's rules and the runtime checks that need only the
    response itself: R1 and the character rule above, R4, R5's per-status
    rules, and R7. The message names the failed rule, never a value.
    """
    problem = _problem(http_status, body)
    if problem is not None:
        raise ValueError(f"not a publishable chat response: {problem}")


def _check_answer_metadata(response: ChatResponse, limit: bool) -> None:
    """The engine's three reports of the output limit agree, and nothing was withheld."""
    issues = response.output_issues
    if OutputIssue.INSTRUCTION_LEAK in issues or OutputIssue.EMPTY_OUTPUT in issues:
        raise ValueError("an answered response with a withheld or empty output")
    stopped = response.output is not None and response.output.done_reason == "length"
    if limit != stopped or limit != (OutputIssue.OUTPUT_LIMIT_REACHED in issues):
        raise ValueError("the engine's output-limit reports disagree")


def _public_action(action: NavigationAction) -> dict[str, str]:
    if type(action) is not NavigationAction or type(action.type) is not ActionType:
        raise ValueError("not an engine navigation action")
    if action.route_id not in PUBLIC_ACTION_ROUTES:
        raise ValueError("the route is not in the public allowlist")
    return {"type": _ACTION_TYPES[action.type], "route_id": action.route_id}


def _problem(http_status: object, body: object) -> str | None:
    if type(body) is not dict or set(body) != set(PUBLIC_FIELDS):
        return "the body does not have exactly the seven members"
    if type(body["request_id"]) is not str or _REQUEST_ID.fullmatch(body["request_id"]) is None:
        return "request_id is not a lowercase canonical version 4 UUID"
    status = body["status"]
    if type(status) is not str or status not in PUBLIC_STATUSES:
        return "status is not a public status"
    failed = status in _FAILURE_CODES
    for name, limit in (
        ("answer_text", MAX_FIXED_TEXT_CHARACTERS if failed else MAX_ANSWER_CHARACTERS),
        ("cloud_preview", PUBLIC_PREVIEW_MAX_CHARACTERS),
    ):
        problem = _text_problem(body[name], limit)
        if problem is not None:
            return f"{name} {problem}"

    continuation, actions, error = body["continuation"], body["actions"], body["error"]
    reasons: list[str] = []
    if continuation is not None:
        if type(continuation) is not dict or set(continuation) != {"route_id", "reasons"}:
            return "continuation does not have exactly route_id and reasons"
        reasons = continuation["reasons"]
        if type(continuation["route_id"]) is not str or continuation["route_id"] != PUBLIC_CONTINUATION_ROUTE:
            return "continuation.route_id is not the continuation route"
        if type(reasons) is not list or any(type(item) is not str for item in reasons):
            return "continuation.reasons is not a list of strings"
        if reasons not in _REASON_LISTS:
            return "continuation.reasons is not an allowed list"

    if type(actions) is not list or len(actions) > len(PUBLIC_ACTION_ROUTES):
        return "actions is not a list of at most five actions"
    for action in actions:
        if type(action) is not dict or set(action) != {"type", "route_id"}:
            return "an action does not have exactly type and route_id"
        if type(action["type"]) is not str or action["type"] not in _ACTION_TYPES.values():
            return "an action type is not navigate or suggest"
        if type(action["route_id"]) is not str or action["route_id"] not in PUBLIC_ACTION_ROUTES:
            return "an action route is not allowlisted"

    code = None
    if error is not None:
        if type(error) is not dict or "code" not in error or not set(error) <= {"code", "retry_after_seconds"}:
            return "error does not have code and at most retry_after_seconds"
        code = error["code"]
        if type(code) is not str or not any(code in codes for codes in _FAILURE_CODES.values()):
            return "error.code is not a public error code"
        seconds = error.get("retry_after_seconds")
        if "retry_after_seconds" in error and (
            code != "rate_limited" or type(seconds) is not int or not 1 <= seconds <= 86400
        ):
            return "retry_after_seconds is allowed only with rate_limited, from 1 to 86400"

    if failed:
        if code not in _FAILURE_CODES[status]:
            return f"{status} needs one of its own error codes"
        if continuation is not None or actions:
            return f"{status} has a continuation or actions"
    else:
        if error is not None:
            return f"{status} has an error"
        kinds = [action["type"] for action in actions]
        distinct = len({action["route_id"] for action in actions}) == len(actions)
        if status == "navigation":
            if kinds != ["navigate"] and not (len(kinds) >= 2 and set(kinds) == {"suggest"} and distinct):
                return "navigation needs one navigate action or 2 to 5 suggest actions with distinct routes"
        elif actions:
            return f"{status} has actions"
        if status != "answered" and "output_limit_reached" in reasons:
            return f"{status} cannot reach the output limit"

    answer, preview = body["answer_text"], body["cloud_preview"]
    shortened = preview != answer  # R4, compared as exact code points
    if ("exceeds_cloud" in reasons) != shortened:
        return "exceeds_cloud is not given exactly when the preview is shortened"
    start = preview[: -len(ELLIPSIS)]
    exact_start = preview.endswith(ELLIPSIS) and 0 < len(start) < len(answer) and answer.startswith(start)
    if shortened and not exact_start:
        return "the shortened preview is not an exact start of answer_text followed by an ellipsis"

    if type(http_status) is not int or http_status not in _HTTP_STATUSES[(status, code)]:
        return "the HTTP status does not go with status and error.code"
    return None


def _text_problem(text: object, limit: int) -> str | None:
    if type(text) is not str:
        return "is not a string"
    if not 1 <= len(text) <= limit:
        return f"has {len(text)} characters, not 1 to {limit}"
    for char in text:
        if unicodedata.category(char) in _UNPUBLISHABLE:
            return f"contains U+{ord(char):04X}, a control, format, separator or surrogate code point"
    if not text.strip():
        return "has no visible character"
    return None


def _plain(value: Any) -> Any:
    """A new copy with plain dicts and lists, as JSON needs."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _read_only(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _read_only(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_read_only(item) for item in value)
    return value
