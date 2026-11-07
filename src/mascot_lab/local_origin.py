"""A local HTTP origin for the public chat API: ``python -m mascot_lab.local_origin``.

This is a local development prototype, not a public or production server.
It listens on the loopback interface only, at ``http://127.0.0.1:8765``,
and has no option to listen anywhere else. It serves:

* ``POST /api/chat/v1``, the public chat API v1 (rastinaghighi.dev,
  ``docs/api/chat-v1.md``, PROPOSED). ``check_chat_request``
  (``origin_request``) validates the request, ``ChatEngine.respond`` answers
  the accepted message with ``MODEL_TAG``, the provisional v1 selection in
  ``configs/selection.yaml``, and ``PublicResponseMapper.map``
  (``public_response``) turns the engine's response into the HTTP status and
  body. Every body has the seven members of the committed response schema
  and passes ``check_public_response``. The engine, its configuration, the
  prompts, the knowledge and the Ollama settings are used unchanged.
* ``GET /health``, which answers ``{"status": "ok"}`` without calling the
  engine or Ollama.

Every other path gets 404 with a small JSON body. These two paths are the
only ones this module serves; paths other than the chat API are outside the
contract.

Before anything else, on every path, the caller must name this server: one
``Host`` header, exactly ``127.0.0.1:8765`` or ``localhost:8765``
(``allowed_hosts``). An ``Origin`` header may be absent, as from PowerShell
and other local tools; if present, it must be one header and exactly one of
``allowed_origins``: this server's two origins and the Astro development
server's, ``http://127.0.0.1:4321`` and ``http://localhost:4321``. Values
are compared exactly, so ``null``, an empty value, a path, another scheme
or port and a different letter case are all refused. This stops DNS
rebinding and cross-origin browser requests. A refused request gets 400:
the chat body with ``invalid_request`` on the chat path, and
``{"error": "invalid_request"}`` elsewhere. No CORS header is ever sent.

The chat request is then handled in this order, and the first failure answers:

1. The validator's checks that need no body: the method (405, with
   ``Allow: POST``), the query string (400), ``Content-Type`` and
   ``Content-Encoding`` (415). The validator runs on an empty body here, and
   only rejections from these steps count.
2. Framing, which belongs to the transport: a ``Transfer-Encoding`` header,
   of any value, and a missing, repeated or malformed ``Content-Length`` are
   refused with 400. ``Content-Length`` must be ASCII digits only.
3. A declared length above 8,192 bytes is refused with 413 (the
   validator's step 4) before any of the body is read.
4. Exactly the declared number of bytes is read, never more, within
   ``request_timeout_s`` in total. A body that ends early or stalls is
   refused with 400.
5. The validator runs on the body. Its rejections give 400, 413 or 415 with
   ``invalid_request``, or 400 with ``empty_input`` or ``input_too_long``.
6. ``ChatEngine.respond`` and ``PublicResponseMapper.map``. An exception
   from the engine gives the mapper's fixed 500 response
   (``engine_exception``), never the exception.

Every response is JSON with ``Cache-Control: no-store``,
``X-Content-Type-Options: nosniff`` and ``Connection: close``: the server
answers one request per connection and then closes it, also after errors.
A request that the standard library cannot parse (a malformed request line,
an unsupported HTTP version, too many or too long header lines, HTTP/0.9)
gets the 400 ``invalid_request`` body.

Error texts are fixed: the engine's configured replies for ``empty_input``,
``input_too_long`` and ``unavailable``, and for ``invalid_request``
``INVALID_REQUEST_TEXT``, the contract's PROVISIONAL gateway text (section
17; wording open, API-07). No text is built from an exception, the request,
a path or a model diagnostic.

Configuration exceptions of this prototype, compared with the contract:

* There is no gateway. The contract has the gateway assign the request ID
  and the origin echo it (sections 11 and 18.2, R8). Here the origin assigns
  a random version 4 UUID (``uuid.uuid4``, from ``os.urandom``) to every
  request itself, rejected ones included, and ignores any request-ID header.
* There is no gateway authentication (O1), no rate limiting, no CORS (the
  Astro development server will proxy same-origin requests) and no TLS.
* The ``Host`` and ``Origin`` checks above take the place of the gateway's
  ``Origin`` check (section 6.1, step 3). They answer 400 instead of the
  gateway's 403, because the origin may send ``invalid_request`` only with
  400, 405, 413 or 415 (R7, ``check_public_response``).

The server is sequential (``socketserver.TCPServer``): it handles one
connection at a time, so at most one model request runs at once, and other
connections wait in the listen backlog. On Windows the port is bound with
``SO_EXCLUSIVEADDRUSE``, so a second server cannot share it.

Logging: one line per request on standard error, with the request ID, the
method and path when they are known ones, the HTTP status, the public status
and error code, the engine's internal status and error code or the
rejection reason, and the elapsed time. Visitor messages, request bodies,
query strings, headers, model output, knowledge, exception messages and
tracebacks are never logged.

Importing this module starts nothing; only ``main`` listens.
"""

import argparse
import http.server
import json
import os
import re
import socket
import socketserver
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from email.message import Message
from types import MappingProxyType
from typing import Any

from mascot_lab.engine import ChatEngine
from mascot_lab.errors import ConfigError, MascotLabError
from mascot_lab.origin_request import (
    MAX_BODY_BYTES,
    REJECTION_OUTCOMES,
    RequestErrorCode,
    RequestRejected,
    check_chat_request,
)
from mascot_lab.public_preview import public_preview
from mascot_lab.public_response import PublicResponse, PublicResponseMapper

PROGRAM = "python -m mascot_lab.local_origin"
HOST = "127.0.0.1"  # loopback only; there is deliberately no way to change it
PORT = 8765
DEV_SERVER_PORT = 4321  # the Astro development server, which proxies to this origin (strict port)
CHAT_PATH = "/api/chat/v1"
HEALTH_PATH = "/health"
# The provisional v1 selection (configs/selection.yaml, ``selected.tag``); a test checks that they agree.
MODEL_TAG = "phi4-mini:3.8b-q4_K_M"
# chat-v1.md, section 17: the PROVISIONAL gateway text for invalid_request (wording open, API-07).
INVALID_REQUEST_TEXT = "Sorry, something went wrong with that message. Please try again."
# Seconds for each socket read and write, and for reading the whole body.
REQUEST_TIMEOUT_S = 10.0

_LAST_HEADER_STEP = 4  # validator steps 1 to 4 need only the request line and headers
_CONTENT_LENGTH = re.compile(r"[0-9]+")
_MAX_LENGTH_DIGITS = len(str(MAX_BODY_BYTES))
_METHODS = ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT")
_HEADERS = MappingProxyType(
    {
        "Content-Type": "application/json; charset=utf-8",
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        "Connection": "close",
    }
)
# Statuses that each fixed error text is checked with at startup (contract section 9).
_ERROR_STATUSES = MappingProxyType(
    {
        RequestErrorCode.INVALID_REQUEST: (400, 405, 413, 415),
        RequestErrorCode.EMPTY_INPUT: (400,),
        RequestErrorCode.INPUT_TOO_LONG: (400,),
    }
)
_SAMPLE_REQUEST_ID = "00000000-0000-4000-8000-000000000000"


class LocalOriginServer(socketserver.TCPServer):
    """A sequential HTTP server on 127.0.0.1 for one ``ChatEngine``.

    ``port`` is 8765 when started with ``python -m``; tests pass 0 for a free
    port. There is no host parameter.
    """

    # On Windows, SO_REUSEADDR would let another socket bind the same port; see server_bind.
    allow_reuse_address = os.name != "nt"

    def __init__(self, engine: ChatEngine, *, port: int = PORT) -> None:
        if not isinstance(engine, ChatEngine):
            raise TypeError(f"engine must be a ChatEngine, not {type(engine).__name__}")
        if type(port) is not int or not 0 <= port <= 65535:
            raise ConfigError(f"the port must be an integer from 0 to 65535, found {port!r}")
        self.engine = engine
        self.model_tag = engine.model(MODEL_TAG).tag  # ConfigError unless it is a configured model
        self.mapper = PublicResponseMapper(engine.chat_config)
        self.request_timeout_s = REQUEST_TIMEOUT_S
        replies = engine.chat_config.replies
        self._error_texts: Mapping[RequestErrorCode, str] = MappingProxyType(
            {
                RequestErrorCode.INVALID_REQUEST: INVALID_REQUEST_TEXT,
                RequestErrorCode.EMPTY_INPUT: replies.empty_input,
                RequestErrorCode.INPUT_TOO_LONG: replies.input_too_long,
            }
        )
        for code, statuses in _ERROR_STATUSES.items():  # fail at startup, not on a visitor's request
            for http_status in statuses:
                try:
                    self.error_response(_SAMPLE_REQUEST_ID, http_status, code)
                except ValueError as exc:
                    raise ConfigError(f"the {code.value} text cannot be a public error text") from exc
        super().__init__((HOST, port), _Handler)
        bound_port = self.server_address[1]  # 8765, or the free port a test asked for
        self.allowed_hosts = allowed_hosts(bound_port)
        self.allowed_origins = allowed_origins(bound_port)

    def server_bind(self) -> None:
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)  # Windows only
        if exclusive is not None:
            self.socket.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        super().server_bind()
        if self.socket.getsockname()[0] != HOST:  # defense in depth
            raise OSError(f"the server is not bound to {HOST}")

    def error_response(self, request_id: str, http_status: int, code: RequestErrorCode) -> PublicResponse:
        """The ``error`` response with ``code`` and its fixed text; ``ValueError`` if it would not conform."""
        text = self._error_texts[code]
        preview = public_preview(text, output_limit_reached=False, chat=self.engine.chat_config)
        body = {
            "request_id": request_id,
            "status": "error",
            "answer_text": text,
            "cloud_preview": preview.cloud_preview,
            "continuation": preview.continuation_json(),
            "actions": [],
            "error": {"code": code.value},
        }
        return PublicResponse(http_status, body)

    def handle_error(self, request: Any, client_address: Any) -> None:
        # The default prints a traceback, and exception messages can quote request data.
        error = sys.exception()
        _log(f"a connection failed: {type(error).__name__}")


class _Handler(http.server.BaseHTTPRequestHandler):
    server: LocalOriginServer
    protocol_version = "HTTP/1.0"  # one request per connection, and never "100 Continue" before the checks

    def setup(self) -> None:
        self.timeout = self.server.request_timeout_s
        self._started = time.perf_counter()
        super().setup()

    def version_string(self) -> str:
        return "mascot-lab-local-origin"

    def log_message(self, format: str, *args: Any) -> None:
        pass  # the base class would log the request line, which can carry visitor text

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        """Answer a request the base class cannot parse with the 400 ``invalid_request`` body."""
        if self.request_version == "HTTP/0.9":  # the version was not understood: still send a status line
            self.request_version = self.protocol_version
        request_id = _new_request_id()
        response = self.server.error_response(request_id, 400, RequestErrorCode.INVALID_REQUEST)
        self._send_public(request_id, response, "unparsable request")

    def _dispatch(self) -> None:
        if self.request_version == "HTTP/0.9":
            self.send_error(400)
            return
        path, _, query = self.path.partition("?")
        refused = _caller_problem(self.headers, hosts=self.server.allowed_hosts, origins=self.server.allowed_origins)
        if refused is not None and path == CHAT_PATH:
            request_id = _new_request_id()
            response = self.server.error_response(request_id, 400, RequestErrorCode.INVALID_REQUEST)
            self._send_public(request_id, response, refused)
        elif refused is not None:
            self._send(400, {"error": "invalid_request"}, "-", refused)
        elif path == CHAT_PATH:
            self._chat(query)
        elif path == HEALTH_PATH:
            if self.command != "GET":
                self._send(405, {"error": "method_not_allowed"}, "-", "health", allow="GET")
            elif query:
                self._send(400, {"error": "invalid_request"}, "-", "health")
            else:
                self._send(200, {"status": "ok"}, "-", "health")
        else:
            self._send(404, {"error": "not_found"}, "-", "unknown path")

    # Methods without a handler get the base class's 501, which send_error turns into a 400.
    do_GET = do_HEAD = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_TRACE = do_CONNECT = _dispatch

    def _chat(self, query: str) -> None:
        request_id = _new_request_id()
        try:
            response, detail = self._answer(request_id, query)
        except Exception as exc:  # a defect here: the fixed 502 response, never a partial one
            response, detail = self.server.mapper.fail_closed(request_id), f"origin failure {type(exc).__name__}"
        self._send_public(request_id, response, detail)

    def _answer(self, request_id: str, query: str) -> tuple[PublicResponse, str]:
        server = self.server
        fields: dict[str, Any] = {
            "method": self.command,
            "query_string": query,
            "content_type": _field(self.headers, "Content-Type"),
            "content_encoding": _field(self.headers, "Content-Encoding"),
            "max_characters": server.engine.chat_config.max_input_characters,
        }
        rejected = _header_rejection(fields, content_length=None)
        if rejected is not None:
            return server.error_response(request_id, rejected.http_status, rejected.error_code), rejected.reason
        declared = _declared_length(self.headers)
        if isinstance(declared, str):
            return server.error_response(request_id, 400, RequestErrorCode.INVALID_REQUEST), declared
        rejected = _header_rejection(fields, content_length=declared)
        if rejected is not None:
            return server.error_response(request_id, rejected.http_status, rejected.error_code), rejected.reason
        body = self._read_body(declared)
        if body is None:
            return server.error_response(request_id, 400, RequestErrorCode.INVALID_REQUEST), "incomplete_body"

        checked = check_chat_request(**fields, content_length=declared, body=body)
        if isinstance(checked, RequestRejected):
            return server.error_response(request_id, checked.http_status, checked.error_code), checked.reason
        try:
            result = server.engine.respond(checked.message, server.model_tag)
        except Exception as exc:  # contract section 8.2, rule 3
            return server.mapper.engine_exception(request_id), f"engine raised {type(exc).__name__}"
        engine = result.status.value if result.error is None else f"{result.status.value}/{result.error.code.value}"
        return server.mapper.map(result, request_id=request_id), f"engine {engine}"

    def _read_body(self, length: int) -> bytes | None:
        """Exactly ``length`` bytes within the request timeout, or None if the body ends early or stalls."""
        if not 0 <= length <= MAX_BODY_BYTES:
            raise ValueError("the body length is not checked")
        deadline = time.monotonic() + self.server.request_timeout_s
        chunks: list[bytes] = []
        remaining = length
        try:
            while remaining:
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self.connection.settimeout(left)
                chunk = self.rfile.read1(remaining)  # never more than what is still missing
                if not chunk:
                    return None
                chunks.append(chunk)
                remaining -= len(chunk)
        except OSError:  # including TimeoutError and a reset connection
            return None
        finally:
            self.connection.settimeout(self.server.request_timeout_s)
        return b"".join(chunks)

    def _send_public(self, request_id: str, response: PublicResponse, detail: str) -> None:
        body = response.to_dict()
        outcome = body["status"] if body["error"] is None else f"{body['status']}/{body['error']['code']}"
        allow = "POST" if response.http_status == 405 else None
        self._send(response.http_status, body, request_id, f"{outcome}; {detail}", allow=allow)

    def _send(
        self, http_status: int, body: Mapping[str, Any], request_id: str, detail: str, *, allow: str | None = None
    ) -> None:
        data = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.close_connection = True
        try:
            self.send_response(http_status)
            for name, value in _HEADERS.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(data)))
            if allow is not None:
                self.send_header("Allow", allow)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)
        except OSError as exc:
            detail += f"; not delivered ({type(exc).__name__})"
        method = self.command if self.command in _METHODS else "-"
        path = getattr(self, "path", "").partition("?")[0]
        target = path if path in (CHAT_PATH, HEALTH_PATH) else "-"
        elapsed_ms = (time.perf_counter() - self._started) * 1000
        _log(f"{request_id} {method} {target} {http_status} {detail} ({elapsed_ms:.0f} ms)")


def _header_rejection(fields: Mapping[str, Any], *, content_length: int | None) -> RequestRejected | None:
    """The validator's rejection from the request line and headers alone (steps 1 to 4), if any.

    It runs on an empty body, so only steps that need no body can count;
    an empty body's own failure (step 6) is ignored.
    """
    result = check_chat_request(**fields, content_length=content_length, body=b"")
    if isinstance(result, RequestRejected) and REJECTION_OUTCOMES[result.reason].step <= _LAST_HEADER_STEP:
        return result
    return None


def allowed_hosts(port: int) -> frozenset[str]:
    """The ``Host`` values accepted by a server on ``port``."""
    return frozenset({f"127.0.0.1:{port}", f"localhost:{port}"})


def allowed_origins(port: int) -> frozenset[str]:
    """The ``Origin`` values accepted by a server on ``port``: its own and the development server's."""
    return frozenset(f"http://{host}" for host in allowed_hosts(port) | allowed_hosts(DEV_SERVER_PORT))


def _caller_problem(headers: Message, *, hosts: frozenset[str], origins: frozenset[str]) -> str | None:
    """Why the ``Host`` or ``Origin`` header is refused, or None (against DNS rebinding and cross-origin requests)."""
    values = headers.get_all("Host")
    if values is None:
        return "host_missing"
    if len(values) != 1:
        return "host_repeated"
    if values[0].strip(" \t") not in hosts:
        return "host_not_allowed"
    values = headers.get_all("Origin")
    if values is None:
        return None  # PowerShell and other local tools send none
    if len(values) != 1:
        return "origin_repeated"
    if values[0].strip(" \t") not in origins:
        return "origin_not_allowed"  # including null, an empty value and anything malformed
    return None


def _field(headers: Message, name: str) -> str | None:
    """A header's value, or None; repeated lines are combined with ", " (RFC 9110, section 5.3)."""
    values = headers.get_all(name)
    return None if values is None else ", ".join(values)


def _declared_length(headers: Message) -> int | str:
    """The declared body length, or why the framing is refused.

    A number with more digits than ``MAX_BODY_BYTES`` (after leading zeros)
    is larger than the limit and is returned as ``MAX_BODY_BYTES + 1``,
    which the validator refuses with 413, without converting it.
    """
    if headers.get_all("Transfer-Encoding") is not None:
        return "transfer_encoding"
    values = headers.get_all("Content-Length")
    if values is None:
        return "content_length_missing"
    if len(values) != 1:
        return "content_length_repeated"
    value = values[0].strip(" \t")
    if _CONTENT_LENGTH.fullmatch(value) is None:
        return "content_length_malformed"
    digits = value.lstrip("0") or "0"
    return int(digits) if len(digits) <= _MAX_LENGTH_DIGITS else MAX_BODY_BYTES + 1


def _new_request_id() -> str:
    return str(uuid.uuid4())  # lowercase canonical version 4, from os.urandom


def _log(line: str) -> None:
    print(f"[local_origin {time.strftime('%H:%M:%S')}] {line}", file=sys.stderr, flush=True)


def create_server(engine: ChatEngine | None = None, *, port: int = PORT) -> LocalOriginServer:
    """Bind the server to 127.0.0.1; the engine is built from the project files unless given."""
    return LocalOriginServer(engine if engine is not None else ChatEngine.from_files(), port=port)


def main(argv: Sequence[str] | None = None) -> int:
    """Serve until interrupted (Ctrl+C).

    Exit status: 0 when stopped, 1 if the port is unavailable, 2 for invalid configuration or arguments.
    """
    argparse.ArgumentParser(
        prog=PROGRAM,
        description=f"Serve the chat API on http://{HOST}:{PORT}{CHAT_PATH} for local development only. "
        "There are no options: the address, the model and every setting come from the project files.",
    ).parse_args(argv)
    try:
        server = create_server()
    except MascotLabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: cannot listen on {HOST}:{PORT}: {exc.strerror or type(exc).__name__}", file=sys.stderr)
        return 1
    with server:
        _log(
            f"listening on http://{HOST}:{PORT} (POST {CHAT_PATH}, GET {HEALTH_PATH}); model {server.model_tag}; "
            f"Ollama timeout {server.engine.client.timeout_s:g} s; Ctrl+C stops"
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            _log("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
