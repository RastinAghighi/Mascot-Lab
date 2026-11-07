"""The local HTTP origin, ``mascot_lab.local_origin``, over real loopback sockets.

Each test starts the server on 127.0.0.1 with a free port (``port=0``) in a
background thread. The engine is the real ``ChatEngine`` built from the
project files, with ``FakeOllama`` in place of the Ollama server, so no test
runs a model or contacts Ollama. Request validation itself is covered by
test_origin_request.py and the response mapping by test_public_response.py;
these tests cover the HTTP transport and its wiring to those parts.
"""

import http.client
import inspect
import json
import re
import socket
import socketserver
import subprocess
import sys
import threading
from collections.abc import Iterator
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator
from ollama_fakes import FakeOllama

from mascot_lab import local_origin
from mascot_lab.engine import ChatEngine
from mascot_lab.ollama import OllamaClient, OllamaError, OllamaErrorCode
from mascot_lab.paths import PROJECT_ROOT

SCHEMA = json.loads((PROJECT_ROOT / "tests" / "contracts" / "chat-response-v1.schema.json").read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=Draft202012Validator.FORMAT_CHECKER)
REQUEST_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
MESSAGE = "What does Rastin work on?"
ANSWER = "This is a scripted test answer."  # synthetic, not a fact
SECRET = r"C:\private\knowledge\facts.md token=sk-synthetic-secret"
JSON = "Content-Type: application/json"
OWN_HOST = "127.0.0.1:<port>"  # the test server's own Host value
NAVIGATION = b'{"message": "take me to Satira"}'


@pytest.fixture
def engine(ollama: FakeOllama) -> ChatEngine:
    return ChatEngine.from_files(client=OllamaClient(transport=ollama))


@pytest.fixture
def server(engine: ChatEngine) -> Iterator[local_origin.LocalOriginServer]:
    server = local_origin.create_server(engine, port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()


def send(server: local_origin.LocalOriginServer, method: str, path: str, body: bytes | None = None) -> tuple[int, dict[str, str], bytes]:
    """One request with the standard library's HTTP client."""
    connection = http.client.HTTPConnection(local_origin.HOST, server.server_address[1], timeout=10)
    try:
        connection.request(method, path, body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, {name.lower(): value for name, value in response.getheaders()}, response.read()
    finally:
        connection.close()


def chat(server: local_origin.LocalOriginServer, message: str) -> tuple[int, dict[str, str], bytes]:
    """A chat request, read until the server closes the connection, so its log line is already written."""
    return exchange(server, framed(json.dumps({"message": message}).encode()))


def raw(
    *lines: str, body: bytes = b"", method: str = "POST", target: str = local_origin.CHAT_PATH, host: str | None = OWN_HOST
) -> bytes:
    """A raw request; ``exchange`` replaces ``<port>`` with the test server's port."""
    head = [f"{method} {target} HTTP/1.1", *([] if host is None else [f"Host: {host}"]), *lines]
    return ("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + body


def framed(body: bytes, *lines: str, host: str | None = OWN_HOST) -> bytes:
    return raw(JSON, f"Content-Length: {len(body)}", *lines, body=body, host=host)


def exchange(server: local_origin.LocalOriginServer, request: bytes, *, end: bool = True) -> tuple[int, dict[str, str], bytes]:
    """Send raw bytes (then end the stream unless ``end`` is false) and read until the server closes."""
    port = server.server_address[1]
    with socket.create_connection((local_origin.HOST, port), timeout=10) as connection:
        connection.sendall(request.replace(b"<port>", str(port).encode("ascii")))
        if end:
            connection.shutdown(socket.SHUT_WR)
        data = b""
        while chunk := connection.recv(65536):
            data += chunk
    head, _, payload = data.partition(b"\r\n\r\n")
    status_line, *lines = head.decode("latin-1").split("\r\n")
    headers = {name.strip().lower(): value.strip() for name, _, value in (line.partition(":") for line in lines)}
    return int(status_line.split()[1]), headers, payload


def check_body(headers: dict[str, str], body: Any) -> None:
    """A chat API response: the committed schema, and the contract's response headers."""
    assert [error.message for error in VALIDATOR.iter_errors(body)] == []
    assert REQUEST_ID.fullmatch(body["request_id"])
    assert headers["content-type"] == "application/json; charset=utf-8"
    assert headers["cache-control"] == "no-store"
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["connection"] == "close"
    assert not [name for name in headers if name.startswith("access-control-")]  # no CORS


def test_a_valid_message_is_answered_by_the_selected_model_unchanged(
    server: local_origin.LocalOriginServer, engine: ChatEngine, ollama: FakeOllama, capsys: pytest.CaptureFixture[str]
) -> None:
    ollama.reply(ANSWER)
    status, headers, payload = chat(server, MESSAGE)
    assert status == 200
    body = json.loads(payload)
    check_body(headers, body)
    assert body["status"] == "answered"
    assert body["answer_text"] == body["cloud_preview"] == ANSWER
    assert body["continuation"] is None and body["actions"] == [] and body["error"] is None

    # Exactly the engine's own request: the selected model, prompts, knowledge and settings.
    (sent,) = ollama.requests
    prepared = engine.prepare(MESSAGE, local_origin.MODEL_TAG)
    assert sent.payload == json.loads(json.dumps(dict(prepared.payload)))
    assert sent.timeout == engine.client.timeout_s
    selection = yaml.safe_load((PROJECT_ROOT / "configs" / "selection.yaml").read_text(encoding="utf-8"))
    assert sent.payload["model"] == local_origin.MODEL_TAG == selection["selected"]["tag"]

    log = capsys.readouterr().err
    assert body["request_id"] in log and "answered" in log
    assert MESSAGE not in log and ANSWER not in log


def test_navigation_is_answered_by_the_application_without_the_model(
    server: local_origin.LocalOriginServer, ollama: FakeOllama
) -> None:
    status, headers, payload = send(server, "POST", local_origin.CHAT_PATH, NAVIGATION)
    assert status == 200
    body = json.loads(payload)
    check_body(headers, body)
    assert body["status"] == "navigation"
    assert body["actions"] == [{"type": "navigate", "route_id": "project_satira"}]
    assert ollama.requests == []


REJECTED = [
    pytest.param(framed(b'{"message": "unfinished'), 400, "invalid_request", id="malformed-json"),
    pytest.param(raw(JSON, "Content-Length: 8193"), 413, "invalid_request", id="declared-oversized"),
    pytest.param(raw(JSON, f"Content-Length: {10**40}"), 413, "invalid_request", id="huge-content-length"),
    pytest.param(raw(JSON), 400, "invalid_request", id="missing-content-length"),
    pytest.param(raw(JSON, "Content-Length: 1e3"), 400, "invalid_request", id="malformed-content-length"),
    pytest.param(raw(JSON, "Content-Length: 2", "Content-Length: 2"), 400, "invalid_request", id="duplicate-content-length"),
    pytest.param(raw(JSON, "Transfer-Encoding: chunked"), 400, "invalid_request", id="transfer-encoding"),
    pytest.param(raw(JSON, "Content-Length: 50", body=b'{"message":'), 400, "invalid_request", id="incomplete-body"),
    pytest.param(raw(JSON, "Content-Length: 0", target="/api/chat/v1?message=hi"), 400, "invalid_request", id="query-string"),
    pytest.param(raw(method="GET"), 405, "invalid_request", id="get-method"),
    pytest.param(raw("Content-Type: text/plain", "Content-Length: 0"), 415, "invalid_request", id="content-type"),
    pytest.param(raw(JSON, *(f"X-Filler-{n}: x" for n in range(101))), 400, "invalid_request", id="too-many-headers"),
    pytest.param(framed(b'{"message": " "}'), 400, "empty_input", id="empty-message"),
]  # fmt: skip


@pytest.mark.parametrize(("request_bytes", "http_status", "code"), REJECTED)
def test_rejected_requests_get_a_fixed_error_and_never_reach_the_engine(
    server: local_origin.LocalOriginServer, ollama: FakeOllama, request_bytes: bytes, http_status: int, code: str
) -> None:
    status, headers, payload = exchange(server, request_bytes)
    assert status == http_status
    body = json.loads(payload)
    check_body(headers, body)
    expected = {"empty_input": server.engine.chat_config.replies.empty_input}.get(code, local_origin.INVALID_REQUEST_TEXT)
    assert body["status"] == "error" and body["error"] == {"code": code}
    assert body["answer_text"] == body["cloud_preview"] == expected
    assert headers.get("allow") == ("POST" if http_status == 405 else None)
    assert ollama.requests == []


def test_a_stalled_body_is_refused_after_the_timeout(server: local_origin.LocalOriginServer, ollama: FakeOllama) -> None:
    server.request_timeout_s = 0.3
    status, headers, payload = exchange(server, raw(JSON, "Content-Length: 50", body=b'{"message":'), end=False)
    assert status == 400
    check_body(headers, json.loads(payload))
    assert ollama.requests == []


def test_the_allowlists_are_exactly_the_approved_hosts_and_origins() -> None:
    assert local_origin.allowed_hosts(local_origin.PORT) == {"127.0.0.1:8765", "localhost:8765"}
    assert local_origin.allowed_origins(local_origin.PORT) == {
        "http://127.0.0.1:8765", "http://localhost:8765", "http://127.0.0.1:4321", "http://localhost:4321"
    }  # fmt: skip


@pytest.mark.parametrize(
    ("host", "origin"),
    [
        ("127.0.0.1:<port>", None),
        ("localhost:<port>", "http://localhost:4321"),
        ("127.0.0.1:<port>", "http://127.0.0.1:4321"),
        ("localhost:<port>", "http://localhost:<port>"),
    ],
)
def test_approved_hosts_and_origins_are_served(
    server: local_origin.LocalOriginServer, host: str, origin: str | None
) -> None:
    lines = [] if origin is None else [f"Origin: {origin}"]
    status, headers, payload = exchange(server, framed(NAVIGATION, *lines, host=host))
    assert status == 200
    body = json.loads(payload)
    check_body(headers, body)
    assert body["status"] == "navigation"
    status, _, payload = exchange(server, raw(*lines, method="GET", target=local_origin.HEALTH_PATH, host=host))
    assert (status, json.loads(payload)) == (200, {"status": "ok"})


REFUSED_CALLERS = [
    pytest.param(None, [], id="host-missing"),
    pytest.param(OWN_HOST, ["Host: 127.0.0.1:<port>"], id="host-repeated"),
    pytest.param("rebound.example:<port>", [], id="host-unexpected"),
    pytest.param("127.0.0.1", [], id="host-without-port"),
    pytest.param(OWN_HOST, ["Origin:"], id="origin-empty"),
    pytest.param(OWN_HOST, ["Origin: null"], id="origin-null"),
    pytest.param(OWN_HOST, ["Origin: http://localhost:4321", "Origin: http://localhost:4321"], id="origin-repeated"),
    pytest.param(OWN_HOST, ["Origin: http://localhost:4321/"], id="origin-malformed"),
    pytest.param(OWN_HOST, ["Origin: http://rebound.example"], id="origin-other-site"),
]  # fmt: skip


@pytest.mark.parametrize(("host", "lines"), REFUSED_CALLERS)
def test_unexpected_hosts_and_origins_are_refused_on_both_endpoints(
    server: local_origin.LocalOriginServer, ollama: FakeOllama, host: str | None, lines: list[str]
) -> None:
    status, headers, payload = exchange(server, framed(NAVIGATION, *lines, host=host))
    assert status == 400
    body = json.loads(payload)
    check_body(headers, body)
    assert body["status"] == "error" and body["error"] == {"code": "invalid_request"}
    assert body["answer_text"] == body["cloud_preview"] == local_origin.INVALID_REQUEST_TEXT
    status, headers, payload = exchange(server, raw(*lines, method="GET", target=local_origin.HEALTH_PATH, host=host))
    assert (status, json.loads(payload)) == (400, {"error": "invalid_request"})
    assert not [name for name in headers if name.startswith("access-control-")]
    assert ollama.requests == []


@pytest.mark.parametrize(
    ("failure", "http_status", "code"),
    [
        (RuntimeError(SECRET), 500, "unavailable"),
        (OllamaError(OllamaErrorCode.TIMEOUT, SECRET), 504, "timeout"),
        (OllamaError(OllamaErrorCode.UNAVAILABLE, SECRET), 503, "unavailable"),
    ],
    ids=["engine-exception", "ollama-timeout", "ollama-unavailable"],
)
def test_engine_failures_give_a_sanitized_response(
    server: local_origin.LocalOriginServer,
    ollama: FakeOllama,
    capsys: pytest.CaptureFixture[str],
    failure: Exception,
    http_status: int,
    code: str,
) -> None:
    ollama.fail(failure)
    status, headers, payload = chat(server, MESSAGE)
    assert status == http_status
    body = json.loads(payload)
    check_body(headers, body)
    assert body["status"] == "unavailable" and body["error"] == {"code": code}
    assert body["answer_text"] == body["cloud_preview"] == server.engine.chat_config.replies.unavailable
    assert b"synthetic-secret" not in payload and b"facts.md" not in payload
    log = capsys.readouterr().err
    assert "synthetic-secret" not in log and MESSAGE not in log and "Traceback" not in log


def test_health_answers_without_the_engine_or_the_model(
    server: local_origin.LocalOriginServer, ollama: FakeOllama, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(server.engine, "respond", lambda *args, **kwargs: calls.append(args))
    status, headers, payload = send(server, "GET", local_origin.HEALTH_PATH)
    assert status == 200 and json.loads(payload) == {"status": "ok"}
    assert headers["cache-control"] == "no-store"
    assert calls == [] and ollama.requests == []


def test_the_server_listens_on_loopback_only_one_request_at_a_time(server: local_origin.LocalOriginServer) -> None:
    assert (local_origin.HOST, local_origin.PORT) == ("127.0.0.1", 8765)
    assert server.server_address[0] == server.socket.getsockname()[0] == "127.0.0.1"
    assert set(inspect.signature(local_origin.create_server).parameters) == {"engine", "port"}
    assert set(inspect.signature(local_origin.LocalOriginServer).parameters) == {"engine", "port"}
    assert isinstance(server, socketserver.TCPServer) and not isinstance(server, socketserver.ThreadingMixIn)
    with pytest.raises(SystemExit) as refused:  # no command-line option, so no other address
        local_origin.main(["--host", "0.0.0.0"])
    assert refused.value.code == 2


IMPORT_SCRIPT = """
import json, sys, threading
sys.path.insert(0, sys.argv[1])
events = []
sys.addaudithook(lambda event, args: events.append(event) if event.startswith("socket.") else None)
import mascot_lab.local_origin
print(json.dumps({"socket_events": events, "threads": threading.active_count()}))
"""


def test_importing_the_module_starts_no_listener() -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-c", IMPORT_SCRIPT, str(PROJECT_ROOT / "src")],
        capture_output=True, text=True, timeout=60, check=True,
    )  # fmt: skip
    assert json.loads(completed.stdout) == {"socket_events": [], "threads": 1}
    assert completed.stderr == ""
