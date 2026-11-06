"""Tests for the Ollama adapter.

Client tests use ``FakeOllama``. ``HttpTransport`` is tested against a small
stand-in HTTP server on 127.0.0.1 that returns scripted responses; no test
contacts a real Ollama server or runs a model.
"""

import http.client
import json
import threading
import urllib.error
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import MappingProxyType
from typing import Any

import pytest
from ollama_fakes import TAG, FakeOllama, chat_body

from mascot_lab import ConfigError, ModelsConfig, parse_models_config
from mascot_lab.config import ModelConfig
from mascot_lab.ollama import (
    DEFAULT_BASE_URL,
    HttpResponse,
    HttpTransport,
    OllamaClient,
    OllamaError,
    OllamaErrorCode,
    OllamaMetrics,
    check_base_url,
    check_options,
)

MESSAGES = [
    {"role": "system", "content": "System text."},
    {"role": "user", "content": "Where does Rastin study?"},
]


def model_config(**options: Any) -> ModelConfig:
    return ModelConfig(
        name="Synthetic",
        tag="family-a:1b-q4_K_M",
        options=MappingProxyType({"num_ctx": 4096, **options}),
        override=None,
        think=False,
    )


# ------------------------------------------------------------ the payload


def test_payload_carries_only_the_tag_messages_think_and_configured_options(
    models_config: ModelsConfig,
) -> None:
    client = OllamaClient(transport=FakeOllama())
    for model in models_config.models:
        payload = client.chat_payload(model, MESSAGES)
        assert payload == {
            "model": model.tag,
            "messages": MESSAGES,
            "stream": False,
            "think": False,
            "options": {
                "num_ctx": 4096,
                "temperature": 0.0,
                "seed": 42,
                "top_p": 1.0,
                "num_predict": 160,
            },
        }


def test_payloads_for_two_models_differ_only_in_the_tag(models_config: ModelsConfig) -> None:
    client = OllamaClient(transport=FakeOllama())
    first, *others = (client.chat_payload(model, MESSAGES) for model in models_config.models)
    for payload in others:
        assert {**payload, "model": first["model"]} == first


def test_think_is_omitted_when_not_configured() -> None:
    config = parse_models_config(
        "schema_version: 1\nbaseline:\n  options:\n    num_ctx: 4096\n"
        "models:\n  - name: A\n    tag: family-a:1b\n"
    )
    payload = OllamaClient(transport=FakeOllama()).chat_payload(config.models[0], MESSAGES)
    assert "think" not in payload


def test_list_options_are_sent_as_json_lists() -> None:
    payload = OllamaClient(transport=FakeOllama()).chat_payload(model_config(stop=("\n\n",)), MESSAGES)
    assert payload["options"]["stop"] == ["\n\n"]


def test_payload_messages_are_copies_with_only_role_and_content() -> None:
    messages = [{"role": "user", "content": "Hi", "images": "x"}]
    payload = OllamaClient(transport=FakeOllama()).chat_payload(model_config(), messages)
    assert payload["messages"] == [{"role": "user", "content": "Hi"}]
    assert payload["messages"][0] is not messages[0]


# ---------------------------------------------------------------- options


def test_the_real_configuration_passes_the_option_check(models_config: ModelsConfig) -> None:
    for model in models_config.models:
        check_options(model.options, where=model.tag)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        pytest.param({"temprature": 0.0}, "not supported", id="misspelled-option"),
        pytest.param({"think": False}, "not supported", id="think-is-not-an-option"),
        pytest.param({"seed": 42.0}, "must be an integer", id="float-seed"),
        pytest.param({"num_predict": True}, "must be an integer", id="boolean-num-predict"),
        pytest.param({"temperature": "0"}, "must be a number", id="string-temperature"),
        pytest.param({"temperature": False}, "must be a number", id="boolean-temperature"),
        pytest.param({"stop": "\n"}, "list of strings", id="string-stop"),
        pytest.param({"num_predict": 0}, "positive", id="zero-num-predict"),
        pytest.param({"num_predict": -1}, "positive", id="unbounded-num-predict"),
        pytest.param({"temperature": -0.1}, "negative", id="negative-temperature"),
        pytest.param({"top_p": 1.5}, "between 0 and 1", id="top-p-above-one"),
    ],
)
def test_unsupported_or_mistyped_options_are_rejected(options: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        check_options(options, where="test")


def test_payload_construction_refuses_an_unsupported_option() -> None:
    with pytest.raises(ConfigError, match="not supported"):
        OllamaClient(transport=FakeOllama()).chat_payload(model_config(mirostat_eta=0.1), MESSAGES)


# ---------------------------------------------------------- client settings


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (DEFAULT_BASE_URL, "http://127.0.0.1:11434"),
        ("http://localhost:11434/", "http://localhost:11434"),
        ("http://[::1]:11434", "http://[::1]:11434"),
        ("http://LOCALHOST:8080", "http://LOCALHOST:8080"),
    ],
)
def test_loopback_urls_are_accepted(url: str, expected: str) -> None:
    assert check_base_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:11434",
        "http://example.com:11434",
        "http://192.168.1.10:11434",
        "http://0.0.0.0:11434",
        "http://127.0.0.1:11434@example.com",
        "http://user:pass@127.0.0.1:11434",
        "http://127.0.0.1:11434/api",
        "http://127.0.0.1:11434?x=1",
        "http://127.0.0.1:notaport",
        "127.0.0.1:11434",
    ],
)
def test_non_loopback_or_decorated_urls_are_rejected(url: str) -> None:
    with pytest.raises(ConfigError):
        check_base_url(url)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True, "120"])
def test_invalid_timeouts_are_rejected(timeout: Any) -> None:
    with pytest.raises(ConfigError, match="timeout"):
        OllamaClient(timeout_s=timeout, transport=FakeOllama())


# ------------------------------------------------------- successful replies


def test_reply_is_parsed_without_modification(ollama: FakeOllama) -> None:
    raw = "  <think>hidden</think>Rastin studies at **Queen's University**. https://x.example  "
    ollama.reply(raw, thinking="separate reasoning", done_reason="length")
    ticks = iter([10.0, 10.25])
    client = OllamaClient(transport=ollama, timeout_s=30, clock=lambda: next(ticks))
    reply = client.chat(model_config(), MESSAGES)
    assert reply.content == raw
    assert reply.thinking == "separate reasoning"
    assert reply.done_reason == "length"
    assert reply.model == TAG
    assert reply.created_at == "2026-11-06T12:00:00Z"
    assert reply.metrics == OllamaMetrics(600, 100, 20, 200, 10, 300)
    assert reply.elapsed_ms == 250.0
    (request,) = ollama.requests
    assert (request.method, request.url, request.timeout) == ("POST", "http://127.0.0.1:11434/api/chat", 30.0)
    assert request.payload["model"] == "family-a:1b-q4_K_M"


def test_missing_metrics_are_recorded_as_unavailable(ollama: FakeOllama) -> None:
    body = chat_body("Answer.")
    for name in ("prompt_eval_count", "prompt_eval_duration", "load_duration"):
        del body[name]
    ollama.json(200, body)
    reply = OllamaClient(transport=ollama).chat(model_config(), MESSAGES)
    assert reply.metrics.prompt_eval_count is None
    assert reply.metrics.load_duration is None
    assert reply.metrics.eval_count == 10


def test_absent_thinking_and_done_reason_are_none(ollama: FakeOllama) -> None:
    body = chat_body("Answer.")
    del body["done_reason"]
    ollama.json(200, body)
    reply = OllamaClient(transport=ollama).chat(model_config(), MESSAGES)
    assert reply.thinking is None
    assert reply.done_reason is None


# ----------------------------------------------------------------- failures


def _with(**changes: Any) -> dict[str, Any]:
    body = chat_body("Answer.")
    for key, value in changes.items():
        if value is _DELETE:
            del body[key]
        else:
            body[key] = value
    return body


_DELETE = object()

FAILURES = [
    pytest.param(
        HttpResponse(404, b'{"error":"model \\"granite4:micro-h\\" not found, try pulling it first"}'),
        OllamaErrorCode.MODEL_NOT_FOUND,
        404,
        id="model-not-pulled",
    ),
    pytest.param(
        HttpResponse(404, b"404 page not found"),
        OllamaErrorCode.OLLAMA_ERROR,
        404,
        id="not-an-ollama-endpoint",
    ),
    pytest.param(
        HttpResponse(400, b'{"error":"\\"x\\" does not support thinking"}'),
        OllamaErrorCode.OLLAMA_ERROR,
        400,
        id="thinking-rejected",
    ),
    pytest.param(
        HttpResponse(500, b'{"error":"llama runner process has terminated"}'),
        OllamaErrorCode.OLLAMA_ERROR,
        500,
        id="server-error",
    ),
    pytest.param(
        HttpResponse(307, b""), OllamaErrorCode.OLLAMA_ERROR, 307, id="refused-redirect"
    ),
    pytest.param(
        HttpResponse(200, b'{"error":"unexpected server error"}'),
        OllamaErrorCode.OLLAMA_ERROR,
        200,
        id="error-in-a-200-body",
    ),
    pytest.param(HttpResponse(200, b"not json"), OllamaErrorCode.INVALID_RESPONSE, 200, id="not-json"),
    pytest.param(HttpResponse(200, b"\xff\xfe"), OllamaErrorCode.INVALID_RESPONSE, 200, id="not-utf8"),
    pytest.param(HttpResponse(200, b"[1, 2]"), OllamaErrorCode.INVALID_RESPONSE, 200, id="json-array"),
    pytest.param(HttpResponse(200, b""), OllamaErrorCode.INVALID_RESPONSE, 200, id="empty-body"),
]

INVALID_BODIES = [
    pytest.param(_with(message=_DELETE), id="no-message"),
    pytest.param(_with(message="text"), id="message-not-an-object"),
    pytest.param(_with(message={"role": "user", "content": "x"}), id="wrong-role"),
    pytest.param(_with(message={"role": "assistant"}), id="no-content"),
    pytest.param(_with(message={"role": "assistant", "content": 5}), id="numeric-content"),
    pytest.param(
        _with(message={"role": "assistant", "content": "x", "thinking": ["a"]}), id="list-thinking"
    ),
    pytest.param(_with(done=False), id="not-done"),
    pytest.param(_with(done=_DELETE), id="no-done"),
    pytest.param(_with(done_reason=1), id="numeric-done-reason"),
    pytest.param(_with(eval_count=-1), id="negative-metric"),
    pytest.param(_with(eval_count=1.5), id="float-metric"),
    pytest.param(_with(total_duration="5"), id="string-metric"),
]


@pytest.mark.parametrize(("response", "code", "status"), FAILURES)
def test_http_failures_are_classified(
    ollama: FakeOllama, response: HttpResponse, code: OllamaErrorCode, status: int
) -> None:
    ollama.queue.append(response)
    with pytest.raises(OllamaError) as caught:
        OllamaClient(transport=ollama).chat(model_config(), MESSAGES)
    assert caught.value.code is code
    assert caught.value.http_status == status
    assert caught.value.elapsed_ms is not None


@pytest.mark.parametrize("body", INVALID_BODIES)
def test_malformed_replies_are_invalid_responses(ollama: FakeOllama, body: dict[str, Any]) -> None:
    ollama.json(200, body)
    with pytest.raises(OllamaError) as caught:
        OllamaClient(transport=ollama).chat(model_config(), MESSAGES)
    assert caught.value.code is OllamaErrorCode.INVALID_RESPONSE


def test_model_not_found_names_the_tag(ollama: FakeOllama) -> None:
    ollama.json(404, {"error": f"model '{TAG}' not found"})
    with pytest.raises(OllamaError, match=f"Ollama has no model '{TAG}'"):
        OllamaClient(transport=ollama).send_chat({"model": TAG, "messages": []})


@pytest.mark.parametrize("code", [OllamaErrorCode.TIMEOUT, OllamaErrorCode.UNAVAILABLE])
def test_transport_failures_pass_through_with_elapsed_time(ollama: FakeOllama, code: OllamaErrorCode) -> None:
    ollama.fail(OllamaError(code, "synthetic failure"))
    ticks = iter([1.0, 3.5])
    client = OllamaClient(transport=ollama, clock=lambda: next(ticks))
    with pytest.raises(OllamaError) as caught:
        client.chat(model_config(), MESSAGES)
    assert caught.value.code is code
    assert caught.value.elapsed_ms == 2500.0


def test_long_error_text_is_truncated(ollama: FakeOllama) -> None:
    ollama.json(500, {"error": "x" * 5000})
    with pytest.raises(OllamaError) as caught:
        OllamaClient(transport=ollama).chat(model_config(), MESSAGES)
    assert len(caught.value.message) < 600


# ------------------------------------------------------------ model metadata


def test_show_reads_capabilities_and_parameters(ollama: FakeOllama) -> None:
    ollama.json(200, {"capabilities": ["completion", "thinking"], "parameters": "stop <|im_end|>"})
    details = OllamaClient(transport=ollama).show("qwen3.5:2b-q4_K_M")
    assert details.capabilities == ("completion", "thinking")
    assert details.parameters == "stop <|im_end|>"
    (request,) = ollama.requests
    assert request.url.endswith("/api/show")
    assert request.payload == {"model": "qwen3.5:2b-q4_K_M"}


def test_show_without_capabilities_reports_them_as_unknown(ollama: FakeOllama) -> None:
    ollama.json(200, {"modelfile": "FROM x"})
    details = OllamaClient(transport=ollama).show(TAG)
    assert details.capabilities is None
    assert details.parameters is None


def test_show_of_a_missing_model(ollama: FakeOllama) -> None:
    ollama.json(404, {"error": f"model '{TAG}' not found"})
    with pytest.raises(OllamaError) as caught:
        OllamaClient(transport=ollama).show(TAG)
    assert caught.value.code is OllamaErrorCode.MODEL_NOT_FOUND


@pytest.mark.parametrize("data", [{"capabilities": "thinking"}, {"capabilities": [1]}, {"parameters": 3}])
def test_show_rejects_malformed_metadata(ollama: FakeOllama, data: dict[str, Any]) -> None:
    ollama.json(200, data)
    with pytest.raises(OllamaError) as caught:
        OllamaClient(transport=ollama).show(TAG)
    assert caught.value.code is OllamaErrorCode.INVALID_RESPONSE


# ------------------------------------------- HTTP transport, local stand-in


class _StandIn(ThreadingHTTPServer):
    """Answers every POST with one scripted response and records the requests."""

    daemon_threads = True
    block_on_close = False

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.received: list[dict[str, Any]] = []
        self.status = 200
        self.body: bytes = json.dumps(chat_body("Answer.")).encode("utf-8")
        self.headers: dict[str, str] = {}
        self.delay = 0.0
        self.release = threading.Event()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def handle_error(self, request: Any, client_address: Any) -> None:
        pass  # a client that timed out has closed its socket


class _Handler(BaseHTTPRequestHandler):
    server: _StandIn

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.server.received.append(
            {"path": self.path, "content_type": self.headers.get("Content-Type"), "body": body}
        )
        if self.server.delay:
            self.server.release.wait(self.server.delay)
        self.send_response(self.server.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.server.body)))
        for name, value in self.server.headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(self.server.body)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def stand_in() -> Iterator[_StandIn]:
    server = _StandIn()
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_transport_posts_json_and_returns_the_reply(stand_in: _StandIn) -> None:
    client = OllamaClient(stand_in.url, timeout_s=5, transport=HttpTransport())
    reply = client.chat(model_config(), MESSAGES)
    assert reply.content == "Answer."
    (received,) = stand_in.received
    assert received["path"] == "/api/chat"
    assert received["content_type"] == "application/json"
    assert json.loads(received["body"])["messages"] == MESSAGES


def test_http_error_statuses_are_returned_as_responses(stand_in: _StandIn) -> None:
    stand_in.status, stand_in.body = 404, b'{"error":"model not found"}'
    response = HttpTransport().send("POST", stand_in.url + "/api/chat", b"{}", 5)
    assert response == HttpResponse(404, b'{"error":"model not found"}')


def test_http_timeout_is_reported_as_timeout(stand_in: _StandIn) -> None:
    stand_in.delay = 10
    client = OllamaClient(stand_in.url, timeout_s=0.3, transport=HttpTransport())
    with pytest.raises(OllamaError) as caught:
        client.chat(model_config(), MESSAGES)
    assert caught.value.code is OllamaErrorCode.TIMEOUT
    assert caught.value.elapsed_ms is not None and caught.value.elapsed_ms >= 250


def test_redirects_are_not_followed(stand_in: _StandIn) -> None:
    stand_in.status, stand_in.body = 307, b""
    stand_in.headers = {"Location": "http://example.com/api/chat"}
    response = HttpTransport().send("POST", stand_in.url + "/api/chat", b"{}", 5)
    assert response.status == 307
    assert len(stand_in.received) == 1


def test_environment_proxies_are_ignored(stand_in: _StandIn, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")  # nothing listens there
    response = HttpTransport().send("POST", stand_in.url + "/api/chat", b"{}", 5)
    assert response.status == 200
    assert len(stand_in.received) == 1


def test_oversized_responses_are_rejected(stand_in: _StandIn) -> None:
    with pytest.raises(OllamaError) as caught:
        HttpTransport(max_response_bytes=10).send("POST", stand_in.url + "/api/chat", b"{}", 5)
    assert caught.value.code is OllamaErrorCode.INVALID_RESPONSE


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (urllib.error.URLError(ConnectionRefusedError(10061, "refused")), OllamaErrorCode.UNAVAILABLE),
        (urllib.error.URLError(TimeoutError("timed out")), OllamaErrorCode.TIMEOUT),
        (TimeoutError("timed out"), OllamaErrorCode.TIMEOUT),
        (http.client.RemoteDisconnected("closed"), OllamaErrorCode.UNAVAILABLE),
        (ConnectionResetError(104, "reset"), OllamaErrorCode.UNAVAILABLE),
    ],
)
def test_connection_failures_are_classified(
    monkeypatch: pytest.MonkeyPatch, error: BaseException, code: OllamaErrorCode
) -> None:
    transport = HttpTransport()

    def fail(*args: Any, **kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(transport._opener, "open", fail)
    with pytest.raises(OllamaError) as caught:
        transport.send("POST", DEFAULT_BASE_URL + "/api/chat", b"{}", 5)
    assert caught.value.code is code
