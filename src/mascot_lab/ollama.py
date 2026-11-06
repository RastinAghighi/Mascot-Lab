"""Adapter for Ollama's local HTTP API.

One adapter serves every configured model. A chat request (``POST
/api/chat``) carries only the model's exact tag, the shared chat messages,
Ollama's top-level ``think`` field, and the model's effective options from
``configs/models.yaml``, so nothing else can differ between models. Requests
are not streamed (``"stream": false``), so the time to the first token is
not measured.

The prompts carry the private knowledge block, so the adapter talks only to
a loopback address, never uses a proxy (even one configured in the
environment), and never follows redirects. Responses are size-limited.

Every failure raises ``OllamaError`` with one of these codes:

* ``model_not_found``: Ollama does not have the requested tag;
* ``ollama_unavailable``: the server could not be reached;
* ``timeout``: no complete response arrived within the timeout;
* ``ollama_error``: Ollama answered with an error;
* ``invalid_response``: the response is not the JSON this adapter expects.
"""

import contextlib
import http.client
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from mascot_lab.config import ModelConfig, OptionValue
from mascot_lab.errors import ConfigError, MascotLabError

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT_S = 120.0
MAX_RESPONSE_BYTES = 8 * 1024 * 1024

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_ERROR_TEXT_LIMIT = 500

# The options this adapter sends, by the JSON type Ollama expects. Ollama
# ignores an option name it does not know, logging only a server-side
# warning, so a misspelled option would silently fall back to the model's
# default. Every configured option must therefore be listed here.
_INTEGER_OPTIONS = frozenset(
    {
        "num_ctx",
        "num_predict",
        "seed",
        "top_k",
        "repeat_last_n",
        "num_keep",
        "num_batch",
        "num_gpu",
        "main_gpu",
        "num_thread",
    }
)
_NUMBER_OPTIONS = frozenset(
    {
        "temperature",
        "top_p",
        "min_p",
        "typical_p",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
    }
)
_STRING_LIST_OPTIONS = frozenset({"stop"})
SUPPORTED_OPTIONS = _INTEGER_OPTIONS | _NUMBER_OPTIONS | _STRING_LIST_OPTIONS

# Counts and durations that Ollama reports with a finished response.
_METRICS = (
    "total_duration",
    "load_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
)


class OllamaErrorCode(StrEnum):
    """Why an Ollama request failed."""

    MODEL_NOT_FOUND = "model_not_found"
    UNAVAILABLE = "ollama_unavailable"
    TIMEOUT = "timeout"
    OLLAMA_ERROR = "ollama_error"
    INVALID_RESPONSE = "invalid_response"


class OllamaError(MascotLabError):
    """A failed Ollama request. ``message`` is for logs and records, not visitors."""

    def __init__(
        self,
        code: OllamaErrorCode,
        message: str,
        *,
        http_status: int | None = None,
        elapsed_ms: float | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.elapsed_ms = elapsed_ms


@dataclass(frozen=True, slots=True)
class OllamaMetrics:
    """Counts and durations as reported by Ollama; durations are in nanoseconds.

    A field is None when Ollama did not report it. Nothing here is estimated.
    """

    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


@dataclass(frozen=True, slots=True)
class ChatReply:
    """A model's answer exactly as Ollama returned it."""

    content: str  # raw answer text, unmodified
    thinking: str | None  # reasoning Ollama returned separately, unmodified
    done_reason: str | None  # "stop", or "length" when num_predict cut the answer
    model: str | None  # the model name Ollama reported
    created_at: str | None
    metrics: OllamaMetrics
    elapsed_ms: float  # wall-clock time of the HTTP request, measured by this client


@dataclass(frozen=True, slots=True)
class ModelDetails:
    """Model metadata from ``POST /api/show``; reading it runs no model."""

    tag: str
    capabilities: tuple[str, ...] | None  # None when Ollama does not report them
    parameters: str | None  # the Modelfile's PARAMETER settings, as Ollama reports them


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """A complete HTTP response: any status code and its body."""

    status: int
    body: bytes


class Transport(Protocol):
    """Sends one HTTP request.

    Returns the response for every HTTP status, including errors. Raises
    ``OllamaError`` with code ``timeout``, ``ollama_unavailable``, or
    ``invalid_response`` (oversized body) when there is no usable response.
    """

    def send(self, method: str, url: str, body: bytes | None, timeout: float) -> HttpResponse: ...


def check_options(options: Mapping[str, OptionValue], *, where: str) -> None:
    """Raise ``ConfigError`` unless every option is supported and well typed."""
    for name, value in options.items():
        if name in _INTEGER_OPTIONS:
            valid, expected = type(value) is int, "an integer"
        elif name in _NUMBER_OPTIONS:
            valid, expected = type(value) in (int, float), "a number"
        elif name in _STRING_LIST_OPTIONS:
            valid, expected = isinstance(value, tuple), "a list of strings"
        else:
            raise ConfigError(
                f"{where}: option {name!r} is not supported by the Ollama adapter; "
                f"supported: {', '.join(sorted(SUPPORTED_OPTIONS))}"
            )
        if not valid:
            raise ConfigError(f"{where}: option {name!r} must be {expected}, found {value!r}")
    if "num_predict" in options and options["num_predict"] <= 0:  # type: ignore[operator]
        raise ConfigError(f"{where}: num_predict must be positive so that output stays bounded")
    if "temperature" in options and options["temperature"] < 0:  # type: ignore[operator]
        raise ConfigError(f"{where}: temperature must not be negative")
    if "top_p" in options and not 0 <= options["top_p"] <= 1:  # type: ignore[operator]
        raise ConfigError(f"{where}: top_p must be between 0 and 1")


def check_base_url(url: str) -> str:
    """Return ``url`` as ``http://host:port`` if it names a loopback Ollama server.

    Any other host is rejected: the prompts carry the private knowledge block.
    """
    try:
        parts = urllib.parse.urlsplit(url)
        _ = parts.port  # raises ValueError for an invalid port
    except ValueError as exc:
        raise ConfigError(f"invalid Ollama URL {url!r}: {exc}") from exc
    if parts.scheme != "http" or parts.hostname not in _LOOPBACK_HOSTS:
        raise ConfigError(
            f"Ollama URL {url!r} must use http:// and a loopback host (127.0.0.1, localhost, "
            "or [::1]): requests carry the private knowledge block"
        )
    if parts.username or parts.password or parts.path.strip("/") or parts.query or parts.fragment:
        raise ConfigError(f"Ollama URL {url!r} must name only a host and port")
    return f"http://{parts.netloc}"


class HttpTransport:
    """Sends requests with urllib, without proxies or redirects, and with a size limit."""

    def __init__(self, *, max_response_bytes: int = MAX_RESPONSE_BYTES) -> None:
        self._max_bytes = max_response_bytes
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),  # ignore proxies configured in the environment
            _RefuseRedirects(),
        )

    def send(self, method: str, url: str, body: bytes | None, timeout: float) -> HttpResponse:
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            try:
                with self._opener.open(request, timeout=timeout) as response:
                    return HttpResponse(response.status, self._read(response))
            except urllib.error.HTTPError as exc:  # an error status is still a response
                with contextlib.closing(exc):
                    return HttpResponse(exc.code, self._read(exc))
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise _timeout(url, timeout) from exc
            raise OllamaError(
                OllamaErrorCode.UNAVAILABLE, f"cannot reach Ollama at {url}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise _timeout(url, timeout) from exc
        except (OSError, http.client.HTTPException) as exc:
            raise OllamaError(
                OllamaErrorCode.UNAVAILABLE, f"the connection to Ollama at {url} failed: {exc!r}"
            ) from exc

    def _read(self, response: Any) -> bytes:
        data: bytes = response.read(self._max_bytes + 1)
        if len(data) > self._max_bytes:
            raise OllamaError(
                OllamaErrorCode.INVALID_RESPONSE,
                f"the response is larger than {self._max_bytes} bytes",
                http_status=getattr(response, "status", None),
            )
        return data


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None  # urllib then raises the 3xx response as an HTTPError


def _timeout(url: str, timeout: float) -> OllamaError:
    return OllamaError(OllamaErrorCode.TIMEOUT, f"no complete response from {url} within {timeout:g} s")


class OllamaClient:
    """Builds and sends requests to one local Ollama server."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: Transport | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if (
            type(timeout_s) not in (int, float)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ConfigError(f"the Ollama timeout must be a positive number of seconds, found {timeout_s!r}")
        self._base_url = check_base_url(base_url)
        self._timeout_s = float(timeout_s)
        self._transport: Transport = transport if transport is not None else HttpTransport()
        self._clock = clock

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def timeout_s(self) -> float:
        return self._timeout_s

    @property
    def chat_url(self) -> str:
        return f"{self._base_url}/api/chat"

    def chat_payload(
        self, model: ModelConfig, messages: Sequence[Mapping[str, str]]
    ) -> dict[str, Any]:
        """Return the ``/api/chat`` request body for ``model`` and ``messages``.

        Only the tag, the messages, ``think``, and the configured options go
        into it, so two models differ only where their configuration does.
        """
        check_options(model.options, where=f"model {model.tag!r}")
        payload: dict[str, Any] = {
            "model": model.tag,
            "messages": [{"role": item["role"], "content": item["content"]} for item in messages],
            "stream": False,
        }
        if model.think is not None:
            payload["think"] = model.think
        payload["options"] = {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in model.options.items()
        }
        return payload

    def send_chat(self, payload: Mapping[str, Any]) -> ChatReply:
        """Send a body built by ``chat_payload`` and return the model's reply."""
        tag = str(payload.get("model"))
        data, elapsed_ms = self._post("/api/chat", payload, tag=tag)
        return _chat_reply(data, elapsed_ms)

    def chat(self, model: ModelConfig, messages: Sequence[Mapping[str, str]]) -> ChatReply:
        """Build and send one chat request."""
        return self.send_chat(self.chat_payload(model, messages))

    def show(self, tag: str) -> ModelDetails:
        """Return the metadata of an installed model (``POST /api/show``); runs no model."""
        data, elapsed_ms = self._post("/api/show", {"model": tag}, tag=tag)
        capabilities = data.get("capabilities")
        if capabilities is not None and not (
            isinstance(capabilities, list) and all(isinstance(item, str) for item in capabilities)
        ):
            raise _invalid("capabilities is not a list of strings", elapsed_ms)
        parameters = data.get("parameters")
        if parameters is not None and not isinstance(parameters, str):
            raise _invalid("parameters is not a string", elapsed_ms)
        return ModelDetails(
            tag=tag,
            capabilities=None if capabilities is None else tuple(capabilities),
            parameters=parameters,
        )

    def _post(self, path: str, payload: Mapping[str, Any], *, tag: str) -> tuple[dict[str, Any], float]:
        body = json.dumps(dict(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
        started = self._clock()
        try:
            response = self._transport.send("POST", self._base_url + path, body, self._timeout_s)
        except OllamaError as exc:
            exc.elapsed_ms = _elapsed_ms(started, self._clock())
            raise
        elapsed_ms = _elapsed_ms(started, self._clock())
        return _decode(response, tag=tag, elapsed_ms=elapsed_ms), elapsed_ms


def _elapsed_ms(started: float, finished: float) -> float:
    return round((finished - started) * 1000, 3)


def _decode(response: HttpResponse, *, tag: str, elapsed_ms: float) -> dict[str, Any]:
    """Return the JSON object of a successful response, or raise ``OllamaError``."""
    try:
        data = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        data = None
    if not isinstance(data, dict):
        data = None
    error = None if data is None else data.get("error")
    if response.status == 200 and data is not None and error is None:
        return data

    if isinstance(error, str) and error:
        detail = error[:_ERROR_TEXT_LIMIT]
    else:
        detail = response.body[:200].decode("utf-8", errors="replace") or "(empty body)"
    if response.status == 404 and isinstance(error, str):
        raise OllamaError(
            OllamaErrorCode.MODEL_NOT_FOUND,
            f"Ollama has no model {tag!r}: {detail}",
            http_status=404,
            elapsed_ms=elapsed_ms,
        )
    if response.status != 200:
        raise OllamaError(
            OllamaErrorCode.OLLAMA_ERROR,
            f"Ollama answered HTTP {response.status}: {detail}",
            http_status=response.status,
            elapsed_ms=elapsed_ms,
        )
    if data is None:
        raise _invalid(f"the response is not a JSON object: {detail}", elapsed_ms)
    raise OllamaError(
        OllamaErrorCode.OLLAMA_ERROR,
        f"Ollama reported an error: {detail}",
        http_status=200,
        elapsed_ms=elapsed_ms,
    )


def _invalid(reason: str, elapsed_ms: float | None) -> OllamaError:
    return OllamaError(
        OllamaErrorCode.INVALID_RESPONSE,
        f"unexpected Ollama response: {reason}",
        http_status=200,
        elapsed_ms=elapsed_ms,
    )


def _chat_reply(data: dict[str, Any], elapsed_ms: float) -> ChatReply:
    message = data.get("message")
    if not isinstance(message, dict):
        raise _invalid("it has no message object", elapsed_ms)
    if message.get("role") != "assistant":
        raise _invalid(f"the message role is {message.get('role')!r}, not 'assistant'", elapsed_ms)
    content = message.get("content")
    if not isinstance(content, str):
        raise _invalid("the message content is not a string", elapsed_ms)
    thinking = message.get("thinking")
    if thinking is not None and not isinstance(thinking, str):
        raise _invalid("the message thinking is not a string", elapsed_ms)
    if data.get("done") is not True:
        raise _invalid("the response is not marked done", elapsed_ms)
    text_fields: dict[str, str | None] = {}
    for name in ("done_reason", "model", "created_at"):
        value = data.get(name)
        if value is not None and not isinstance(value, str):
            raise _invalid(f"{name} is not a string", elapsed_ms)
        text_fields[name] = value
    metrics: dict[str, int | None] = {}
    for name in _METRICS:
        value = data.get(name)
        if value is not None and (type(value) is not int or value < 0):
            raise _invalid(f"{name} is not a non-negative integer", elapsed_ms)
        metrics[name] = value
    return ChatReply(
        content=content,
        thinking=thinking,
        done_reason=text_fields["done_reason"],
        model=text_fields["model"],
        created_at=text_fields["created_at"],
        metrics=OllamaMetrics(**metrics),
        elapsed_ms=elapsed_ms,
    )
