"""A scripted stand-in for the Ollama server. No test contacts a real one.

``FakeOllama`` is a ``Transport``: it replays queued HTTP responses (or
errors) in order, records every request, and fails the test on any request
it was not scripted for. Every count and duration in these responses is
synthetic test data, not a measurement.
"""

import json
from dataclasses import dataclass
from typing import Any

from mascot_lab.ollama import HttpResponse

TAG = "granite4:micro-h"


@dataclass(frozen=True)
class SentRequest:
    method: str
    url: str
    payload: Any
    timeout: float


class FakeOllama:
    """Replays queued responses and records every request; never opens a socket."""

    def __init__(self) -> None:
        self.queue: list[HttpResponse | BaseException] = []
        self.requests: list[SentRequest] = []

    def send(self, method: str, url: str, body: bytes | None, timeout: float) -> HttpResponse:
        payload = None if body is None else json.loads(body.decode("utf-8"))
        self.requests.append(SentRequest(method, url, payload, timeout))
        if not self.queue:
            raise AssertionError(f"unscripted request: {method} {url}")
        item = self.queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def reply(self, content: str, **fields: Any) -> "FakeOllama":
        """Queue a successful /api/chat response with ``content``."""
        return self.json(200, chat_body(content, **fields))

    def json(self, status: int, data: Any) -> "FakeOllama":
        """Queue a response whose body is ``data`` as JSON."""
        return self.raw(status, json.dumps(data).encode("utf-8"))

    def raw(self, status: int, body: bytes) -> "FakeOllama":
        self.queue.append(HttpResponse(status, body))
        return self

    def fail(self, error: BaseException) -> "FakeOllama":
        """Queue an error raised by the transport, such as a timeout."""
        self.queue.append(error)
        return self


def chat_body(
    content: str,
    *,
    thinking: str | None = None,
    done_reason: str | None = "stop",
    model: str = TAG,
    **overrides: Any,
) -> dict[str, Any]:
    """A non-streamed /api/chat response body as Ollama documents it."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if thinking is not None:
        message["thinking"] = thinking
    body: dict[str, Any] = {
        "model": model,
        "created_at": "2026-11-06T12:00:00Z",
        "message": message,
        "done": True,
        "done_reason": done_reason,
        # Synthetic values: tests check that numbers pass through unchanged.
        "total_duration": 600,
        "load_duration": 100,
        "prompt_eval_count": 20,
        "prompt_eval_duration": 200,
        "eval_count": 10,
        "eval_duration": 300,
    }
    body.update(overrides)
    return body
