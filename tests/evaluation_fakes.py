"""Synthetic evaluation cases and an Ollama stand-in for evaluation tests.

Every case here is synthetic test input, not part of the evaluation set,
which the owner writes. ``ModelServer`` is a ``Transport`` that answers by
endpoint and model instead of replaying a queue, so a test can run many
models and cases. It never opens a socket, and every count and duration in
its responses is synthetic test data, not a measurement.
"""

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from ollama_fakes import SentRequest, chat_body

from mascot_lab.ollama import HttpResponse

ANSWER = "Rastin is a third-year Computing student at Queen's University."

NAVIGATION_CASE: dict[str, Any] = {
    "id": "synthetic-nav-qmind",
    "category": "synthetic-navigation",
    "message": "take me to QMIND",
    "expected_route": {"status": "routed", "route_id": "project_qmind"},
}
QUESTION_CASE: dict[str, Any] = {
    "id": "synthetic-question",
    "category": "synthetic-education",
    "message": "Where does Rastin study?",
    "expected_behavior": "SYNTHETIC-REVIEWER-GUIDANCE",
    "expected_route": {"status": "not_navigation"},
    "notes": "SYNTHETIC-AUTHOR-NOTE",
}
AMBIGUOUS_CASE: dict[str, Any] = {
    "id": "synthetic-ambiguous",
    "category": "synthetic-navigation",
    "message": "take me to the research",
    "expected_route": {
        "status": "ambiguous_destination",
        "candidates": ["project_ingenuity", "project_qmind"],
    },
}

type Answer = Callable[[str, str], dict[str, Any] | HttpResponse | BaseException]


def answer_with(content: str = ANSWER, **fields: Any) -> Answer:
    """An answer function that gives every model the same synthetic reply."""
    return lambda model, message: chat_body(content, model=model, **fields)


def write_cases(path: Path, *cases: dict[str, Any]) -> Path:
    """Write ``cases`` as a JSON Lines file at ``path``."""
    path.write_text("".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8")
    return path


class ModelServer:
    """Answers ``/api/show`` and ``/api/chat`` by model; records every request."""

    def __init__(
        self,
        answer: Answer | None = None,
        *,
        missing: Iterable[str] = (),
        capabilities: list[str] | None = None,
        parameters: str | None = 'stop "<end>"',
    ) -> None:
        self.answer = answer if answer is not None else answer_with()
        self.missing = set(missing)
        self.capabilities = ["completion"] if capabilities is None else capabilities
        self.parameters = parameters
        self.show_failure: BaseException | None = None
        self.requests: list[SentRequest] = []

    def send(self, method: str, url: str, body: bytes | None, timeout: float) -> HttpResponse:
        payload = None if body is None else json.loads(body.decode("utf-8"))
        self.requests.append(SentRequest(method, url, payload, timeout))
        endpoint = url.rsplit("/", 1)[-1]
        if endpoint == "show":
            if self.show_failure is not None:
                raise self.show_failure
            tag = payload["model"]
            if tag in self.missing:
                return _json(404, {"error": f"model '{tag}' not found"})
            return _json(200, {"capabilities": self.capabilities, "parameters": self.parameters})
        if endpoint == "chat":
            result = self.answer(payload["model"], payload["messages"][-1]["content"])
            if isinstance(result, BaseException):
                raise result
            return result if isinstance(result, HttpResponse) else _json(200, result)
        raise AssertionError(f"unexpected request: {method} {url}")

    @property
    def chats(self) -> list[SentRequest]:
        return [request for request in self.requests if request.url.endswith("/api/chat")]

    @property
    def shows(self) -> list[SentRequest]:
        return [request for request in self.requests if request.url.endswith("/api/show")]


def _json(status: int, data: Any) -> HttpResponse:
    return HttpResponse(status, json.dumps(data).encode("utf-8"))
