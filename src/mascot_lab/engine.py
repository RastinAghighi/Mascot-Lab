"""The shared chat engine: one request pipeline for every configured model.

``ChatEngine.respond(message, model)`` handles one visitor message in this
fixed order and stops at the first step that settles it:

1. Input validation. Line endings become LF. Control characters other than
   line breaks and tabs are removed, as are invisible format characters
   (Unicode category Cf, such as zero-width spaces, bidirectional controls,
   and tag characters), which can hide text from a human reader. Surrounding
   whitespace is trimmed. An empty message, or one longer than
   ``input.max_characters`` in ``configs/chat.yaml``, is refused.
2. Deterministic routing. ``Router`` recognizes navigation commands, and the
   application answers them without a model: a routed command navigates, an
   ambiguous one suggests its candidate routes, and unknown destinations and
   URLs are refused.
3. Approved knowledge loading. The bounded knowledge block is read from
   ``knowledge/facts.md`` for every request that reaches this step, so the
   facts are supplied server-side on each request (design plan, p. 7).
4. Shared prompt construction. ``PromptBuilder`` builds the same system and
   user messages for every model.
5. Ollama request. ``OllamaClient`` sends them with the model's exact tag and
   its configured ``think`` field and options.
6. Output validation. ``OutputValidator`` derives the display text from the
   raw answer, which the response keeps unchanged.

Every outcome is a ``ChatResponse``. Its navigation actions come only from
the router's allowlist, and its continuation link only from
``configs/chat.yaml``: nothing a model writes can add an action or a link.
The engine is single-turn, like the cloud, which shows only the latest
question and answer (p. 7).
"""

import copy
import dataclasses
import functools
import os
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from mascot_lab._textfile import normalize_newlines
from mascot_lab.chat_config import ChatConfig, load_chat_config
from mascot_lab.config import ModelConfig, ModelsConfig, load_models_config
from mascot_lab.errors import ConfigError, KnowledgeError
from mascot_lab.knowledge import KnowledgeBlock, load_knowledge
from mascot_lab.ollama import ChatReply, OllamaClient, OllamaError, check_options
from mascot_lab.output import OutputIssue, OutputValidator, cloud_preview
from mascot_lab.prompt import BuiltPrompt, PromptBuilder, SystemInstruction, load_system_instruction
from mascot_lab.router import RouteResult, RouteStatus, Router

_ISSUE_ORDER = {issue: index for index, issue in enumerate(OutputIssue)}


class ChatStatus(StrEnum):
    """The outcome of one message."""

    ANSWERED = "answered"  # the model's validated answer
    ROUTED = "routed"  # a direct navigation request; no model was called
    AMBIGUOUS_DESTINATION = "ambiguous_destination"  # navigation naming several routes
    UNKNOWN_DESTINATION = "unknown_destination"  # navigation to a destination not allowlisted
    URL_REJECTED = "url_rejected"  # navigation containing a URL, address, or path
    INVALID_INPUT = "invalid_input"  # an empty or too long message; nothing else ran
    WITHHELD = "withheld"  # output validation refused to show the model's answer
    ERROR = "error"  # no usable answer; see ``error.code``


class ActionType(StrEnum):
    """What the website does with a navigation action (design plan, p. 7)."""

    NAVIGATE = "navigate"  # go to the route now: "show me Satira" goes straight to the page
    SUGGEST = "suggest"  # offer the route, as the reply's "take me there" button


class ErrorCode(StrEnum):
    """Why no usable answer was produced."""

    EMPTY_INPUT = "empty_input"
    INPUT_TOO_LONG = "input_too_long"
    KNOWLEDGE_UNAVAILABLE = "knowledge_unavailable"
    MODEL_NOT_FOUND = "model_not_found"
    OLLAMA_UNAVAILABLE = "ollama_unavailable"
    TIMEOUT = "timeout"
    OLLAMA_ERROR = "ollama_error"
    INVALID_RESPONSE = "invalid_response"
    EMPTY_OUTPUT = "empty_output"


class ContinuationReason(StrEnum):
    """Why an answer continues on the mascot's page."""

    EXCEEDS_CLOUD = "exceeds_cloud"  # the answer is longer than the cloud preview
    OUTPUT_LIMIT_REACHED = "output_limit_reached"  # generation stopped at num_predict


@dataclass(frozen=True, slots=True)
class NavigationAction:
    """A semantic navigation action; ``route_id`` is always from the allowlist."""

    type: ActionType
    route_id: str


@dataclass(frozen=True, slots=True)
class Continuation:
    """The cloud ends with a link to the mascot's page, where the answer continues."""

    route_id: str
    reasons: tuple[ContinuationReason, ...]


@dataclass(frozen=True, slots=True)
class ChatError:
    """Why no usable answer was produced. ``message`` is for records, not visitors."""

    code: ErrorCode
    message: str
    http_status: int | None = None
    elapsed_ms: float | None = None


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """The structured result of one message.

    The first five fields are what a visitor would see: ``answer_text`` (the
    validated model answer, or a fixed reply), ``cloud_preview`` (its start,
    sized for the thought cloud), ``continuation`` (set when the answer goes
    on past the preview or was cut at the output limit), and ``actions``. The
    rest is the record: what ran, with the raw model output in ``output``,
    kept separate from the display text.
    """

    status: ChatStatus
    answer_text: str
    cloud_preview: str
    continuation: Continuation | None
    actions: tuple[NavigationAction, ...]
    model: str | None  # tag of the selected model, also when no model was called
    message: str | None  # the validated message; None when input validation failed
    route: RouteResult | None  # None when input validation failed
    prompt: BuiltPrompt | None  # set when steps 3 and 4 ran
    request: Mapping[str, Any] | None  # Ollama request settings, without the messages
    output: ChatReply | None  # the raw model output, when Ollama returned one
    output_issues: tuple[OutputIssue, ...]
    error: ChatError | None
    configs: Mapping[str, str]  # SHA-256 of each configuration file and the instruction

    def __post_init__(self) -> None:
        if not isinstance(self.status, ChatStatus):
            raise TypeError(f"status must be a ChatStatus, not {type(self.status).__name__}")
        if not self.answer_text.strip() or not self.cloud_preview.strip():
            raise ValueError("a response always carries display text")
        types = [action.type for action in self.actions]
        if self.status is ChatStatus.ROUTED:
            actions_valid = types == [ActionType.NAVIGATE]
        elif self.status is ChatStatus.AMBIGUOUS_DESTINATION:
            actions_valid = (
                len(types) >= 2
                and set(types) == {ActionType.SUGGEST}
                and len({action.route_id for action in self.actions}) == len(self.actions)
            )
        else:
            actions_valid = not self.actions
        failed = self.status in (ChatStatus.ERROR, ChatStatus.INVALID_INPUT)
        model_answered = self.status in (ChatStatus.ANSWERED, ChatStatus.WITHHELD)
        consistent = (
            actions_valid
            and failed == (self.error is not None)
            and (not model_answered or (self.output is not None and self.prompt is not None))
            and (
                self.status is not ChatStatus.WITHHELD
                or OutputIssue.INSTRUCTION_LEAK in self.output_issues
            )
        )
        if not consistent:
            raise ValueError(f"inconsistent {self.status.value} response: {self!r}")

    def to_dict(self, *, include_system_message: bool = False) -> dict[str, Any]:
        """Return a JSON-serializable record of this response.

        The system message is identified by its SHA-256; its text is included
        only when ``include_system_message`` is true.
        """
        prompt: dict[str, Any] | None = None
        if self.prompt is not None:
            prompt = {
                "user_message": self.prompt.user_message,
                "system_instruction_sha256": self.prompt.system_instruction_sha256,
                "knowledge_sha256": self.prompt.knowledge_sha256,
                "system_message_sha256": self.prompt.system_message_sha256,
            }
            if include_system_message:
                prompt["system_message"] = self.prompt.system_message
        continuation = None
        if self.continuation is not None:
            continuation = {
                "route_id": self.continuation.route_id,
                "reasons": [reason.value for reason in self.continuation.reasons],
            }
        error = None
        if self.error is not None:
            error = {
                "code": self.error.code.value,
                "message": self.error.message,
                "http_status": self.error.http_status,
                "elapsed_ms": self.error.elapsed_ms,
            }
        return {
            "status": self.status.value,
            "answer_text": self.answer_text,
            "cloud_preview": self.cloud_preview,
            "continuation": continuation,
            "actions": [{"type": action.type.value, "route_id": action.route_id} for action in self.actions],
            "model": self.model,
            "message": self.message,
            "route": None if self.route is None else route_to_dict(self.route),
            "prompt": prompt,
            "request": None if self.request is None else copy.deepcopy(dict(self.request)),
            "output": None if self.output is None else output_to_dict(self.output),
            "output_issues": [issue.value for issue in self.output_issues],
            "error": error,
            "configs": dict(self.configs),
        }


@dataclass(frozen=True, slots=True)
class Routing:
    """Steps 1 and 2 for one message."""

    message: str | None  # the validated message; None when input validation failed
    route: RouteResult | None  # None when input validation failed
    response: ChatResponse | None  # set when the application answers without a model


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    """A message that passed steps 1 to 4 and needs the model."""

    model: ModelConfig
    message: str
    route: RouteResult
    knowledge: KnowledgeBlock
    prompt: BuiltPrompt
    payload: Mapping[str, Any]  # exactly what step 5 sends to /api/chat


class ChatEngine:
    """Answers visitor messages with one configured model at a time."""

    def __init__(
        self,
        *,
        models: ModelsConfig,
        router: Router,
        chat: ChatConfig,
        instruction: SystemInstruction,
        knowledge_loader: Callable[[], KnowledgeBlock],
        client: OllamaClient,
    ) -> None:
        route_ids = router.config.route_ids
        missing = [route_id for route_id in route_ids if route_id not in chat.route_labels]
        unknown = [route_id for route_id in chat.route_labels if route_id not in route_ids]
        if missing or unknown:
            raise ConfigError(
                f"{chat.source}: route_labels must name exactly the routes of "
                f"{router.config.source}; missing: {', '.join(missing) or 'none'}; "
                f"not routes: {', '.join(unknown) or 'none'}"
            )
        for model in models.models:
            check_options(model.options, where=f"{models.source}: {model.tag}")
        knowledge_loader()  # fail fast; the block is loaded again for every request

        self._models = models
        self._router = router
        self._chat = chat
        self._instruction = instruction
        self._knowledge_loader = knowledge_loader
        self._client = client
        self._approved_routes = frozenset(route_ids)
        self._validator = OutputValidator(instruction, link_placeholder=chat.replies.link_removed)
        self._configs = MappingProxyType(
            {
                "models": models.sha256,
                "routes": router.config.sha256,
                "chat": chat.sha256,
                "system_instruction": instruction.sha256,
            }
        )

    @classmethod
    def from_files(
        cls,
        *,
        client: OllamaClient | None = None,
        models_path: str | os.PathLike[str] | None = None,
        routes_path: str | os.PathLike[str] | None = None,
        chat_path: str | os.PathLike[str] | None = None,
        instruction_path: str | os.PathLike[str] | None = None,
        knowledge_path: str | os.PathLike[str] | None = None,
    ) -> "ChatEngine":
        """Build an engine from the project files (each path has a default)."""
        return cls(
            models=load_models_config(models_path),
            router=Router.from_file(routes_path),
            chat=load_chat_config(chat_path),
            instruction=load_system_instruction(instruction_path),
            knowledge_loader=functools.partial(load_knowledge, knowledge_path),
            client=client if client is not None else OllamaClient(),
        )

    @property
    def models(self) -> ModelsConfig:
        return self._models

    @property
    def router(self) -> Router:
        return self._router

    @property
    def chat_config(self) -> ChatConfig:
        return self._chat

    @property
    def instruction(self) -> SystemInstruction:
        return self._instruction

    @property
    def client(self) -> OllamaClient:
        return self._client

    @property
    def configs(self) -> Mapping[str, str]:
        """SHA-256 of each configuration file and the shared instruction."""
        return self._configs

    def load_knowledge(self) -> KnowledgeBlock:
        """Load the bounded knowledge block now, as step 3 does."""
        return self._knowledge_loader()

    def model(self, key: str | ModelConfig) -> ModelConfig:
        """Return the configured model whose exact tag, or name, is ``key``."""
        wanted = key.tag if isinstance(key, ModelConfig) else key
        if not isinstance(wanted, str):
            raise TypeError(f"model must be a tag or name, not {type(wanted).__name__}")
        for model in self._models.models:
            if model.tag == wanted:
                return model
        for model in self._models.models:
            if model.name.casefold() == wanted.casefold():
                return model
        raise ConfigError(
            f"{wanted!r} is not a configured model; configured tags: {', '.join(self._models.tags)}"
        )

    def respond(self, message: str, model: str | ModelConfig) -> ChatResponse:
        """Handle one message with one configured model (steps 1 to 6)."""
        prepared = self.prepare(message, model)
        if isinstance(prepared, ChatResponse):
            return prepared
        return self.send(prepared)

    def route(self, message: str) -> Routing:
        """Run steps 1 and 2 only; never loads knowledge or calls a model."""
        outcome = self._front(message, model=None)
        if isinstance(outcome, ChatResponse):
            return Routing(message=outcome.message, route=outcome.route, response=outcome)
        text, route = outcome
        return Routing(message=text, route=route, response=None)

    def prepare(self, message: str, model: str | ModelConfig) -> ChatResponse | PreparedRequest:
        """Run steps 1 to 4.

        Returns the final response when no model is needed (or the knowledge
        cannot be loaded), and otherwise the request that step 5 would send.
        """
        selected = self.model(model)
        outcome = self._front(message, model=selected.tag)
        if isinstance(outcome, ChatResponse):
            return outcome
        text, route = outcome
        try:
            knowledge = self._knowledge_loader()
        except KnowledgeError as exc:
            return self._response(
                ChatStatus.ERROR,
                self._chat.replies.unavailable,
                model=selected.tag,
                message=text,
                route=route,
                error=ChatError(ErrorCode.KNOWLEDGE_UNAVAILABLE, str(exc)),
            )
        prompt = PromptBuilder(self._instruction, knowledge).build(text)
        payload = self._client.chat_payload(selected, prompt.chat_messages())
        return PreparedRequest(
            model=selected,
            message=text,
            route=route,
            knowledge=knowledge,
            prompt=prompt,
            payload=MappingProxyType(payload),
        )

    def send(self, prepared: PreparedRequest) -> ChatResponse:
        """Run steps 5 and 6 for a request from ``prepare``."""
        settings = {key: value for key, value in prepared.payload.items() if key != "messages"}
        settings["timeout_s"] = self._client.timeout_s
        record: dict[str, Any] = {
            "model": prepared.model.tag,
            "message": prepared.message,
            "route": prepared.route,
            "prompt": prepared.prompt,
            "request": MappingProxyType(settings),
        }
        try:
            reply = self._client.send_chat(prepared.payload)
        except OllamaError as exc:
            error = ChatError(ErrorCode(exc.code.value), exc.message, exc.http_status, exc.elapsed_ms)
            return self._response(ChatStatus.ERROR, self._chat.replies.unavailable, error=error, **record)

        issues: list[OutputIssue] = []
        if reply.thinking is not None and reply.thinking.strip():
            issues.append(OutputIssue.THINKING_RETURNED)
        checked = self._validator.check(reply.content, knowledge=prepared.knowledge)
        issues.extend(checked.issues)
        limit_reached = reply.done_reason == "length"
        if limit_reached:
            issues.append(OutputIssue.OUTPUT_LIMIT_REACHED)

        if checked.withheld:
            return self._response(
                ChatStatus.WITHHELD, self._chat.replies.withheld, output=reply, issues=issues, **record
            )
        if not checked.text:
            issues.append(OutputIssue.EMPTY_OUTPUT)
            reason = "the model's answer has no displayable text"
            if limit_reached:
                reason += "; generation stopped at the output limit"
            error = ChatError(ErrorCode.EMPTY_OUTPUT, reason, elapsed_ms=reply.elapsed_ms)
            return self._response(
                ChatStatus.ERROR,
                self._chat.replies.unavailable,
                output=reply,
                issues=issues,
                error=error,
                **record,
            )
        return self._response(
            ChatStatus.ANSWERED,
            checked.text,
            output=reply,
            issues=issues,
            limit_reached=limit_reached,
            **record,
        )

    def _front(self, message: str, *, model: str | None) -> ChatResponse | tuple[str, RouteResult]:
        """Steps 1 and 2: a final response, or the validated text and its route."""
        if not isinstance(message, str):
            raise TypeError(f"message must be str, not {type(message).__name__}")
        text = clean_input(message)
        replies = self._chat.replies
        limit = self._chat.max_input_characters
        if not text:
            error = ChatError(ErrorCode.EMPTY_INPUT, "the message is empty")
            return self._response(ChatStatus.INVALID_INPUT, replies.empty_input, model=model, error=error)
        if len(text) > limit:
            error = ChatError(
                ErrorCode.INPUT_TOO_LONG, f"the message has {len(text)} characters; the limit is {limit}"
            )
            return self._response(ChatStatus.INVALID_INPUT, replies.input_too_long, model=model, error=error)

        route = self._router.route(text)
        actions: tuple[NavigationAction, ...] = ()
        match route.status:
            case RouteStatus.NOT_NAVIGATION:
                return text, route
            case RouteStatus.ROUTED if route.route_id is not None:
                status = ChatStatus.ROUTED
                answer = replies.routed.replace("{label}", self._chat.route_labels[route.route_id])
                actions = (NavigationAction(ActionType.NAVIGATE, route.route_id),)
            case RouteStatus.AMBIGUOUS_DESTINATION:
                status = ChatStatus.AMBIGUOUS_DESTINATION
                labels = join_alternatives([self._chat.route_labels[c] for c in route.candidates])
                answer = replies.ambiguous_destination.replace("{labels}", labels)
                actions = tuple(NavigationAction(ActionType.SUGGEST, c) for c in route.candidates)
            case RouteStatus.UNKNOWN_DESTINATION:
                status, answer = ChatStatus.UNKNOWN_DESTINATION, replies.unknown_destination
            case RouteStatus.URL_REJECTED:
                status, answer = ChatStatus.URL_REJECTED, replies.url_rejected
            case _:
                raise ValueError(f"unexpected route result {route!r}")
        return self._response(status, answer, actions=actions, model=model, message=text, route=route)

    def _response(
        self,
        status: ChatStatus,
        answer_text: str,
        *,
        actions: tuple[NavigationAction, ...] = (),
        model: str | None = None,
        message: str | None = None,
        route: RouteResult | None = None,
        prompt: BuiltPrompt | None = None,
        request: Mapping[str, Any] | None = None,
        output: ChatReply | None = None,
        issues: Sequence[OutputIssue] = (),
        error: ChatError | None = None,
        limit_reached: bool = False,
    ) -> ChatResponse:
        for action in actions:
            if action.route_id not in self._approved_routes:  # defense in depth
                raise ValueError(f"refusing an action for {action.route_id!r}: not allowlisted")
        preview, cut = cloud_preview(
            answer_text,
            max_lines=self._chat.cloud_max_lines,
            line_characters=self._chat.cloud_line_characters,
        )
        reasons: list[ContinuationReason] = []
        if cut:
            reasons.append(ContinuationReason.EXCEEDS_CLOUD)
        if limit_reached:
            reasons.append(ContinuationReason.OUTPUT_LIMIT_REACHED)
        continuation = None
        if reasons:
            continuation = Continuation(self._chat.continuation_route, tuple(reasons))
        return ChatResponse(
            status=status,
            answer_text=answer_text,
            cloud_preview=preview,
            continuation=continuation,
            actions=actions,
            model=model,
            message=message,
            route=route,
            prompt=prompt,
            request=request,
            output=output,
            output_issues=tuple(sorted(set(issues), key=_ISSUE_ORDER.__getitem__)),
            error=error,
            configs=self._configs,
        )


def clean_input(message: str) -> str:
    """Return ``message`` as step 1 passes it on.

    Line endings become LF; control characters other than line breaks and
    tabs, and invisible format characters (Unicode category Cf), are
    removed; surrounding whitespace is trimmed.
    """
    text = normalize_newlines(message)
    text = "".join(
        char for char in text if char in "\n\t" or unicodedata.category(char) not in ("Cc", "Cf")
    )
    return text.strip()


def join_alternatives(items: Sequence[str]) -> str:
    """Join ``items`` as alternatives: "A", "A or B", "A, B, or C"."""
    if len(items) <= 2:
        return " or ".join(items)
    return f"{', '.join(items[:-1])}, or {items[-1]}"


def route_to_dict(route: RouteResult) -> dict[str, Any]:
    """Return a JSON-serializable form of a router result."""
    return {
        "status": route.status.value,
        "route_id": route.route_id,
        "candidates": list(route.candidates),
        "matched_phrases": list(route.matched_phrases),
    }


def output_to_dict(output: ChatReply) -> dict[str, Any]:
    """Return a JSON-serializable form of a raw model output."""
    return {
        "content": output.content,
        "thinking": output.thinking,
        "done_reason": output.done_reason,
        "model": output.model,
        "created_at": output.created_at,
        "elapsed_ms": output.elapsed_ms,
        "metrics": dataclasses.asdict(output.metrics),
    }
