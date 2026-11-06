"""Deterministic navigation: direct commands to allowlisted semantic route IDs.

The router runs before any model and never calls one. It recognizes direct
navigation commands such as "take me to QMIND" or "show me Satira", resolves
their destination against the allowlist in ``configs/routes.yaml``, and
returns a structured ``RouteResult``. A routed result carries a semantic
route ID from that allowlist and nothing else: never a URL, a path, or text
taken from the message. Mapping route IDs to real pages is left to the
website, whose paths are not decided.

A message is a navigation command when, after optional openers such as
"please" or "could you", it starts with a command:

* a go command ("take me to", "go to", "navigate to", ...) always asks to
  move, so a destination that is not allowlisted is an unknown destination;
* a show command ("show me", "open", "visit", ...) also starts ordinary
  questions ("show me what Satira does"), so it counts as navigation only
  when its destination is allowlisted, contains a URL, or ends with a page
  word such as "page" or "website". Otherwise the message is left for the
  model.

The destination must consist of allowlisted phrases, compared after
normalization (``mascot_lab.phrases``); nothing is guessed. Phrases joined by
"and", "or", or commas that name different routes, and ambiguous terms, give
an ambiguous result. A command that contains a URL, domain name, IP address,
email address, or path is refused, except an allowlisted bare domain name
standing alone ("rastinaghighi.dev").
"""

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

from mascot_lab.errors import ConfigError
from mascot_lab.phrases import (
    CONNECTORS,
    Phrase,
    normalize_text,
    phrase_key,
    phrase_set,
    strip_leading,
    strip_trailing,
    tokenize,
)
from mascot_lab.routes import Destination, RoutesConfig, load_routes_config

# Words that may come before a command: "please", "could you", "I'd like to".
_OPENERS = phrase_set(
    "please", "pls", "plz", "kindly", "just", "now", "so", "then", "hey", "hi", "hello", "ok",
    "okay", ",", ";", "can you", "could you", "would you", "will you", "can u", "could u",
    "would u", "can i", "could i", "may i", "i want to", "i need to", "i wanna",
    "i would like to", "i'd like to", "id like to", "let's", "lets", "let us", "go ahead and",
)

# Go commands always ask to move somewhere.
_GO_COMMANDS = phrase_set(
    "take me to", "take me back to", "bring me to", "bring me back to", "send me to", "get me to",
    "lead me to", "point me to", "direct me to", "redirect me to", "go to", "go back to",
    "go over to", "head to", "head over to", "navigate to", "navigate me to", "jump to",
    "skip to", "switch to", "return to",
)

# Show commands also begin ordinary questions ("show me what Satira does").
_SHOW_COMMANDS = phrase_set(
    "show me", "show", "let me see", "see", "view", "open", "open up", "pull up", "bring up",
    "visit", "check out",
)

# Words that may end a command: "please", "thanks", "now".
_CLOSERS = phrase_set(
    "please", "pls", "plz", "thanks", "thx", "thank you", "now", "right now", "asap", "for me",
    ",", ";",
)

# A show command whose destination ends with one of these words asks for a page.
_PAGE_WORDS = frozenset({"page", "webpage", "homepage", "section", "tab", "site", "website"})

_COMMANDS: dict[Phrase, bool] = {  # command -> whether it is a go command
    **dict.fromkeys(_SHOW_COMMANDS, False),
    **dict.fromkeys(_GO_COMMANDS, True),
}
_LONGEST_COMMAND = max(map(len, _COMMANDS))

# Signs of a URL, domain name, IP address, email address, or path in normalized text.
_URL_SIGNS = re.compile(
    r"[/\\]"  # a path, or the "//" of a URL
    r"|\bwww\."  # a "www." host name
    r"|[^\W_]\.[^\W\d_]{2}"  # a domain name such as example.com
    r"|\d{1,3}(?:\.\d{1,3}){3}"  # an IPv4 address
    r"|[^\s@]@[^\s@]"  # an email address, or user information in a URL
    r"|%[0-9a-f]{2}"  # percent-encoding
    r"|\blocalhost\b"
    r"|\b(?:https?|ftps?|sftp|wss?|file|data|javascript|vbscript|mailto|tel|sms|blob|intent)\s*:"
    r"|\]\s*\("  # a Markdown link
    r"|\b(?:href|src)\b"  # an HTML link attribute
)


class RouteStatus(StrEnum):
    """The outcome of routing one message."""

    ROUTED = "routed"  # navigate to ``route_id``
    NOT_NAVIGATION = "not_navigation"  # not a direct navigation command; leave it to the model
    UNKNOWN_DESTINATION = "unknown_destination"  # a navigation command for no allowlisted route
    AMBIGUOUS_DESTINATION = "ambiguous_destination"  # names several routes, see ``candidates``
    URL_REJECTED = "url_rejected"  # a navigation command containing a URL, address, or path


@dataclass(frozen=True, slots=True)
class RouteResult:
    """The structured result of routing one message.

    ``route_id`` is set only when the status is ``ROUTED``, and
    ``candidates`` (two or more route IDs, in configuration order) only when
    it is ``AMBIGUOUS_DESTINATION``. ``matched_phrases`` lists the aliases or
    ambiguous terms that matched, as written in the configuration. No field
    ever holds text from the message.
    """

    status: RouteStatus
    route_id: str | None = None
    candidates: tuple[str, ...] = ()
    matched_phrases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, RouteStatus):
            raise TypeError(f"status must be a RouteStatus, not {type(self.status).__name__}")
        if not isinstance(self.candidates, tuple) or not isinstance(self.matched_phrases, tuple):
            raise TypeError("candidates and matched_phrases must be tuples")
        if self.status is RouteStatus.ROUTED:
            consistent = (
                isinstance(self.route_id, str)
                and bool(self.route_id)
                and not self.candidates
                and bool(self.matched_phrases)
            )
        elif self.status is RouteStatus.AMBIGUOUS_DESTINATION:
            consistent = (
                self.route_id is None
                and len(self.candidates) >= 2
                and len(set(self.candidates)) == len(self.candidates)
                and bool(self.matched_phrases)
            )
        else:
            consistent = self.route_id is None and not self.candidates and not self.matched_phrases
        if not consistent:
            raise ValueError(f"inconsistent {self.status.value} result: {self!r}")

    @property
    def is_navigation(self) -> bool:
        """Whether the message was a navigation command, routed or refused."""
        return self.status is not RouteStatus.NOT_NAVIGATION


class Router:
    """Routes messages to the semantic route IDs of one validated allowlist."""

    def __init__(self, config: RoutesConfig) -> None:
        self._config = config
        self._order = {route_id: index for index, route_id in enumerate(config.route_ids)}
        self._domain_names = _standalone(config.domain_names)
        self._check_allowlist()

    @classmethod
    def from_file(cls, path: str | os.PathLike[str] | None = None) -> "Router":
        """Build a router from ``path`` (default: ``configs/routes.yaml``)."""
        return cls(load_routes_config(path))

    @property
    def config(self) -> RoutesConfig:
        return self._config

    def route(self, message: str) -> RouteResult:
        """Route one message. Deterministic and offline; any text gets a result."""
        if not isinstance(message, str):
            raise TypeError(f"message must be str, not {type(message).__name__}")
        tokens = strip_leading(tokenize(message), _OPENERS)
        command = _command(tokens)
        if command is None:
            return RouteResult(RouteStatus.NOT_NAVIGATION)
        size, go = command
        destination = strip_trailing(tokens[size:], _CLOSERS)
        if self._has_url(normalize_text(message)):
            return RouteResult(RouteStatus.URL_REJECTED)
        resolved = self._resolve(destination)
        if resolved is not None:
            return resolved
        if go or (destination and destination[-1] in _PAGE_WORDS):
            return RouteResult(RouteStatus.UNKNOWN_DESTINATION)
        return RouteResult(RouteStatus.NOT_NAVIGATION)

    def _resolve(self, destination: Phrase) -> RouteResult | None:
        """Resolve a destination made only of allowlisted phrases, or return None."""
        entries: list[Destination] = []
        for segment in _segments(destination):
            entry = self._config.destinations.get(phrase_key(segment))
            if entry is None:
                return None
            entries.append(entry)
        if not entries:
            return None
        named = {route_id for entry in entries for route_id in entry.route_ids}
        route_ids = tuple(sorted(named, key=self._order.__getitem__))
        phrases = tuple(dict.fromkeys(entry.phrase for entry in entries))
        if len(route_ids) == 1:
            return RouteResult(RouteStatus.ROUTED, route_id=route_ids[0], matched_phrases=phrases)
        return RouteResult(
            RouteStatus.AMBIGUOUS_DESTINATION, candidates=route_ids, matched_phrases=phrases
        )

    def _has_url(self, text: str) -> bool:
        if self._domain_names is not None:
            text = self._domain_names.sub(" ", text)
        return _URL_SIGNS.search(text) is not None

    def _check_allowlist(self) -> None:
        """Fail closed unless every configured phrase routes as configured.

        A phrase can be valid on its own and still be unusable in a command,
        for example an alias that is a word the router ignores ("please").
        """
        expected = [
            (alias, RouteResult(RouteStatus.ROUTED, route_id=route.id, matched_phrases=(alias,)))
            for route in self._config.routes
            for alias in route.aliases
        ]
        expected += [
            (
                term.term,
                RouteResult(
                    RouteStatus.AMBIGUOUS_DESTINATION,
                    candidates=term.route_ids,
                    matched_phrases=(term.term,),
                ),
            )
            for term in self._config.ambiguous_terms
        ]
        for phrase, result in expected:
            message = f"take me to {phrase}"
            actual = self.route(message)
            if actual != result:
                raise ConfigError(
                    f"{self._config.source}: {phrase!r} cannot be reached: {message!r} gives "
                    f"{actual.status.value}, not {result.status.value}; the phrase conflicts "
                    "with the router's command words"
                )


def _command(tokens: Phrase) -> tuple[int, bool] | None:
    """Return the length of the command that starts ``tokens`` and whether it is a go command."""
    for size in range(min(_LONGEST_COMMAND, len(tokens)), 0, -1):
        go = _COMMANDS.get(tokens[:size])
        if go is not None:
            return size, go
    return None


def _segments(tokens: Phrase) -> Iterator[Phrase]:
    """Split a destination at connectors such as "and", "or", and commas."""
    segment: list[str] = []
    for token in tokens:
        if token in CONNECTORS:
            if segment:
                yield tuple(segment)
            segment = []
        else:
            segment.append(token)
    if segment:
        yield tuple(segment)


def _standalone(names: tuple[str, ...]) -> re.Pattern[str] | None:
    """Match the allowlisted bare domain names where they stand alone.

    Such a name may appear in a command as written, but never as part of a
    longer host name, URL, or address.
    """
    if not names:
        return None
    alternatives = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
    return re.compile(rf"(?<![\w.@/\\:%-])(?:{alternatives})(?![\w@/\\:%-]|\.[^\W_])")
