"""Chat engine configuration (``configs/chat.yaml``).

One file holds the settings the chat engine shares across all models:

* ``input.max_characters``: the longest message accepted;
* ``cloud``: the text-level approximation of the thought cloud's "about 3
  short lines" (design plan, p. 7);
* ``continuation_route``: the semantic route of the mascot's page, where a
  long answer continues;
* ``route_labels``: the name used in replies for each navigation route;
* ``replies``: the fixed replies for outcomes handled without a model.

Labels and replies are plain single-line text, never a URL, address, or
path. Loading reuses the strict YAML loader and validators of
``mascot_lab.config`` and fails closed with ``ConfigError``.
"""

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

from mascot_lab._textfile import read_text_file, sha256_hex
from mascot_lab.config import _StrictLoader, _check_keys, _mapping, _text
from mascot_lab.errors import ConfigError
from mascot_lab.paths import PROJECT_ROOT, display_path
from mascot_lab.routes import _route_id

CHAT_CONFIG_PATH = PROJECT_ROOT / "configs" / "chat.yaml"
SCHEMA_VERSION = 1

_ROOT_KEYS = frozenset(
    {"schema_version", "input", "cloud", "continuation_route", "route_labels", "replies"}
)
_INPUT_KEYS = frozenset({"max_characters"})
_CLOUD_KEYS = frozenset({"max_lines", "line_characters"})

# A cloud line must hold a short word plus the preview's ellipsis.
MIN_LINE_CHARACTERS = 10

# Every reply, with the one placeholder it must contain (None: no placeholder).
REPLY_PLACEHOLDERS: Mapping[str, str | None] = MappingProxyType(
    {
        "routed": "{label}",
        "ambiguous_destination": "{labels}",
        "unknown_destination": None,
        "url_rejected": None,
        "empty_input": None,
        "input_too_long": None,
        "unavailable": None,
        "withheld": None,
        "link_removed": None,
    }
)

# Signs of a URL, an address, or a path, which labels and replies must not contain.
_ADDRESS_SIGNS = re.compile(r"://|\bwww\.|@|[/\\]", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"\{[^{}]*\}")


@dataclass(frozen=True, slots=True)
class Replies:
    """The fixed application replies; see ``configs/chat.yaml``."""

    routed: str  # contains {label}
    ambiguous_destination: str  # contains {labels}
    unknown_destination: str
    url_rejected: str
    empty_input: str
    input_too_long: str
    unavailable: str  # any failure to get a usable answer from the model
    withheld: str  # output validation refused to show the model's answer
    link_removed: str  # replaces a link removed from a model answer


@dataclass(frozen=True, slots=True)
class ChatConfig:
    """The validated chat engine configuration."""

    schema_version: int
    max_input_characters: int
    cloud_max_lines: int
    cloud_line_characters: int
    continuation_route: str
    route_labels: Mapping[str, str]  # in configuration order
    replies: Replies
    source: str
    sha256: str  # of the canonical JSON form, so comments and layout do not affect it


def load_chat_config(path: str | os.PathLike[str] | None = None) -> ChatConfig:
    """Load and validate ``path`` (default: ``configs/chat.yaml``)."""
    file_path = CHAT_CONFIG_PATH if path is None else Path(path)
    text = read_text_file(file_path, error=ConfigError, label="chat configuration")
    return parse_chat_config(text, source=display_path(file_path))


def parse_chat_config(text: str, *, source: str = "<string>") -> ChatConfig:
    """Validate configuration ``text``; raise ``ConfigError`` on any problem."""
    try:
        data = yaml.load(text, Loader=_StrictLoader)  # _StrictLoader subclasses SafeLoader
    except yaml.YAMLError as exc:
        raise ConfigError(f"{source}: invalid YAML: {exc}") from exc

    root = _mapping(data, f"{source}: configuration root")
    _check_keys(root, f"{source}: configuration root", _ROOT_KEYS, _ROOT_KEYS)
    version = root["schema_version"]
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ConfigError(f"{source}: schema_version must be {SCHEMA_VERSION}, found {version!r}")

    user_input = _mapping(root["input"], f"{source}: input")
    _check_keys(user_input, f"{source}: input", _INPUT_KEYS, _INPUT_KEYS)
    max_characters = _integer(user_input["max_characters"], f"{source}: input.max_characters", 1)

    cloud = _mapping(root["cloud"], f"{source}: cloud")
    _check_keys(cloud, f"{source}: cloud", _CLOUD_KEYS, _CLOUD_KEYS)
    max_lines = _integer(cloud["max_lines"], f"{source}: cloud.max_lines", 1)
    line_characters = _integer(
        cloud["line_characters"], f"{source}: cloud.line_characters", MIN_LINE_CHARACTERS
    )

    continuation_route = _route_id(root["continuation_route"], f"{source}: continuation_route")
    route_labels = _labels(root["route_labels"], f"{source}: route_labels")
    replies = _replies(root["replies"], f"{source}: replies")

    # Hashed only after validation, which leaves nothing that JSON cannot encode.
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return ChatConfig(
        schema_version=version,
        max_input_characters=max_characters,
        cloud_max_lines=max_lines,
        cloud_line_characters=line_characters,
        continuation_route=continuation_route,
        route_labels=route_labels,
        replies=replies,
        source=source,
        sha256=sha256_hex(canonical),
    )


def _integer(value: object, where: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{where} must be an integer of at least {minimum}, found {value!r}")
    return value


def _plain_text(value: object, where: str) -> str:
    text = _text(value, where)
    if "\n" in text or "\r" in text:
        raise ConfigError(f"{where} must be a single line")
    if _ADDRESS_SIGNS.search(text):
        raise ConfigError(f"{where}: {text!r} must not contain a URL, address, or path")
    return text


def _labels(value: object, where: str) -> Mapping[str, str]:
    raw = _mapping(value, where)
    if not raw:
        raise ConfigError(f"{where} must not be empty")
    labels: dict[str, str] = {}
    for key, item in raw.items():
        route_id = _route_id(key, f"{where}: key")
        label = _plain_text(item, f"{where}.{route_id}")
        if "{" in label or "}" in label:
            raise ConfigError(f"{where}.{route_id} must not contain braces")
        labels[route_id] = label
    return MappingProxyType(labels)


def _replies(value: object, where: str) -> Replies:
    raw = _mapping(value, where)
    names = frozenset(REPLY_PLACEHOLDERS)
    _check_keys(raw, where, names, names)
    texts: dict[str, str] = {}
    for name, placeholder in REPLY_PLACEHOLDERS.items():
        text = _plain_text(raw[name], f"{where}.{name}")
        found = _PLACEHOLDER.findall(text)
        rest = _PLACEHOLDER.sub("", text)
        expected = [placeholder] if placeholder else []
        if found != expected or "{" in rest or "}" in rest:
            if placeholder:
                raise ConfigError(
                    f"{where}.{name} must contain {placeholder} exactly once and no other braces"
                )
            raise ConfigError(f"{where}.{name} must not contain placeholders or braces")
        texts[name] = text
    return Replies(**texts)
