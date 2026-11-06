"""Tests for configs/chat.yaml and its strict loader."""

import re
from collections.abc import Callable
from typing import Any

import pytest
import yaml

from mascot_lab import ConfigError
from mascot_lab.chat_config import (
    REPLY_PLACEHOLDERS,
    ChatConfig,
    load_chat_config,
    parse_chat_config,
)
from mascot_lab.paths import KNOWLEDGE_PATH
from mascot_lab.routes import load_routes_config


@pytest.fixture(scope="module")
def chat() -> ChatConfig:
    return load_chat_config()


def valid_data() -> dict[str, Any]:
    """A small, valid synthetic configuration."""
    return {
        "schema_version": 1,
        "input": {"max_characters": 100},
        "cloud": {"max_lines": 3, "line_characters": 40},
        "continuation_route": "mascot_chat",
        "route_labels": {"project_a": "Project A", "project_b": "Project B"},
        "replies": {
            "routed": "Taking you to {label}.",
            "ambiguous_destination": "Which one, {labels}?",
            "unknown_destination": "I can't take you there.",
            "url_rejected": "I can't open web addresses.",
            "empty_input": "Please type a question.",
            "input_too_long": "That message is too long.",
            "unavailable": "Sorry, try again later.",
            "withheld": "Sorry, I can't share that.",
            "link_removed": "(link removed)",
        },
    }


def dump(data: object) -> str:
    return yaml.safe_dump(data, sort_keys=False)


def registry_route_ids() -> set[str]:
    """The semantic route IDs in the navigation registry table of knowledge/facts.md."""
    text = KNOWLEDGE_PATH.read_text(encoding="utf-8")
    return set(re.findall(r"^\|\s*`([a-z_]+)`\s*\|", text, flags=re.MULTILINE))


# ---------------------------------------------------------------- real config


def test_real_config_loads_with_its_documented_values(chat: ChatConfig) -> None:
    assert chat.source == "configs/chat.yaml"
    assert chat.max_input_characters == 1000
    assert (chat.cloud_max_lines, chat.cloud_line_characters) == (3, 45)
    assert chat.continuation_route == "mascot_chat"
    assert re.fullmatch(r"[0-9a-f]{64}", chat.sha256)


def test_labels_name_exactly_the_configured_routes(chat: ChatConfig) -> None:
    assert tuple(chat.route_labels) == load_routes_config().route_ids


def test_every_route_id_used_by_the_engine_is_in_the_facts_registry(chat: ChatConfig) -> None:
    used = set(load_routes_config().route_ids) | {chat.continuation_route}
    assert used <= registry_route_ids()
    assert "mascot_chat" in registry_route_ids()


def test_continuation_route_is_not_a_command_route(chat: ChatConfig) -> None:
    # Part 2 decision: visitors reach the mascot's page through the cloud's
    # continuation link or by clicking the mascot, not by a typed command.
    assert chat.continuation_route not in load_routes_config().route_ids


def test_replies_carry_only_their_own_placeholder(chat: ChatConfig) -> None:
    for name, placeholder in REPLY_PLACEHOLDERS.items():
        text = getattr(chat.replies, name)
        assert re.findall(r"\{[^{}]*\}", text) == ([placeholder] if placeholder else []), name


def test_loaded_config_is_read_only(chat: ChatConfig) -> None:
    with pytest.raises(TypeError):
        chat.route_labels["project_qmind"] = "Other"  # type: ignore[index]


def test_hash_ignores_comments_and_layout_but_tracks_values() -> None:
    text = dump(valid_data())
    base = parse_chat_config(text)
    relaid = parse_chat_config("# comment\n" + text.replace("max_characters: 100", "max_characters:   100"))
    changed = parse_chat_config(text.replace("max_characters: 100", "max_characters: 101"))
    assert relaid.sha256 == base.sha256
    assert changed.sha256 != base.sha256


# -------------------------------------------------------- invalid synthetic


def _set(path: str, value: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        *parents, last = path.split(".")
        target = data
        for key in parents:
            target = target[key]
        target[last] = value
        return data

    return mutate


def _delete(path: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        *parents, last = path.split(".")
        target = data
        for key in parents:
            target = target[key]
        del target[last]
        return data

    return mutate


INVALID = [
    pytest.param(lambda data: [data], "root must be a mapping", id="root-is-a-list"),
    pytest.param(_set("system_prompt", "Be nice."), "unknown key", id="unknown-root-key"),
    pytest.param(_delete("replies"), "missing required key.*replies", id="missing-replies"),
    pytest.param(_set("schema_version", 2), "schema_version", id="wrong-schema-version"),
    pytest.param(_set("input.max_characters", 0), "at least 1", id="zero-input-limit"),
    pytest.param(_set("input.max_characters", "1000"), "integer", id="string-input-limit"),
    pytest.param(_set("input.max_characters", True), "integer", id="boolean-input-limit"),
    pytest.param(_set("input.min_characters", 1), "unknown key", id="unknown-input-key"),
    pytest.param(_set("cloud.max_lines", 0), "at least 1", id="zero-lines"),
    pytest.param(_set("cloud.line_characters", 9), "at least 10", id="too-narrow-lines"),
    pytest.param(_set("cloud.line_characters", 45.0), "integer", id="float-line-width"),
    pytest.param(_set("continuation_route", "mascot chat"), "semantic route id", id="bad-route-id"),
    pytest.param(_set("continuation_route", "https://example.com"), "semantic route id", id="url-route"),
    pytest.param(_set("route_labels", {}), "must not be empty", id="no-labels"),
    pytest.param(_set("route_labels.Project-A", "A"), "semantic route id", id="bad-label-key"),
    pytest.param(_set("route_labels.project_a", "https://a.example"), "URL", id="url-label"),
    pytest.param(_set("route_labels.project_a", "A {label}"), "braces", id="label-with-braces"),
    pytest.param(_set("route_labels.project_a", 7), "non-empty string", id="numeric-label"),
    pytest.param(_delete("replies.withheld"), "missing required key.*withheld", id="missing-reply"),
    pytest.param(_set("replies.greeting", "Hi."), "unknown key", id="unknown-reply"),
    pytest.param(_set("replies.routed", "Taking you there."), "exactly once", id="routed-without-label"),
    pytest.param(
        _set("replies.routed", "{label} and {label}"), "exactly once", id="routed-with-two-labels"
    ),
    pytest.param(
        _set("replies.ambiguous_destination", "Which, {label}?"), "exactly once", id="wrong-placeholder"
    ),
    pytest.param(_set("replies.unavailable", "Sorry {name}."), "placeholders", id="unexpected-placeholder"),
    pytest.param(_set("replies.withheld", "Sorry }"), "placeholders or braces", id="stray-brace"),
    pytest.param(_set("replies.url_rejected", "See https://x.example"), "URL", id="reply-with-url"),
    pytest.param(_set("replies.url_rejected", "See www.example"), "URL", id="reply-with-www"),
    pytest.param(_set("replies.unavailable", "Mail me@x"), "URL", id="reply-with-address"),
    pytest.param(_set("replies.unavailable", "Try /help"), "URL", id="reply-with-path"),
    pytest.param(_set("replies.unavailable", "Line one\nLine two"), "single line", id="multi-line-reply"),
    pytest.param(_set("replies.unavailable", " Padded."), "whitespace", id="padded-reply"),
]


@pytest.mark.parametrize(("mutate", "message"), INVALID)
def test_invalid_configuration_fails_closed(
    mutate: Callable[[dict[str, Any]], object], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_chat_config(dump(mutate(valid_data())))


def test_a_date_value_fails_closed_instead_of_breaking_the_hash() -> None:
    text = dump(valid_data()).replace("Sorry, try again later.", "2026-11-06")
    with pytest.raises(ConfigError, match="non-empty string"):
        parse_chat_config(text)


def test_duplicate_keys_fail_closed() -> None:
    text = dump(valid_data()).replace(
        "  withheld: Sorry, I can't share that.\n",
        "  withheld: Sorry, I can't share that.\n  withheld: Other.\n",
    )
    with pytest.raises(ConfigError, match="duplicate key 'withheld'"):
        parse_chat_config(text)


def test_missing_file_fails_closed(tmp_path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_chat_config(tmp_path / "chat.yaml")
