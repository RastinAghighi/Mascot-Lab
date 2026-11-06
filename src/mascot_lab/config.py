"""Shared model configuration (``configs/models.yaml``).

One file lists the candidate Ollama tags and the single option baseline that
all of them share. A model may differ from the baseline only through an
``overrides`` block that states a ``reason``, so every exception is documented
in the configuration itself. Prompts and knowledge cannot be configured per
model: unknown keys are rejected at every level. Validation fails closed.
"""

import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml
from yaml.constructor import ConstructorError

from mascot_lab._textfile import read_text_file, sha256_hex
from mascot_lab.errors import ConfigError
from mascot_lab.paths import MODELS_CONFIG_PATH, display_path

SCHEMA_VERSION = 1

type OptionValue = bool | int | float | str | tuple[str, ...]

_ROOT_KEYS = frozenset({"schema_version", "baseline", "models"})
_BASELINE_KEYS = frozenset({"options"})
_MODEL_KEYS = frozenset({"name", "tag", "overrides"})
_MODEL_REQUIRED_KEYS = frozenset({"name", "tag"})
_OVERRIDE_KEYS = frozenset({"reason", "options"})

# An explicit Ollama tag: a model name, one colon, and a fixed tag.
_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*:[A-Za-z0-9][A-Za-z0-9._-]*$")
_OPTION_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class ModelOverride:
    """A documented, model-specific exception to the shared baseline."""

    reason: str
    options: Mapping[str, OptionValue]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """One candidate model and the options it runs with."""

    name: str
    tag: str
    options: Mapping[str, OptionValue]  # effective: baseline plus any documented override
    override: ModelOverride | None


@dataclass(frozen=True, slots=True)
class ModelsConfig:
    """The validated configuration for every candidate model."""

    schema_version: int
    baseline_options: Mapping[str, OptionValue]
    models: tuple[ModelConfig, ...]
    source: str
    sha256: str  # of the canonical JSON form, so comments and layout do not affect it

    @property
    def tags(self) -> tuple[str, ...]:
        return tuple(model.tag for model in self.models)


def load_models_config(path: str | os.PathLike[str] | None = None) -> ModelsConfig:
    """Load and validate ``path`` (default: ``configs/models.yaml``)."""
    file_path = MODELS_CONFIG_PATH if path is None else Path(path)
    text = read_text_file(file_path, error=ConfigError, label="model configuration")
    return parse_models_config(text, source=display_path(file_path))


def parse_models_config(text: str, *, source: str = "<string>") -> ModelsConfig:
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

    baseline = _mapping(root["baseline"], f"{source}: baseline")
    _check_keys(baseline, f"{source}: baseline", _BASELINE_KEYS, _BASELINE_KEYS)
    baseline_options = _options(baseline["options"], f"{source}: baseline.options")
    if "num_ctx" not in baseline_options:
        raise ConfigError(f"{source}: baseline.options must set num_ctx")

    raw_models = root["models"]
    if not isinstance(raw_models, list) or not raw_models:
        raise ConfigError(f"{source}: models must be a non-empty list")
    models: list[ModelConfig] = []
    for index, raw in enumerate(raw_models):
        where = f"{source}: models[{index}]"
        entry = _mapping(raw, where)
        _check_keys(entry, where, _MODEL_KEYS, _MODEL_REQUIRED_KEYS)
        name = _text(entry["name"], f"{where}.name")
        tag = _tag(entry["tag"], f"{where}.tag")
        if any(model.tag == tag for model in models):
            raise ConfigError(f"{where}: duplicate tag {tag!r}")
        if any(model.name == name for model in models):
            raise ConfigError(f"{where}: duplicate name {name!r}")
        override = None
        options = dict(baseline_options)
        if "overrides" in entry:
            override = _override(entry["overrides"], f"{where}.overrides")
            options.update(override.options)
        models.append(ModelConfig(name, tag, MappingProxyType(options), override))

    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return ModelsConfig(
        schema_version=version,
        baseline_options=MappingProxyType(baseline_options),
        models=tuple(models),
        source=source,
        sha256=sha256_hex(canonical),
    )


class _StrictLoader(yaml.SafeLoader):
    """A SafeLoader that rejects duplicate, non-string, and merge (``<<``) keys.

    PyYAML otherwise keeps the last of two duplicate keys without warning.
    """

    def construct_mapping(self, node: yaml.Node, deep: bool = False) -> dict[Any, Any]:
        if isinstance(node, yaml.MappingNode):
            seen: set[str] = set()
            for key_node, _ in node.value:
                if key_node.tag == "tag:yaml.org,2002:merge":
                    raise ConstructorError(
                        None, None, "merge keys ('<<') are not allowed", key_node.start_mark
                    )
                key = self.construct_object(key_node, deep=deep)
                if not isinstance(key, str):
                    raise ConstructorError(
                        None, None, f"keys must be strings, found {key!r}", key_node.start_mark
                    )
                if key in seen:
                    raise ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
                seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _mapping(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping")
    return value


def _check_keys(
    mapping: dict[str, Any], where: str, allowed: frozenset[str], required: frozenset[str]
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {', '.join(unknown)}; allowed: {', '.join(sorted(allowed))}"
        )
    missing = sorted(required - set(mapping))
    if missing:
        raise ConfigError(f"{where}: missing required key(s) {', '.join(missing)}")


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a non-empty string")
    if value != value.strip():
        raise ConfigError(f"{where} must not have leading or trailing whitespace")
    return value


def _tag(value: object, where: str) -> str:
    tag = _text(value, where)
    if not _TAG.fullmatch(tag):
        raise ConfigError(f"{where}: {tag!r} is not an explicit Ollama tag such as 'name:tag'")
    if tag.lower().endswith(":latest"):
        raise ConfigError(f"{where}: {tag!r} can change when re-pulled; use an exact tag")
    return tag


def _options(value: object, where: str) -> dict[str, OptionValue]:
    raw = _mapping(value, where)
    if not raw:
        raise ConfigError(f"{where} must not be empty")
    options: dict[str, OptionValue] = {}
    for name, item in raw.items():
        if not _OPTION_NAME.fullmatch(name):
            raise ConfigError(f"{where}: invalid option name {name!r}")
        options[name] = _option_value(item, f"{where}.{name}")
    if "num_ctx" in options:
        num_ctx = options["num_ctx"]
        if type(num_ctx) is not int or num_ctx <= 0:
            raise ConfigError(f"{where}.num_ctx must be a positive integer, found {num_ctx!r}")
    return options


def _option_value(value: object, where: str) -> OptionValue:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError(f"{where} must be a finite number")
        return value
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, list) and value and all(isinstance(item, str) and item for item in value):
        return tuple(value)
    raise ConfigError(
        f"{where} must be a boolean, number, string, or non-empty list of strings, found {value!r}"
    )


def _override(value: object, where: str) -> ModelOverride:
    raw = _mapping(value, where)
    _check_keys(raw, where, _OVERRIDE_KEYS, _OVERRIDE_KEYS)
    reason = _text(raw["reason"], f"{where}.reason")
    options = _options(raw["options"], f"{where}.options")
    return ModelOverride(reason=reason, options=MappingProxyType(options))
