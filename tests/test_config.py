"""Tests for configs/models.yaml and its strict loader."""

import dataclasses
import re
from collections.abc import Callable
from typing import Any

import pytest
import yaml

from mascot_lab import ConfigError, ModelsConfig, load_models_config, parse_models_config
from mascot_lab.paths import PROJECT_ROOT

# The six candidates exactly as listed in docs/slm-development-handoff.md, section 3.
EXPECTED_MODELS = [
    ("Granite 4.0 H Micro", "granite4:micro-h"),
    ("Gemma 2B Instruct (original)", "gemma:2b-instruct-q4_K_M"),
    ("Gemma 2B Instruct v1.1", "gemma:2b-instruct-v1.1-q4_K_M"),
    ("Qwen3.5 2B", "qwen3.5:2b-q4_K_M"),
    ("Phi-4-mini", "phi4-mini:3.8b-q4_K_M"),
    ("Granite 4.2 3B", "granite4.2:3b-q4_K_M"),
]


def valid_data() -> dict[str, Any]:
    """A small, valid synthetic configuration."""
    return {
        "schema_version": 1,
        "baseline": {"options": {"num_ctx": 4096}},
        "models": [
            {"name": "Model A", "tag": "family-a:1b-q4_K_M"},
            {"name": "Model B", "tag": "family-b:2b-q4_K_M"},
        ],
    }


def dump(data: object) -> str:
    return yaml.safe_dump(data, sort_keys=False)


# ---------------------------------------------------------------- real config


def test_config_lists_exactly_the_six_handoff_models(models_config: ModelsConfig) -> None:
    assert [(model.name, model.tag) for model in models_config.models] == EXPECTED_MODELS


def test_config_matches_the_model_table_in_the_handoff(models_config: ModelsConfig) -> None:
    handoff = (PROJECT_ROOT / "docs" / "slm-development-handoff.md").read_text(encoding="utf-8")
    table = re.findall(r"^\|\s*([^|`]+?)\s*\|\s*`([^`]+)`\s*\|", handoff, flags=re.MULTILINE)
    assert table == [(model.name, model.tag) for model in models_config.models]


def test_all_models_share_the_4096_context_baseline(models_config: ModelsConfig) -> None:
    assert dict(models_config.baseline_options) == {"num_ctx": 4096}
    for model in models_config.models:
        # Adding a documented exception must be a deliberate change to this test.
        assert model.override is None, model.tag
        assert dict(model.options) == {"num_ctx": 4096}, model.tag


def test_config_records_a_stable_hash_and_relative_source(models_config: ModelsConfig) -> None:
    assert re.fullmatch(r"[0-9a-f]{64}", models_config.sha256)
    assert load_models_config().sha256 == models_config.sha256
    assert models_config.source == "configs/models.yaml"


def test_loaded_config_is_read_only(models_config: ModelsConfig) -> None:
    with pytest.raises(TypeError):
        models_config.baseline_options["num_ctx"] = 1
    with pytest.raises(TypeError):
        models_config.models[0].options["num_ctx"] = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        models_config.models[0].tag = "other:tag"


# ---------------------------------------------------------- valid synthetic


def test_valid_config_file_loads(tmp_path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(dump(valid_data()), encoding="utf-8")
    config = load_models_config(path)
    assert config.tags == ("family-a:1b-q4_K_M", "family-b:2b-q4_K_M")
    assert config.schema_version == 1


def test_documented_override_applies_only_to_its_model() -> None:
    data = valid_data()
    data["models"][1]["overrides"] = {
        "reason": "Synthetic test exception.",
        "options": {"num_ctx": 2048},
    }
    first, second = parse_models_config(dump(data)).models
    assert first.override is None
    assert dict(first.options) == {"num_ctx": 4096}
    assert second.override is not None
    assert second.override.reason == "Synthetic test exception."
    assert dict(second.options) == {"num_ctx": 2048}


def test_hash_ignores_comments_and_layout_but_tracks_values() -> None:
    text = dump(valid_data())
    base = parse_models_config(text)
    relaid = parse_models_config("# a comment\n" + text.replace("num_ctx: 4096", "num_ctx:   4096  # note"))
    changed = parse_models_config(text.replace("num_ctx: 4096", "num_ctx: 2048"))
    assert relaid.sha256 == base.sha256
    assert changed.sha256 != base.sha256


def test_list_of_strings_option_is_allowed() -> None:
    data = valid_data()
    data["baseline"]["options"]["stop"] = ["\n\n"]
    assert parse_models_config(dump(data)).baseline_options["stop"] == ("\n\n",)


# -------------------------------------------------------- invalid synthetic


def _model(index: int, **changes: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        data["models"][index].update(changes)
        return data

    return mutate


def _without_model_key(key: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        del data["models"][0][key]
        return data

    return mutate


def _root(**changes: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda data: {**data, **changes}


def _baseline_options(options: dict[str, Any]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return _root(baseline={"options": options})


INVALID_STRUCTURES = [
    pytest.param(lambda data: [data], "root must be a mapping", id="root-is-a-list"),
    pytest.param(_root(system_prompt="Be nice."), "unknown key", id="unknown-root-key"),
    pytest.param(
        lambda data: {k: v for k, v in data.items() if k != "models"},
        "missing required key.*models",
        id="missing-models",
    ),
    pytest.param(_root(schema_version=2), "schema_version", id="wrong-schema-version"),
    pytest.param(_root(schema_version=True), "schema_version", id="boolean-schema-version"),
    pytest.param(_root(models=[]), "non-empty list", id="empty-model-list"),
    pytest.param(_root(models={"name": "Model A"}), "non-empty list", id="models-not-a-list"),
    pytest.param(_root(models=["family-a:1b"]), "must be a mapping", id="model-not-a-mapping"),
    pytest.param(_model(0, system_prompt="Answer differently."), "unknown key", id="per-model-prompt"),
    pytest.param(_model(0, knowledge="other.md"), "unknown key", id="per-model-knowledge"),
    pytest.param(_without_model_key("tag"), "missing required key.*tag", id="missing-tag"),
    pytest.param(_model(0, tag=12), "tag must be a non-empty string", id="numeric-tag"),
    pytest.param(_model(0, tag="gemma"), "explicit Ollama tag", id="implicit-tag"),
    pytest.param(_model(0, tag="gemma:latest"), "exact tag", id="latest-tag"),
    pytest.param(_model(0, tag="gemma: 2b"), "explicit Ollama tag", id="tag-with-space"),
    pytest.param(_model(0, tag=" gemma:2b "), "whitespace", id="padded-tag"),
    pytest.param(_model(0, name=" "), "name must be a non-empty string", id="blank-name"),
    pytest.param(_model(0, tag="family-b:2b-q4_K_M"), "duplicate tag", id="duplicate-tag"),
    pytest.param(_model(0, name="Model B"), "duplicate name", id="duplicate-name"),
    pytest.param(_root(baseline={}), "missing required key.*options", id="baseline-without-options"),
    pytest.param(_baseline_options({}), "must not be empty", id="empty-baseline-options"),
    pytest.param(_baseline_options({"temperature": 0}), "must set num_ctx", id="missing-num-ctx"),
    pytest.param(_baseline_options({"num_ctx": 0}), "positive integer", id="zero-num-ctx"),
    pytest.param(_baseline_options({"num_ctx": "4096"}), "positive integer", id="string-num-ctx"),
    pytest.param(_baseline_options({"num_ctx": 4096.0}), "positive integer", id="float-num-ctx"),
    pytest.param(_baseline_options({"num_ctx": True}), "positive integer", id="boolean-num-ctx"),
    pytest.param(
        _baseline_options({"num_ctx": 4096, "Num-Predict": 5}), "invalid option name", id="bad-option-name"
    ),
    pytest.param(_baseline_options({"num_ctx": 4096, "seed": None}), "must be a boolean", id="null-option"),
    pytest.param(
        _baseline_options({"num_ctx": 4096, "temperature": float("nan")}), "finite", id="nan-option"
    ),
    pytest.param(
        _baseline_options({"num_ctx": 4096, "stop": []}), "must be a boolean", id="empty-list-option"
    ),
    pytest.param(
        _baseline_options({"num_ctx": 4096, "nested": {"a": 1}}), "must be a boolean", id="mapping-option"
    ),
    pytest.param(
        _model(0, overrides={"options": {"num_ctx": 2048}}),
        "missing required key.*reason",
        id="override-without-reason",
    ),
    pytest.param(
        _model(0, overrides={"reason": "  ", "options": {"num_ctx": 2048}}),
        "reason must be a non-empty string",
        id="override-with-blank-reason",
    ),
    pytest.param(
        _model(0, overrides={"reason": "Synthetic."}),
        "missing required key.*options",
        id="override-without-options",
    ),
    pytest.param(
        _model(0, overrides={"reason": "Synthetic.", "options": {"num_ctx": 2048}, "system": "x"}),
        "unknown key",
        id="override-of-the-prompt",
    ),
    pytest.param(
        _model(0, overrides={"reason": "Synthetic.", "options": {"num_ctx": -1}}),
        "positive integer",
        id="override-with-bad-num-ctx",
    ),
]


@pytest.mark.parametrize(("mutate", "message"), INVALID_STRUCTURES)
def test_invalid_structure_fails_closed(
    mutate: Callable[[dict[str, Any]], object], message: str
) -> None:
    text = dump(mutate(valid_data()))
    with pytest.raises(ConfigError, match=message):
        parse_models_config(text)


VALID_YAML = dump(valid_data())

INVALID_YAML = [
    pytest.param("", "root must be a mapping", id="empty-file"),
    pytest.param("models: [\n", "invalid YAML", id="syntax-error"),
    pytest.param(VALID_YAML + "---\nschema_version: 1\n", "invalid YAML", id="two-documents"),
    pytest.param(
        VALID_YAML.replace("  tag: family-a:1b-q4_K_M\n", "  tag: family-a:1b-q4_K_M\n  tag: other:1b\n"),
        "duplicate key 'tag'",
        id="duplicate-key",
    ),
    pytest.param(
        "defaults: &defaults\n  num_ctx: 4096\n"
        + VALID_YAML.replace("  options:\n    num_ctx: 4096\n", "  options:\n    <<: *defaults\n"),
        "merge keys",
        id="merge-key",
    ),
    pytest.param(VALID_YAML + "4096: num_ctx\n", "keys must be strings", id="non-string-key"),
]


@pytest.mark.parametrize(("text", "message"), INVALID_YAML)
def test_invalid_yaml_fails_closed(text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_models_config(text)


def test_missing_config_file_fails_closed(tmp_path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_models_config(tmp_path / "models.yaml")


def test_non_utf8_config_file_fails_closed(tmp_path) -> None:
    path = tmp_path / "models.yaml"
    path.write_bytes(dump(valid_data()).encode("utf-8") + b"\xff")
    with pytest.raises(ConfigError, match="UTF-8"):
        load_models_config(path)
