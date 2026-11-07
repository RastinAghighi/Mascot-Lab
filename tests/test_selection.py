"""The provisional v1 selection record, configs/selection.yaml.

The record must name models by exact tags that configs/models.yaml
configures, must not claim a deployment, and must point to its document.
Recording a selection must not change the shared six-model configuration.
These tests read the real files only.
"""

import re
from typing import Any

import pytest
import yaml

from mascot_lab import ModelsConfig
from mascot_lab.config import _StrictLoader
from mascot_lab.paths import PROJECT_ROOT

SELECTION = PROJECT_ROOT / "configs" / "selection.yaml"
DOCUMENT = PROJECT_ROOT / "docs" / "model-selection-v1.md"


@pytest.fixture(scope="module")
def selection() -> dict[str, Any]:
    data = yaml.load(SELECTION.read_text(encoding="utf-8"), Loader=_StrictLoader)
    assert isinstance(data, dict)
    return data


def test_the_record_is_provisional_and_claims_no_deployment(selection: dict[str, Any]) -> None:
    assert selection["schema_version"] == 1
    assert selection["status"] == "provisional"
    assert selection["deployment"]["deployed"] is False
    assert selection["deployment"]["public_deployment_ready"] is False


@pytest.mark.parametrize("role", ["selected", "backup"])
def test_each_choice_is_a_configured_model_by_its_exact_tag(
    selection: dict[str, Any], models_config: ModelsConfig, role: str
) -> None:
    choice = selection[role]
    configured = {model.tag: model.name for model in models_config.models}
    assert configured.get(choice["tag"]) == choice["name"]
    assert re.fullmatch(r"[0-9a-f]{12}", str(choice["ollama_model_id"]))


def test_the_selected_and_backup_models_differ(selection: dict[str, Any]) -> None:
    assert selection["selected"]["tag"] != selection["backup"]["tag"]


def test_the_shared_configuration_still_has_six_models_and_no_overrides(models_config: ModelsConfig) -> None:
    assert len(models_config.models) == 6
    assert not any(model.override for model in models_config.models)


def test_the_document_names_both_choices_by_tag(selection: dict[str, Any]) -> None:
    assert selection["document"] == "docs/model-selection-v1.md"
    text = DOCUMENT.read_text(encoding="utf-8")
    for role in ("selected", "backup"):
        assert f"`{selection[role]['tag']}`" in text
