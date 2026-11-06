"""Shared fixtures.

Tests only read the real project files. Synthetic inputs are written under
pytest's ``tmp_path``, so knowledge/facts.md is never modified. No test
contacts a real Ollama server: model responses come from ``FakeOllama``
(``ollama_fakes.py``), and the HTTP transport is tested against a local
stand-in server.
"""

import pytest
from ollama_fakes import FakeOllama

from mascot_lab import (
    KnowledgeBlock,
    ModelsConfig,
    PromptBuilder,
    SystemInstruction,
    load_knowledge,
    load_models_config,
    load_system_instruction,
)

# Text that appears in knowledge/facts.md only outside the knowledge markers.
# test_knowledge.py checks that each sample really is present in the file.
OUTSIDE_BLOCK_SAMPLES = (
    "canonical mascot facts",
    "**Version:** 1.0",
    "Instructions for whoever uses this file",
    "MASCOT KNOWLEDGE",
    "Navigation registry",
    "project_qmind",
    "project_ingenuity",
    "project_satira",
    "project_deeplense",
    "project_website",
    "mascot_chat",
    "Explicitly absent, restricted, or undecided",
    "Provenance and confidence audit",
    "Disclosure note",
    "Update protocol",
)


@pytest.fixture(scope="session")
def outside_block_samples() -> tuple[str, ...]:
    return OUTSIDE_BLOCK_SAMPLES


@pytest.fixture(scope="session")
def models_config() -> ModelsConfig:
    return load_models_config()


@pytest.fixture(scope="session")
def knowledge() -> KnowledgeBlock:
    return load_knowledge()


@pytest.fixture(scope="session")
def instruction() -> SystemInstruction:
    return load_system_instruction()


@pytest.fixture(scope="session")
def builder(instruction: SystemInstruction, knowledge: KnowledgeBlock) -> PromptBuilder:
    return PromptBuilder(instruction, knowledge)


@pytest.fixture
def ollama() -> FakeOllama:
    """A scripted Ollama stand-in; it fails the test on any unscripted request."""
    return FakeOllama()
