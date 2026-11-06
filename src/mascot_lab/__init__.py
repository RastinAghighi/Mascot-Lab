"""Mascot-Lab: one shared local evaluation harness for the candidate models of
the planned rastinaghighi.dev mascot.

The foundation (Part 1) provides the model configuration, the bounded
knowledge loader, and the shared prompt builder, exported here. Deterministic
navigation (Part 2) is in ``mascot_lab.router``. The shared chat engine
(Part 3) is in ``mascot_lab.engine``, with its Ollama adapter in
``mascot_lab.ollama``, output validation in ``mascot_lab.output``, and the
command line in ``mascot_lab.cli`` (``python -m mascot_lab``). The evaluation
infrastructure (Part 4) is in ``mascot_lab.cases`` (the case file),
``mascot_lab.runner`` (planning and running), ``mascot_lab.measurements``,
``mascot_lab.recorder`` (the files under ``results/``), and
``mascot_lab.review`` (the blank human review sheet). Importing the package
never contacts Ollama. See ``docs/architecture.md``.
"""

from mascot_lab.config import (
    ModelConfig,
    ModelOverride,
    ModelsConfig,
    load_models_config,
    parse_models_config,
)
from mascot_lab.errors import CaseError, ConfigError, KnowledgeError, MascotLabError, PromptError
from mascot_lab.knowledge import KnowledgeBlock, extract_knowledge, load_knowledge
from mascot_lab.prompt import (
    BuiltPrompt,
    PromptBuilder,
    SystemInstruction,
    load_system_instruction,
)

__all__ = [
    "BuiltPrompt",
    "CaseError",
    "ConfigError",
    "KnowledgeBlock",
    "KnowledgeError",
    "MascotLabError",
    "ModelConfig",
    "ModelOverride",
    "ModelsConfig",
    "PromptBuilder",
    "PromptError",
    "SystemInstruction",
    "extract_knowledge",
    "load_knowledge",
    "load_models_config",
    "load_system_instruction",
    "parse_models_config",
]
