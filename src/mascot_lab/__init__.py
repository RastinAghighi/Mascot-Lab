"""Mascot-Lab: one shared local evaluation harness for the candidate models of
the planned rastinaghighi.dev mascot.

This foundation provides the model configuration, the bounded knowledge
loader, and the shared prompt builder. It never calls Ollama.
"""

from mascot_lab.config import (
    ModelConfig,
    ModelOverride,
    ModelsConfig,
    load_models_config,
    parse_models_config,
)
from mascot_lab.errors import ConfigError, KnowledgeError, MascotLabError, PromptError
from mascot_lab.knowledge import KnowledgeBlock, extract_knowledge, load_knowledge
from mascot_lab.prompt import (
    BuiltPrompt,
    PromptBuilder,
    SystemInstruction,
    load_system_instruction,
)

__all__ = [
    "BuiltPrompt",
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
