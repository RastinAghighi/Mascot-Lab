"""The shared system instruction and prompt builder.

Every candidate model receives the same two chat messages for a question:

* a system message: ``prompts/system.md`` with its single
  ``{{MASCOT_KNOWLEDGE}}`` placeholder replaced by the bounded knowledge block;
* a user message: the question itself, never merged into the system message.

The builder takes no model-specific input, so the candidates cannot receive
different instructions. The instruction asks for grounded, brief answers; it
is not a security control, and resisting prompt injection still needs
server-side input and output checks.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from mascot_lab._textfile import normalize_newlines, read_text_file, sha256_hex
from mascot_lab.errors import PromptError
from mascot_lab.knowledge import KnowledgeBlock, load_knowledge
from mascot_lab.paths import SYSTEM_INSTRUCTION_PATH, display_path

KNOWLEDGE_PLACEHOLDER = "{{MASCOT_KNOWLEDGE}}"
_ANY_PLACEHOLDER = re.compile(r"\{\{.*?\}\}", re.DOTALL)


@dataclass(frozen=True, slots=True)
class SystemInstruction:
    """The shared instruction template and its SHA-256."""

    template: str
    source: str = "<string>"
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        count = self.template.count(KNOWLEDGE_PLACEHOLDER)
        if count != 1:
            raise PromptError(
                f"{self.source}: expected {KNOWLEDGE_PLACEHOLDER} exactly once, found {count}"
            )
        unknown = sorted(set(_ANY_PLACEHOLDER.findall(self.template)) - {KNOWLEDGE_PLACEHOLDER})
        if unknown:
            raise PromptError(f"{self.source}: unknown placeholder(s) {', '.join(unknown)}")
        if not self.template.replace(KNOWLEDGE_PLACEHOLDER, "").strip():
            raise PromptError(f"{self.source}: the instruction has no text besides the placeholder")
        object.__setattr__(self, "sha256", sha256_hex(self.template))


def load_system_instruction(path: str | os.PathLike[str] | None = None) -> SystemInstruction:
    """Load the shared instruction (default: ``prompts/system.md``).

    The whole file is the template; surrounding whitespace is trimmed.
    """
    file_path = SYSTEM_INSTRUCTION_PATH if path is None else Path(path)
    template = read_text_file(file_path, error=PromptError, label="system instruction").strip()
    return SystemInstruction(template=template, source=display_path(file_path))


@dataclass(frozen=True, slots=True)
class BuiltPrompt:
    """The messages for one request plus the hashes that identify their parts."""

    system_message: str
    user_message: str
    system_instruction_sha256: str
    knowledge_sha256: str
    system_message_sha256: str

    def chat_messages(self) -> list[dict[str, str]]:
        """Return the messages in the role/content form used by chat APIs."""
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.user_message},
        ]


class PromptBuilder:
    """Assembles the one prompt that every candidate model receives."""

    def __init__(self, instruction: SystemInstruction, knowledge: KnowledgeBlock) -> None:
        self._instruction = instruction
        self._knowledge = knowledge
        self._system_message = instruction.template.replace(KNOWLEDGE_PLACEHOLDER, knowledge.text)
        self._system_message_sha256 = sha256_hex(self._system_message)

    @classmethod
    def from_files(
        cls,
        *,
        instruction_path: str | os.PathLike[str] | None = None,
        knowledge_path: str | os.PathLike[str] | None = None,
    ) -> "PromptBuilder":
        """Load the shared instruction and the bounded knowledge block from disk."""
        return cls(load_system_instruction(instruction_path), load_knowledge(knowledge_path))

    @property
    def instruction(self) -> SystemInstruction:
        return self._instruction

    @property
    def knowledge(self) -> KnowledgeBlock:
        return self._knowledge

    @property
    def system_message(self) -> str:
        return self._system_message

    def build(self, question: str) -> BuiltPrompt:
        """Build the messages for ``question``.

        Surrounding whitespace is trimmed and line endings become LF; the
        question is otherwise passed through unchanged as the user message.
        """
        if not isinstance(question, str):
            raise TypeError(f"question must be str, not {type(question).__name__}")
        user_message = normalize_newlines(question).strip()
        if not user_message:
            raise PromptError("the question is empty")
        return BuiltPrompt(
            system_message=self._system_message,
            user_message=user_message,
            system_instruction_sha256=self._instruction.sha256,
            knowledge_sha256=self._knowledge.sha256,
            system_message_sha256=self._system_message_sha256,
        )
