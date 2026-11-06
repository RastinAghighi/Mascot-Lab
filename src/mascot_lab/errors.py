"""Exception types. Every loader fails closed by raising one of these."""


class MascotLabError(Exception):
    """Base class for Mascot-Lab loading and validation errors."""


class ConfigError(MascotLabError):
    """The model configuration is missing, malformed, or breaks the schema."""


class KnowledgeError(MascotLabError):
    """The knowledge file is unreadable or its knowledge boundaries are malformed."""


class PromptError(MascotLabError):
    """The shared system instruction or a prompt input is invalid."""


class CaseError(MascotLabError):
    """An evaluation case file is missing, malformed, or breaks the case schema."""
