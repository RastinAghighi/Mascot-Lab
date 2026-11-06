"""Default locations of the shared project files.

The package is used from its source checkout (``pip install -e .``), so the
project root is two directories above this file: ``<root>/src/mascot_lab``.
Every loader also accepts an explicit path.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"
KNOWLEDGE_PATH = PROJECT_ROOT / "knowledge" / "facts.md"
SYSTEM_INSTRUCTION_PATH = PROJECT_ROOT / "prompts" / "system.md"


def display_path(path: Path) -> str:
    """Return ``path`` relative to the project root when it lies inside it.

    Records then name project files (``knowledge/facts.md``) without embedding
    local directory names.
    """
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)
