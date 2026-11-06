"""Helpers shared by the loaders: strict UTF-8 reading and SHA-256 hashing."""

import hashlib
from pathlib import Path

from mascot_lab.errors import MascotLabError


def normalize_newlines(text: str) -> str:
    """Convert CRLF and lone CR line endings to LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def sha256_hex(text: str) -> str:
    """Return the hexadecimal SHA-256 digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text_file(path: Path, *, error: type[MascotLabError], label: str) -> str:
    """Read a UTF-8 text file with LF line endings, or raise ``error``.

    A leading byte-order mark is dropped and line endings are normalized, so
    hashes do not depend on the platform or on Git line-ending settings.
    """
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise error(f"{label} not found: {path}") from exc
    except OSError as exc:
        raise error(f"cannot read {label} {path}: {exc}") from exc
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise error(f"{label} is not valid UTF-8: {path}: {exc}") from exc
    return normalize_newlines(text)
