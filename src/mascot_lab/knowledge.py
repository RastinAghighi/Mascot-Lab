"""Bounded loader for the approved mascot knowledge.

Only the text between these two marker lines of ``knowledge/facts.md`` may
reach a model::

    <!-- BEGIN MASCOT KNOWLEDGE -->
    <!-- END MASCOT KNOWLEDGE -->

The rest of that file (usage notes, the navigation registry, provenance and
the update protocol) is application metadata and is never returned. Prose may
name the markers outside the block, as the file's usage note does, but only
the standalone comment lines count as boundaries.

The loader fails closed: missing, duplicated, out-of-order, malformed, or
nested markers, or an empty block, raise ``KnowledgeError`` instead of
producing a partial or guessed block.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from mascot_lab._textfile import normalize_newlines, read_text_file, sha256_hex
from mascot_lab.errors import KnowledgeError
from mascot_lab.paths import KNOWLEDGE_PATH, display_path

BEGIN_MARKER = "<!-- BEGIN MASCOT KNOWLEDGE -->"
END_MARKER = "<!-- END MASCOT KNOWLEDGE -->"

# A boundary: the HTML comment alone on its line. Whitespace around the line
# and inside the comment delimiters is tolerated; the marker words are not.
_MARKER_LINE = re.compile(r"^\s*<!--\s*(BEGIN|END) MASCOT KNOWLEDGE\s*-->\s*$")

# Any mention of a marker phrase, in any letter case or spacing.
_MARKER_PHRASE = re.compile(r"\b(?:BEGIN|END)\s+MASCOT\s+KNOWLEDGE\b", re.IGNORECASE)

# A line holding only a marker phrase and punctuation, such as a lowercase,
# unterminated, or heading-style marker. That is a broken boundary, not prose.
_PHRASE_ONLY_LINE = re.compile(r"^\W*(?:BEGIN|END)\s+MASCOT\s+KNOWLEDGE\W*$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class KnowledgeBlock:
    """Approved knowledge text plus the SHA-256 that identifies it in records.

    The hash covers the UTF-8 bytes of ``text`` exactly as injected, after
    line endings are normalized to LF.
    """

    text: str
    source: str = "<string>"
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise KnowledgeError(f"{self.source}: the knowledge block is empty")
        if _MARKER_PHRASE.search(self.text):
            raise KnowledgeError(f"{self.source}: the knowledge block contains marker text")
        object.__setattr__(self, "sha256", sha256_hex(self.text))


def extract_knowledge(document: str, *, source: str = "<string>") -> KnowledgeBlock:
    """Return the knowledge between the marker lines of ``document``.

    Blank lines directly inside the markers are dropped; everything else
    between them is kept unchanged, in file order.
    """
    lines = normalize_newlines(document).split("\n")
    begins: list[int] = []
    ends: list[int] = []
    mentions: list[int] = []
    for number, line in enumerate(lines, start=1):
        marker = _MARKER_LINE.match(line)
        if marker:
            (begins if marker.group(1) == "BEGIN" else ends).append(number)
        elif _MARKER_PHRASE.search(line):
            if "<!--" in line or "-->" in line or _PHRASE_ONLY_LINE.match(line):
                raise KnowledgeError(
                    f"{source}:{number}: malformed knowledge marker {line.strip()!r}; "
                    f"write {BEGIN_MARKER!r} or {END_MARKER!r} alone on a line"
                )
            mentions.append(number)  # prose naming a marker, allowed only outside the block

    begin = _single_marker(begins, BEGIN_MARKER, source)
    end = _single_marker(ends, END_MARKER, source)
    if end < begin:
        raise KnowledgeError(
            f"{source}: {END_MARKER!r} on line {end} precedes {BEGIN_MARKER!r} on line {begin}"
        )
    for number in mentions:
        if begin < number < end:
            raise KnowledgeError(f"{source}:{number}: marker text inside the knowledge block")

    body = lines[begin : end - 1]  # 1-based marker lines, so this is strictly between them
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    if not body:
        raise KnowledgeError(f"{source}: no knowledge between lines {begin} and {end}")
    return KnowledgeBlock(text="\n".join(body), source=source)


def load_knowledge(path: str | os.PathLike[str] | None = None) -> KnowledgeBlock:
    """Load the bounded knowledge block from ``path`` (default: ``knowledge/facts.md``).

    The file is only read, never written.
    """
    file_path = KNOWLEDGE_PATH if path is None else Path(path)
    document = read_text_file(file_path, error=KnowledgeError, label="knowledge file")
    return extract_knowledge(document, source=display_path(file_path))


def _single_marker(numbers: list[int], marker: str, source: str) -> int:
    if not numbers:
        raise KnowledgeError(f"{source}: missing {marker!r} line")
    if len(numbers) > 1:
        where = ", ".join(map(str, numbers))
        raise KnowledgeError(f"{source}: {marker!r} appears {len(numbers)} times (lines {where})")
    return numbers[0]
