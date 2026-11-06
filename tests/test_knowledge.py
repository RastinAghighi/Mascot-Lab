"""Tests for the bounded knowledge loader."""

import hashlib
import re

import pytest

from mascot_lab import KnowledgeBlock, KnowledgeError, extract_knowledge, load_knowledge
from mascot_lab.knowledge import BEGIN_MARKER, END_MARKER
from mascot_lab.paths import KNOWLEDGE_PATH

ENTRY = re.compile(r"^- \*\*([A-Z]\d{2})\.\*\*", re.MULTILINE)


def doc(*lines: str) -> str:
    return "\n".join(lines) + "\n"


# Mirrors the real file's layout: a usage note that names both markers in
# prose comes before the block, and application metadata follows it.
MINIMAL = doc(
    "# Title",
    "Inject only the material between `BEGIN MASCOT KNOWLEDGE` and `END MASCOT KNOWLEDGE`.",
    BEGIN_MARKER,
    "",
    "- Fact one.",
    "",
    "- Fact two.",
    "",
    END_MARKER,
    "## Routing table: application metadata",
)


# ------------------------------------------------------------- real facts.md


def test_real_block_is_exactly_the_text_between_the_marker_lines(knowledge: KnowledgeBlock) -> None:
    raw = KNOWLEDGE_PATH.read_text(encoding="utf-8")
    start = raw.index(BEGIN_MARKER + "\n") + len(BEGIN_MARKER + "\n")
    stop = raw.index("\n" + END_MARKER)
    assert knowledge.text == raw[start:stop].strip("\n")
    assert knowledge.source == "knowledge/facts.md"


def test_real_block_keeps_every_fact_entry_in_file_order(knowledge: KnowledgeBlock) -> None:
    raw = KNOWLEDGE_PATH.read_text(encoding="utf-8")
    entries = ENTRY.findall(raw)
    assert entries, "no fact entries found in facts.md"
    assert ENTRY.findall(knowledge.text) == entries
    assert len(set(entries)) == len(entries)


def test_real_block_excludes_everything_outside_the_markers(
    knowledge: KnowledgeBlock, outside_block_samples: tuple[str, ...]
) -> None:
    raw = KNOWLEDGE_PATH.read_text(encoding="utf-8")
    for sample in outside_block_samples:
        assert sample in raw, f"sample is no longer in facts.md, update conftest: {sample!r}"
        assert sample not in knowledge.text, sample


def test_real_block_hash_is_the_sha256_of_its_text(knowledge: KnowledgeBlock) -> None:
    assert knowledge.sha256 == hashlib.sha256(knowledge.text.encode("utf-8")).hexdigest()
    assert re.fullmatch(r"[0-9a-f]{64}", knowledge.sha256)
    assert load_knowledge().sha256 == knowledge.sha256


def test_loading_never_modifies_the_facts_file() -> None:
    before = KNOWLEDGE_PATH.read_bytes()
    load_knowledge()
    assert KNOWLEDGE_PATH.read_bytes() == before


# ---------------------------------------------------------- valid synthetic


def test_extracts_only_the_text_between_the_marker_lines() -> None:
    assert extract_knowledge(MINIMAL).text == "- Fact one.\n\n- Fact two."


def test_prose_naming_the_markers_is_not_a_boundary() -> None:
    # A plain substring search would treat the usage note as the block and
    # extract "` and `". Only standalone comment lines are boundaries.
    block = extract_knowledge(MINIMAL)
    assert "Inject only" not in block.text
    assert "Routing table" not in block.text


def test_line_endings_and_byte_order_mark_do_not_change_text_or_hash(tmp_path) -> None:
    lf = tmp_path / "lf.md"
    lf.write_bytes(MINIMAL.encode("utf-8"))
    crlf_bom = tmp_path / "crlf_bom.md"
    crlf_bom.write_bytes(MINIMAL.replace("\n", "\r\n").encode("utf-8-sig"))
    first, second = load_knowledge(lf), load_knowledge(crlf_bom)
    assert first.text == second.text == "- Fact one.\n\n- Fact two."
    assert first.sha256 == second.sha256


def test_marker_lines_tolerate_surrounding_whitespace() -> None:
    document = MINIMAL.replace(BEGIN_MARKER, "  <!--BEGIN MASCOT KNOWLEDGE-->\t").replace(
        END_MARKER, "<!--   END MASCOT KNOWLEDGE   -->  "
    )
    assert extract_knowledge(document).text == "- Fact one.\n\n- Fact two."


def test_inner_blank_lines_and_indentation_are_preserved() -> None:
    document = doc(BEGIN_MARKER, "", "    indented first line", "", "", "last line", "   ", END_MARKER)
    assert extract_knowledge(document).text == "    indented first line\n\n\nlast line"


# -------------------------------------------------------- malformed inputs

MALFORMED = [
    pytest.param(doc("- Fact."), "missing.*BEGIN", id="no-markers"),
    pytest.param(doc("- Fact.", END_MARKER), "missing.*BEGIN", id="missing-begin"),
    pytest.param(doc(BEGIN_MARKER, "- Fact."), "missing.*END", id="missing-end"),
    pytest.param(doc(END_MARKER, "- Fact.", BEGIN_MARKER), "precedes", id="end-before-begin"),
    pytest.param(
        doc(BEGIN_MARKER, "- A.", BEGIN_MARKER, "- B.", END_MARKER), "BEGIN.*2 times", id="duplicate-begin"
    ),
    pytest.param(
        doc(BEGIN_MARKER, "- A.", END_MARKER, "- B.", END_MARKER), "END.*2 times", id="duplicate-end"
    ),
    pytest.param(
        doc(BEGIN_MARKER, "- A.", END_MARKER, BEGIN_MARKER, "- B.", END_MARKER), "2 times", id="two-blocks"
    ),
    pytest.param(doc(BEGIN_MARKER, END_MARKER), "no knowledge", id="empty-block"),
    pytest.param(doc(BEGIN_MARKER, "", "   ", "\t", END_MARKER), "no knowledge", id="blank-block"),
    pytest.param(
        doc(BEGIN_MARKER, "- A.", "Stop at END MASCOT KNOWLEDGE later.", END_MARKER),
        "inside the knowledge block",
        id="marker-text-inside-block",
    ),
    pytest.param(doc(BEGIN_MARKER, "- A.", "<!-- end mascot knowledge -->"), "malformed", id="lowercase"),
    pytest.param(doc(BEGIN_MARKER, "- A.", "<!-- END MASCOT KNOWLEDGE"), "malformed", id="unterminated"),
    pytest.param(doc("BEGIN MASCOT KNOWLEDGE", "- A.", END_MARKER), "malformed", id="bare-phrase"),
    pytest.param(doc("## BEGIN MASCOT KNOWLEDGE", "- A.", END_MARKER), "malformed", id="heading"),
    pytest.param(doc(BEGIN_MARKER + " - A.", END_MARKER), "malformed", id="text-after-marker"),
    pytest.param(
        doc("<!-- BEGIN MASCOT KNOWLEDGE v2 -->", "- A.", END_MARKER), "malformed", id="extra-words"
    ),
    pytest.param(
        doc("<!-- BEGIN  MASCOT KNOWLEDGE -->", "- A.", END_MARKER), "malformed", id="double-space"
    ),
]


@pytest.mark.parametrize(("document", "message"), MALFORMED)
def test_malformed_boundaries_fail_closed(document: str, message: str) -> None:
    with pytest.raises(KnowledgeError, match=message):
        extract_knowledge(document)


@pytest.mark.parametrize("text", ["", " \n\t", "- A.\n" + END_MARKER, "- A. begin mascot knowledge"])
def test_knowledge_block_rejects_empty_or_marker_text(text: str) -> None:
    with pytest.raises(KnowledgeError):
        KnowledgeBlock(text)


def test_missing_file_fails_closed(tmp_path) -> None:
    with pytest.raises(KnowledgeError, match="not found"):
        load_knowledge(tmp_path / "facts.md")


def test_non_utf8_file_fails_closed(tmp_path) -> None:
    path = tmp_path / "facts.md"
    path.write_bytes(MINIMAL.encode("utf-8") + b"\xff")
    with pytest.raises(KnowledgeError, match="UTF-8"):
        load_knowledge(path)


def test_directory_path_fails_closed(tmp_path) -> None:
    with pytest.raises(KnowledgeError, match="cannot read"):
        load_knowledge(tmp_path)
