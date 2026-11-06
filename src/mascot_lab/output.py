"""Output control: the display text derived from a model's raw answer.

The raw answer stays unchanged in the response; this module derives the
text a visitor would see and the thought cloud's preview of it. It never
creates navigation: actions come only from the router, so nothing a model
writes can navigate or add a link.

``OutputValidator.check`` applies these steps in order:

1. Hidden reasoning in ``<think>`` or ``<thinking>`` tags is removed,
   including reasoning cut off before its closing tag.
2. Links and contact details are replaced with a placeholder: URLs,
   ``www.`` hosts, email and IP addresses, social media handles, phone
   numbers, file paths, Markdown and HTML links (their link text is kept),
   and domain names that do not appear in the approved knowledge. A name
   that does appear there, such as ``rastinaghighi.dev`` or ``Three.js``,
   stays as plain text, never as a link.
3. Formatting is removed: HTML tags, Markdown emphasis, code marks,
   headings, list and quote markers, and invisible control or format
   characters. Runs of whitespace become single spaces.
4. A text that repeats ``LEAK_WORDS`` (12) or more consecutive words of the
   shared instruction, outside its knowledge block, is withheld. Only
   verbatim copies are detected, not paraphrases.

``cloud_preview`` approximates the thought cloud's "about 3 short lines"
(design plan, p. 7) at the text level by word-wrapping. The real fit
depends on the cloud's width and font and must be checked visually.
"""

import functools
import re
import textwrap
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum

from mascot_lab.knowledge import KnowledgeBlock
from mascot_lab.prompt import KNOWLEDGE_PLACEHOLDER, SystemInstruction

LEAK_WORDS = 12
ELLIPSIS = "…"


class OutputIssue(StrEnum):
    """Something output validation found or changed, in reporting order."""

    THINKING_RETURNED = "thinking_returned"  # Ollama returned reasoning separately; never shown
    THINKING_REMOVED = "thinking_removed"  # reasoning tags were removed from the answer text
    LINK_REMOVED = "link_removed"
    FORMATTING_REMOVED = "formatting_removed"
    INSTRUCTION_LEAK = "instruction_leak"  # the answer copies the instruction and is withheld
    OUTPUT_LIMIT_REACHED = "output_limit_reached"  # generation stopped at num_predict
    EMPTY_OUTPUT = "empty_output"  # no display text remained


@dataclass(frozen=True, slots=True)
class CheckedOutput:
    """The result of output validation for one raw answer."""

    text: str  # display text; empty when withheld or when nothing displayable remains
    issues: tuple[OutputIssue, ...]
    withheld: bool


# Reasoning blocks, closing tags without an opening tag (the opening tag was
# part of the prompt template), and opening tags cut off before closing.
_THINK_BLOCK = re.compile(r"<(think|thinking)\b[^<>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_THINK_CLOSE = re.compile(r"</(?:think|thinking)\s*>", re.IGNORECASE)
_THINK_OPEN = re.compile(r"<(?:think|thinking)\b[^<>]*>", re.IGNORECASE)

_MARKDOWN_LINK = re.compile(r"!?\[([^\[\]]*)\]\([^()\s]*(?:\s+\"[^\"]*\")?\s*\)")
_AUTOLINK = re.compile(r"<(?:[a-z][a-z0-9+.-]*:|www\.)[^<>\s]*>", re.IGNORECASE)
_HTML_TAG = re.compile(r"</?([A-Za-z][A-Za-z0-9-]*)(?:\s[^<>]*)?/?>")
_LINK_ATTRIBUTE = re.compile(r"\b(?:href|src|action)\s*=", re.IGNORECASE)
# Tags that separate text, so removing them leaves a space; other tags leave nothing.
_BLOCK_TAGS = frozenset(
    {"br", "p", "div", "li", "ul", "ol", "dl", "dt", "dd", "hr", "pre", "blockquote", "table"}
    | {"tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6"}
)

# Addresses that are never displayed, whatever the knowledge says.
_ADDRESSES = re.compile(
    r"\b(?:https?|ftps?|sftp|wss?|file|mailto|tel|sms|data|javascript|vbscript|blob|intent):\S+"
    r"|\bwww\.\S+"
    r"|[\w.+-]+@[\w-]+(?:\.[\w-]+)+"  # an email address
    r"|(?<![\w@])@[A-Za-z0-9_][A-Za-z0-9_.]*[A-Za-z0-9_]"  # a social media handle
    r"|\b\d{1,3}(?:\.\d{1,3}){3}\b(?::\d+)?(?:/\S*)?"  # an IPv4 address
    r"|(?<![\w+])\+?\d(?:[ .()-]{0,2}\d){9,14}(?![\w])"  # a phone number: 10 to 15 digits
    r"|\b[A-Za-z]:\\\S*"  # a Windows path
    r"|\\\\\S+"  # a network path
    r"|(?<![\w/.~-])~?/[\w.~-]+(?:/[\w.~-]*)*",  # a path such as /projects/qmind
    re.IGNORECASE,
)
# A domain name. The top-level label must be lowercase, so a missing space
# after a sentence ("University.He") is not taken for one. Group 2 is a path.
_DOMAIN = re.compile(
    r"(?<![\w@.-])((?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[a-z]{2,24})(?![\w-])"
    r"((?::\d+)?/\S*)?"
)
_TRAILING_PUNCTUATION = ".,;:!?)]}'\"’”»"
_OPENING_BRACKETS = {")": "(", "]": "[", "}": "{"}

_RULE_LINE = re.compile(r"^[ \t]*(?:[-*_][ \t]*){3,}$", re.MULTILINE)
_LINE_MARKER = re.compile(
    r"^[ \t]{0,3}(?:#{1,6}[ \t]+|>[ \t]?|[-*+•][ \t]+|\d{1,2}[.)][ \t]+)", re.MULTILINE
)
_MARKS = re.compile(r"\*\*|__|`")
_SINGLE_STAR = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])")
_INVISIBLE = frozenset({"Cc", "Cf"})

_WORD = re.compile(r"[^\W_]+")


class OutputValidator:
    """Derives display text from raw model answers for one shared instruction."""

    def __init__(
        self, instruction: SystemInstruction, *, link_placeholder: str, leak_words: int = LEAK_WORDS
    ) -> None:
        if leak_words < 2:
            raise ValueError("leak_words must be at least 2")
        before, after = instruction.template.split(KNOWLEDGE_PLACEHOLDER)
        self._leak_words = leak_words
        self._instruction_runs = _runs(_words(before), leak_words) | _runs(_words(after), leak_words)
        self._placeholder = link_placeholder

    def check(self, content: str, *, knowledge: KnowledgeBlock) -> CheckedOutput:
        """Validate one raw answer against the knowledge block it was given."""
        found: set[OutputIssue] = set()

        text = _remove_thinking(content)
        if text != content:
            found.add(OutputIssue.THINKING_REMOVED)

        # Links in markup first, so that what remains of the markup cannot look like a path.
        before = text
        text = _MARKDOWN_LINK.sub(lambda match: match.group(1), text)
        text = _AUTOLINK.sub(self._placeholder, text)
        if text != before:
            found.add(OutputIssue.LINK_REMOVED)
        text = _HTML_TAG.sub(lambda match: self._remove_tag(match, found), text)

        before = text
        allowed, knowledge_runs = _knowledge_profile(knowledge.text, self._leak_words)
        text = _ADDRESSES.sub(self._redact, text)
        text = _DOMAIN.sub(
            lambda match: match.group(0)
            if match.group(1).casefold() in allowed and not match.group(2)
            else self._redact(match),
            text,
        )
        if text != before:
            found.add(OutputIssue.LINK_REMOVED)

        before = text
        text = _remove_formatting(text)
        if text != before:
            found.add(OutputIssue.FORMATTING_REMOVED)
        text = " ".join(text.split())

        # Repeating approved facts is fine, so runs shared with the knowledge are allowed.
        leaks = self._instruction_runs - knowledge_runs
        withheld = any(run in leaks for run in _iter_runs(_words(text), self._leak_words))
        if withheld:
            found.add(OutputIssue.INSTRUCTION_LEAK)
            text = ""
        issues = tuple(issue for issue in OutputIssue if issue in found)
        return CheckedOutput(text=text, issues=issues, withheld=withheld)

    def _remove_tag(self, match: re.Match[str], found: set[OutputIssue]) -> str:
        if _LINK_ATTRIBUTE.search(match.group(0)):
            found.add(OutputIssue.LINK_REMOVED)
        else:
            found.add(OutputIssue.FORMATTING_REMOVED)
        return " " if match.group(1).lower() in _BLOCK_TAGS else ""

    def _redact(self, match: re.Match[str]) -> str:
        """Replace an address with the placeholder, keeping the sentence's own punctuation."""
        found = match.group(0)
        end = len(found)
        while end > 0 and found[end - 1] in _TRAILING_PUNCTUATION:
            opener = _OPENING_BRACKETS.get(found[end - 1])
            if opener is not None and found.count(opener, 0, end) >= found.count(found[end - 1], 0, end):
                break  # a balanced bracket belongs to the address, as in "alert(1)"
            end -= 1
        return self._placeholder + found[end:]


def domain_names(text: str) -> frozenset[str]:
    """Return the domain-like names in ``text``, case-folded (``rastinaghighi.dev``)."""
    return frozenset(match.group(1).casefold() for match in _DOMAIN.finditer(text))


@functools.lru_cache(maxsize=8)
def _knowledge_profile(
    text: str, leak_words: int
) -> tuple[frozenset[str], frozenset[tuple[str, ...]]]:
    """The domain-like names and word runs of a knowledge text, computed once per text."""
    return domain_names(text), _runs(_words(text), leak_words)


def cloud_preview(text: str, *, max_lines: int, line_characters: int) -> tuple[str, bool]:
    """Return the start of ``text`` that fits the cloud, and whether text was cut.

    Text that fits ``max_lines`` word-wrapped lines of ``line_characters``
    is returned unchanged. Longer text is cut at a word boundary and ends
    with an ellipsis, so the preview never exceeds the lines.
    """
    lines = textwrap.wrap(text, width=line_characters, break_on_hyphens=False)
    if len(lines) <= max_lines:
        return text, False
    kept = lines[:max_lines]
    last = kept[-1]
    while len(last) + len(ELLIPSIS) > line_characters and " " in last:
        last = last.rsplit(" ", 1)[0]
    last = last[: line_characters - len(ELLIPSIS)].rstrip(" ,;:.-–—")
    kept[-1] = last + ELLIPSIS
    return " ".join(line for line in kept if line), True


def _remove_thinking(text: str) -> str:
    cleaned = _THINK_BLOCK.sub(" ", text)
    closing = list(_THINK_CLOSE.finditer(cleaned))
    if closing:
        cleaned = cleaned[closing[-1].end() :]
    opening = _THINK_OPEN.search(cleaned)
    if opening:
        cleaned = cleaned[: opening.start()]
    return cleaned


def _remove_formatting(text: str) -> str:
    text = _RULE_LINE.sub("", text)
    text = _LINE_MARKER.sub("", text)
    text = _MARKS.sub("", text)
    text = _SINGLE_STAR.sub(lambda match: match.group(1), text)
    return "".join(
        char for char in text if char.isspace() or unicodedata.category(char) not in _INVISIBLE
    )


def _words(text: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(text.casefold()))


def _iter_runs(words: Sequence[str], size: int) -> Iterator[tuple[str, ...]]:
    for start in range(len(words) - size + 1):
        yield tuple(words[start : start + size])


def _runs(words: Sequence[str], size: int) -> frozenset[tuple[str, ...]]:
    return frozenset(_iter_runs(words, size))
