"""Output control: the display text derived from a model's raw answer.

The raw answer stays unchanged in the response; this module derives the
text a visitor would see and the thought cloud's preview of it. It never
creates navigation: actions come only from the router, so nothing a model
writes can navigate or add a link.

``OutputValidator.check`` applies these steps in order:

1. Hidden reasoning in ``<think>`` or ``<thinking>`` tags is removed,
   including reasoning cut off before its closing tag.
2. Markdown that step 4 does not cover loses its delimiters and keeps its
   text: a table becomes one line per row, with the cells separated by
   semicolons, and ``~~strikethrough~~`` and ``_emphasis_`` become plain
   words. Underscores inside words (``project_qmind``), single tildes and
   pipes outside a table stay. The prompt's internal annotations are
   removed too: the knowledge block's bracketed source tags (``[O]``,
   ``[D, O]``) and entry codes (``A01.``, ``[I01]``), the delimiter tags
   around the knowledge in the instruction (``<approved_facts>``) and the
   template placeholder (``{{MASCOT_KNOWLEDGE}}``), also escaped, in another
   letter case or spread over lines. The codes and tags are read from the
   knowledge block actually given and the delimiters from the instruction,
   so ``[1]``, ``[sic]`` and other codes stay. This step comes before the
   links step, so that nothing it removes can join the parts of an address,
   and it first removes invisible characters, which cannot hide an
   annotation from it.
3. Links and contact details are replaced with a placeholder: URLs,
   ``www.`` hosts, email and IP addresses, social media handles, phone
   numbers, file paths, Markdown and HTML links (their link text is kept),
   and domain names that do not appear in the approved knowledge. A name
   that does appear there, such as ``rastinaghighi.dev`` or ``Three.js``,
   stays as plain text, never as a link.
4. Formatting is removed: HTML tags, Markdown emphasis, code marks,
   headings, list and quote markers, and invisible control or format
   characters. Runs of whitespace become single spaces.
5. A text that repeats ``LEAK_WORDS`` (12) or more consecutive words of the
   shared instruction, outside its knowledge block, is withheld. Only
   verbatim copies are detected, not paraphrases. Two texts are checked:
   the text derived without step 2, as before that step existed, so that
   removing an annotation can never hide a copy; and the display text, so
   that a copy with annotations scattered through it is found too.

``cloud_preview`` approximates the thought cloud's "about 3 short lines"
(design plan, p. 7) at the text level by word-wrapping. The real fit
depends on the cloud's width and font and must be checked visually.
"""

import functools
import re
import textwrap
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
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

# Step 2, Markdown. A table is a row with an unescaped pipe, then a delimiter
# row of pipes, dashes and colons only (``|---|:--:|``), then the rows up to
# the first line without a pipe. Unlike GitHub, a delimiter row with another
# number of cells than the header is accepted, since such a row is never
# prose. Code fences are left alone. Strikethrough is exactly two tildes on
# each side, and emphasis underscores cannot touch a letter or digit outside
# them, so ``snake_case`` and ``~5`` never match.
_TABLE_PIPE = re.compile(r"(?<!\\)\|")
_TABLE_DELIMITER_CELL = re.compile(r":?-+:?")
_FENCE = re.compile(r"[ \t]*(?:```|~~~)")
_ROW_END = ".!?:;…"
_STRIKETHROUGH = re.compile(r"(?<!~)~~(?=[^\s~])([^\n]*?[^\s~])~~(?!~)")
_UNDERSCORE_EMPHASIS = re.compile(r"(?<![\w\\])_(?=[^\s_])([^\n]*?[^\s_\\])_(?!\w)")

# Step 2, annotations. Every pattern starts with the spaces before the
# annotation, which are dropped with it when punctuation follows ("[O].").
_BRACKET = re.compile(r"(?P<pre>[ \t]*)\\?\[(?P<body>[^\[\]]{1,200}?)\\?\]")
_ENTRY = re.compile(r"[ \t]*[-*+][ \t]+\*\*([A-Z]+[0-9]+)\.\*\*.*?(?:\[([^\[\]]+)\])?[ \t]*")
_OPENING_DELIMITER = re.compile(r"<\s*([A-Za-z_][\w.-]*)\s*>\s*\Z")
_CLOSING_DELIMITER = re.compile(r"\s*<\s*/\s*([A-Za-z_][\w.-]*)\s*>")
_ESCAPE = re.compile(r"\\(.)", re.DOTALL)
_DASHES = re.compile(r"[\u2010-\u2015\u2212]")
_RANGE_DASH = r"[ \t]*[-\u2010-\u2015\u2212][ \t]*"
_CLOSING_PUNCTUATION = frozenset(".,;:!?)]}…")

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
        self._instruction_parts = (before, after)
        self._placeholder = link_placeholder

    def check(self, content: str, *, knowledge: KnowledgeBlock) -> CheckedOutput:
        """Validate one raw answer against the knowledge block it was given."""
        found: set[OutputIssue] = set()

        text = _remove_thinking(content)
        if text != content:
            found.add(OutputIssue.THINKING_REMOVED)

        allowed, knowledge_runs = _knowledge_profile(knowledge.text, self._leak_words)
        annotations, simplified_instruction_runs, simplified_knowledge_runs = _annotation_profile(
            *self._instruction_parts, knowledge.text, self._leak_words
        )
        # The text as derived before step 2 existed, which only the leak check reads.
        unsimplified = self._display_text(text, allowed, set())
        before = text
        text = _simplify(text, annotations)
        if text != before:
            found.add(OutputIssue.FORMATTING_REMOVED)
        text = self._display_text(text, allowed, found)

        # Repeating approved facts is fine, so runs shared with the knowledge are allowed.
        leaks = self._instruction_runs - knowledge_runs
        simplified_leaks = (self._instruction_runs | simplified_instruction_runs) - (
            knowledge_runs | simplified_knowledge_runs
        )
        withheld = self._repeats(unsimplified, leaks) or self._repeats(text, simplified_leaks)
        if withheld:
            found.add(OutputIssue.INSTRUCTION_LEAK)
            text = ""
        issues = tuple(issue for issue in OutputIssue if issue in found)
        return CheckedOutput(text=text, issues=issues, withheld=withheld)

    def _display_text(self, text: str, allowed: frozenset[str], found: set[OutputIssue]) -> str:
        """Steps 3 and 4, recording what they change in ``found``."""
        # Links in markup first, so that what remains of the markup cannot look like a path.
        before = text
        text = _MARKDOWN_LINK.sub(lambda match: match.group(1), text)
        text = _AUTOLINK.sub(self._placeholder, text)
        if text != before:
            found.add(OutputIssue.LINK_REMOVED)
        text = _HTML_TAG.sub(lambda match: self._remove_tag(match, found), text)

        before = text
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
        return " ".join(text.split())

    def _repeats(self, text: str, leaks: frozenset[tuple[str, ...]]) -> bool:
        return any(run in leaks for run in _iter_runs(_words(text), self._leak_words))

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


@dataclass(frozen=True, slots=True)
class _Annotations:
    """The internal annotations of one instruction and knowledge block, as patterns."""

    structure: re.Pattern[str]  # the delimiter tags and the placeholder
    tags: frozenset[str]  # the knowledge's source tags, as _tag_text gives them but without spaces
    led: re.Pattern[str] | None  # a bracket's text that starts with a source letter or an entry code
    codes: tuple[re.Pattern[str], ...]  # entry codes: as labels, at the start of a line, and inline

    def remove(self, text: str) -> str:
        text = self.structure.sub(_drop, text)
        text = _BRACKET.sub(self._bracket, text)
        for pattern in self.codes:
            text = pattern.sub(_drop, text)
        return text

    def _bracket(self, match: re.Match[str]) -> str:
        body = _tag_text(match.group("body"))
        if body.replace(" ", "") in self.tags or (self.led is not None and self.led.match(body)):
            return _drop(match)
        return match.group(0)


@functools.lru_cache(maxsize=8)
def _annotation_profile(
    before: str, after: str, knowledge: str, leak_words: int
) -> tuple[_Annotations, frozenset[tuple[str, ...]], frozenset[tuple[str, ...]]]:
    """The annotations of an instruction and a knowledge text, and the word runs of each after step 2.

    ``before`` and ``after`` are the instruction around its placeholder.
    """
    annotations = _annotations(before, after, knowledge)

    def runs(text: str) -> frozenset[tuple[str, ...]]:
        return _runs(_words(_simplify(text, annotations)), leak_words)

    return annotations, runs(before) | runs(after), runs(knowledge)


def _annotations(before: str, after: str, knowledge: str) -> _Annotations:
    """Read the entry codes and source tags from the knowledge and the delimiters from the instruction.

    An entry is a list item that starts with a bold code, ``- **A01.** ...``;
    its source tag is the bracket that ends its line, ``[O]``. A source letter
    is a single capital that starts a tag or one of its ``;`` or ``,`` parts.
    The delimiters are the tags directly around the placeholder.
    """
    codes: set[str] = set()
    tags: set[str] = set()
    letters: set[str] = set()
    for line in knowledge.splitlines():
        entry = _ENTRY.fullmatch(line)
        if entry is None:
            continue
        codes.add(entry.group(1))
        if entry.group(2) is not None:
            tags.add(_tag_text(entry.group(2)).replace(" ", ""))
            for part in re.split(r"[;,]", entry.group(2)):
                first = part.split()[:1]
                if first and len(first[0]) == 1 and "A" <= first[0] <= "Z":
                    letters.add(first[0].casefold())

    delimiters = (_OPENING_DELIMITER.search(before), _CLOSING_DELIMITER.match(after))
    names = [match.group(1) for match in delimiters if match is not None]
    # Only a name with an underscore is removed on its own, so a delimiter named like a word is not.
    identifiers = {KNOWLEDGE_PLACEHOLDER.strip("{}"), *(name for name in names if "_" in name)}
    words = "|".join(_identifier(name) for name in _longest_first(identifiers))
    forms = [
        rf"(?:\\?\{{){{1,3}}\s*(?:{words})(?!\w)\s*(?:\\?\}}){{0,3}}",  # {{MASCOT_KNOWLEDGE}}, { mascot_knowledge }
        rf"(?<![\w\\])(?:{words})(?!\w)(?:\\?\}}){{0,3}}",  # the name alone, as an identifier
    ]
    if names:
        tag_names = "|".join(_identifier(name) for name in _longest_first(names))
        forms.insert(0, rf"(?:<|&lt;|\\<)\s*(?:\\?/)?\s*(?:{tag_names})\s*(?:\\?/)?\s*(?:>|&gt;|\\>)")
    structure = re.compile(rf"(?P<pre>[ \t]*)(?:{'|'.join(forms)})", re.IGNORECASE)

    led = None
    code_patterns: tuple[re.Pattern[str], ...] = ()
    alternatives = "|".join(re.escape(code) for code in _longest_first(codes))
    code_range = rf"(?:{alternatives})(?:{_RANGE_DASH}(?:{alternatives}))?"
    leaders = [code_range.casefold()] if codes else []
    if letters:
        leaders.append(f"[{''.join(sorted(letters))}]")
    if leaders:
        led = re.compile(rf"(?:{'|'.join(leaders)})(?:$|\s?[,;/&+]|\s(?:and|for)\b)")
    if codes:
        marks = r"(?:\\?[*_])"
        code_patterns = tuple(
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in (
                rf"(?P<pre>[ \t]*)(?<![\w\\]){marks}{{1,3}}(?:{alternatives})\\?[.:]{marks}{{1,3}}",  # **A01.**
                rf"^(?P<pre>[ \t]*(?:[-*+•][ \t]+)?)(?:{alternatives})\\?[.:](?=\s|$)",  # A01. at a line start
                rf"(?P<pre>[ \t]*)(?<![\w\\]){marks}{{0,3}}(?:\\?\({code_range}\\?\)|{code_range}){marks}{{0,3}}"
                rf"(?!\w|\.\d)",  # A01, (I01), Q01–Q03
            )
        )
    return _Annotations(structure=structure, tags=frozenset(tags), led=led, codes=code_patterns)


def _longest_first(names: Iterable[str]) -> list[str]:
    return sorted(set(names), key=lambda name: (-len(name), name))


def _identifier(name: str) -> str:
    """A pattern for ``name`` whose underscores may be escaped, as in ``approved\\_facts``."""
    return r"\\?_".join(re.escape(part) for part in name.split("_"))


def _tag_text(body: str) -> str:
    """A bracket's text compared without escapes, emphasis, dash variants, extra spaces or letter case."""
    body = _ESCAPE.sub(r"\1", body).replace("*", "").replace("`", "")
    return " ".join(_DASHES.sub("-", body).split()).casefold()


def _drop(match: re.Match[str]) -> str:
    """Remove an annotation, keeping the spaces before it only when a word follows."""
    following = match.string[match.end() : match.end() + 1]
    keep = bool(following) and not following.isspace() and following not in _CLOSING_PUNCTUATION
    return match.group("pre") if keep else ""


def _simplify(text: str, annotations: _Annotations) -> str:
    """Step 2: strikethrough and underscore emphasis, then annotations, then tables.

    Invisible characters go first, as step 4 would remove them, so that one
    inside ``[O]`` or ``A01`` cannot hide it here and reveal it there. Emphasis
    goes before annotations so that ``_[O]_`` or ``~~A01~~`` leaves no
    delimiters behind, and tables last so that a cell emptied of an
    annotation is dropped.
    """
    text = _visible(text)
    text = _STRIKETHROUGH.sub(lambda match: match.group(1), text)
    text = _UNDERSCORE_EMPHASIS.sub(lambda match: match.group(1), text)
    return _flatten_tables(annotations.remove(text))


def _flatten_tables(text: str) -> str:
    """Each Markdown table outside a code fence becomes one plain line per row."""
    lines = text.split("\n")
    result: list[str] = []
    fenced = False
    index = 0
    while index < len(lines):
        line = lines[index]
        if _FENCE.match(line):
            fenced = not fenced
        header = None if fenced else _table_cells(line)
        if header is not None and index + 1 < len(lines) and _is_delimiter_row(lines[index + 1]):
            rows = [header]
            index += 2
            while index < len(lines) and (cells := _table_cells(lines[index])) is not None:
                rows.append(cells)
                index += 1
            result.extend(_table_line(cells) for cells in rows)
            continue
        result.append(line)
        index += 1
    return "\n".join(result)


def _table_cells(line: str) -> list[str] | None:
    """The cells of a table row, or None for a line without an unescaped pipe."""
    row = line.strip()
    if not _TABLE_PIPE.search(row):
        return None
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|") and not row.endswith("\\|"):
        row = row[:-1]
    return [cell.strip() for cell in _TABLE_PIPE.split(row)]


def _is_delimiter_row(line: str) -> bool:
    cells = _table_cells(line)
    return cells is not None and all(_TABLE_DELIMITER_CELL.fullmatch(cell) for cell in cells)


def _table_line(cells: list[str]) -> str:
    """One row as a sentence: its non-empty cells, an escaped pipe kept as a pipe."""
    row = "; ".join(cell.replace("\\|", "|") for cell in cells if cell)
    return row if not row or row[-1] in _ROW_END else row + "."


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
    return _visible(text)


def _visible(text: str) -> str:
    """``text`` without control or format characters, except whitespace."""
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
