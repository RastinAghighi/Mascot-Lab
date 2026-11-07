"""The thought-cloud preview and continuation in the form the public chat API requires.

The public chat API (rastinaghighi.dev, ``docs/api/chat-v1.md``, sections 7,
14 and 18.2, requirements R4 and R5) requires a shortened ``cloud_preview``
to be an exact start of ``answer_text`` followed by an ellipsis (U+2026).
The engine's own ``mascot_lab.output.cloud_preview`` joins its word-wrapped
lines with spaces, so it inserts a space wherever it split a word longer
than a cloud line, including in text without spaces such as Chinese or
Japanese. That preview stays as it is for the engine, the command line and
the evaluation records. ``public_preview`` is for the HTTP adapter, which
does not exist yet; nothing in the engine or the command line uses it.

``public_preview`` lays the text out exactly as the engine's preview does
(the same greedy wrap at ``cloud.line_characters`` code points per line),
but it records where each line starts and ends in the text instead of
copying the lines, so a shortened preview is a slice of the text:

* Text that fits ``cloud.max_lines`` lines is returned unchanged.
* Otherwise the lines before the last are kept whole and the last line is
  cut so that it still holds the ellipsis: at its last space, or within a
  word only when that word fills the whole line. A cut within a word moves
  back so that it does not split a user-perceived character, such as a
  letter and its combining accent or an emoji sequence. Spaces and the
  punctuation that the engine's preview also drops are removed from the end
  of the kept text.

The text is never normalized, and nothing in the kept part is inserted,
removed, replaced or reordered. The cut is the engine's, so where the
engine's preview is already an exact start of the text followed by the
ellipsis, both previews are equal, with two rare exceptions: a cut that the
engine places inside a user-perceived character moves back here, and when
the last line holds nothing but spaces and that punctuation, the engine
keeps a space before its ellipsis, which this preview drops as well.

The continuation reasons follow R5: ``exceeds_cloud`` exactly when the
preview is shortened, and ``output_limit_reached`` exactly when the caller
passes the engine's report that generation stopped at the output limit
(``reached_output_limit``). That report is never inferred from the text.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from mascot_lab.chat_config import MIN_LINE_CHARACTERS, ChatConfig
from mascot_lab.engine import ChatResponse, Continuation, ContinuationReason
from mascot_lab.errors import ConfigError
from mascot_lab.output import ELLIPSIS

# The response schema's maxLength of cloud_preview, and the const of continuation.route_id.
PUBLIC_PREVIEW_MAX_CHARACTERS = 137
PUBLIC_CONTINUATION_ROUTE = "mascot_chat"

# What ``output.cloud_preview`` strips from the end of the kept text before the ellipsis.
_TRAILING = " ,;:.-–—"
# textwrap's chunks: runs of spaces and runs of anything else.
_CHUNK = re.compile(r" +|[^ ]+")

_ZWJ = "‍"
_MARK_CATEGORIES = frozenset({"Mn", "Mc", "Me"})  # combining marks, including variation selectors
# Other characters that belong to the character before them: zero-width non-joiner and joiner,
# the Thai and Lao vowels AM, halfwidth katakana sound marks, emoji modifiers, and tag characters.
_ATTACHING = (
    (0x200C, 0x200D), (0x0E33, 0x0E33), (0x0EB3, 0x0EB3), (0xFF9E, 0xFF9F), (0x1F3FB, 0x1F3FF), (0xE0020, 0xE007F)
)
_REGIONAL_INDICATORS = ((0x1F1E6, 0x1F1FF),)
_HANGUL_LEADING = ((0x1100, 0x115F), (0xA960, 0xA97C))
_HANGUL_FOLLOWING = ((0x1160, 0x11FF), (0xD7B0, 0xD7FB))  # vowel and trailing consonant jamo
_HANGUL = _HANGUL_LEADING + _HANGUL_FOLLOWING + ((0xAC00, 0xD7A3),)
_VIRAMA = 9  # canonical combining class


@dataclass(frozen=True, slots=True)
class PublicPreview:
    """The ``cloud_preview`` and ``continuation`` of one public response."""

    cloud_preview: str
    continuation: Continuation | None

    def continuation_json(self) -> dict[str, Any] | None:
        """``continuation`` as the response body carries it: None, or a new dict."""
        if self.continuation is None:
            return None
        return {
            "route_id": self.continuation.route_id,
            "reasons": [reason.value for reason in self.continuation.reasons],
        }


def public_preview(answer_text: str, *, output_limit_reached: bool, chat: ChatConfig) -> PublicPreview:
    """Return the public preview and continuation of ``answer_text``.

    ``answer_text`` is the final display text, after output validation; it
    is not changed. ``output_limit_reached`` is the engine's report that
    generation stopped at the output limit (see ``reached_output_limit``).
    The cloud's size and the continuation route come from ``chat``; a
    configuration that would break the public API raises ``ConfigError``.
    """
    if not isinstance(answer_text, str):
        raise TypeError(f"answer_text must be str, not {type(answer_text).__name__}")
    if type(output_limit_reached) is not bool:
        raise TypeError(f"output_limit_reached must be bool, not {type(output_limit_reached).__name__}")
    if not answer_text.strip():
        raise ValueError("answer_text must contain a character other than whitespace")
    max_lines, width, budget = _cloud(chat)

    kept = _kept_length(answer_text, max_lines, width, budget)
    if kept is None:
        preview = answer_text
    else:
        # Guaranteed by _kept_length; checked so that a defect fails instead of breaking R4.
        if not 0 < kept < len(answer_text) or kept + len(ELLIPSIS) > budget:
            raise RuntimeError(f"invalid preview length {kept} for a text of {len(answer_text)}")
        preview = answer_text[:kept] + ELLIPSIS

    reasons: list[ContinuationReason] = []
    if kept is not None:
        reasons.append(ContinuationReason.EXCEEDS_CLOUD)
    if output_limit_reached:
        reasons.append(ContinuationReason.OUTPUT_LIMIT_REACHED)
    continuation = Continuation(chat.continuation_route, tuple(reasons)) if reasons else None
    return PublicPreview(cloud_preview=preview, continuation=continuation)


def reached_output_limit(response: ChatResponse) -> bool:
    """Whether the engine reported that generation stopped at the output limit.

    The engine gives this continuation reason only to an answered message
    whose Ollama reply had ``done_reason`` "length", as the public API
    allows; withheld answers and errors never carry it.
    """
    continuation = response.continuation
    return continuation is not None and ContinuationReason.OUTPUT_LIMIT_REACHED in continuation.reasons


def _cloud(chat: ChatConfig) -> tuple[int, int, int]:
    """The configured cloud: lines, characters per line, and the longest preview."""
    max_lines, width = chat.cloud_max_lines, chat.cloud_line_characters
    if max_lines < 1 or width < MIN_LINE_CHARACTERS:
        raise ConfigError(f"{chat.source}: invalid cloud of {max_lines} lines of {width} characters")
    budget = max_lines * width + max_lines - 1  # the lines and the spaces that join them
    if budget > PUBLIC_PREVIEW_MAX_CHARACTERS:
        raise ConfigError(
            f"{chat.source}: the cloud allows previews of {budget} characters, but the public "
            f"API allows {PUBLIC_PREVIEW_MAX_CHARACTERS}; its bound must be revised first"
        )
    if chat.continuation_route != PUBLIC_CONTINUATION_ROUTE:
        raise ConfigError(
            f"{chat.source}: continuation_route {chat.continuation_route!r} is not the public "
            f"API's {PUBLIC_CONTINUATION_ROUTE!r}"
        )
    return max_lines, width, budget


def _kept_length(text: str, max_lines: int, width: int, budget: int) -> int | None:
    """How many code points of ``text`` a shortened preview keeps; None when it fits."""
    lines = _wrap_spans(text, width)
    if len(lines) <= max_lines and len(text) <= budget:
        return None
    floor, limit = 0, budget - len(ELLIPSIS)
    if len(lines) > max_lines:
        # The lines before the last are kept whole; the last keeps room for the ellipsis.
        floor, end = lines[max_lines - 1]
        limit = min(limit, end, floor + width - len(ELLIPSIS))
    cut = text.rfind(" ", floor + 1, limit + 1)  # the last word boundary on the last line
    if cut == -1:  # one word fills the line, so the cut falls within it
        cut = _character_start(text, limit)
    kept = text[:cut].rstrip(_TRAILING)
    return len(kept) if kept else cut


def _wrap_spans(text: str, width: int) -> list[tuple[int, int]]:
    """The cloud's lines of ``text``, as ``(start, end)`` positions in it.

    This is ``textwrap.wrap(text, width, break_on_hyphens=False)``, which
    ``output.cloud_preview`` uses, step for step, but it keeps positions
    instead of copying text. Lines break at spaces, and a chunk longer than
    a whole line is split. textwrap also treats tabs and line breaks as
    spaces; display text contains neither.
    """
    chunks = [[match.start(), match.end()] for match in _CHUNK.finditer(text)]
    chunks.reverse()  # a stack, as in textwrap
    lines: list[tuple[int, int]] = []
    while chunks:
        line: list[tuple[int, int]] = []
        length = 0
        if lines and _blank(text, chunks[-1]):  # a line does not start with the space it broke at
            chunks.pop()
        while chunks and length + chunks[-1][1] - chunks[-1][0] <= width:
            start, end = chunks.pop()
            line.append((start, end))
            length += end - start
        if chunks and chunks[-1][1] - chunks[-1][0] > width:  # too long for any line: split it
            start = chunks[-1][0]
            split = start + width - length
            line.append((start, split))
            chunks[-1][0] = split
        if line and _blank(text, line[-1]):
            line.pop()
        if line:
            lines.append((line[0][0], line[-1][1]))
    return lines


def _blank(text: str, span: tuple[int, int] | list[int]) -> bool:
    return not text[span[0] : span[1]].strip()


def _character_start(text: str, cut: int) -> int:
    """``cut``, moved back so that it does not split a user-perceived character.

    When every earlier position would split one, as in a letter followed by
    hundreds of combining marks, ``cut`` is returned unchanged.
    """
    position = cut
    while position > 0 and _splits_character(text, position):
        position -= 1
    return position or cut


def _splits_character(text: str, position: int) -> bool:
    """Whether a cut before ``text[position]`` would split a user-perceived character.

    An approximation of the extended grapheme clusters of Unicode Standard
    Annex 29, which the standard library does not provide. Where it is
    unsure it keeps characters together, which only makes a preview shorter.
    Prepended characters are not handled.
    """
    before, after = text[position - 1], text[position]
    if unicodedata.category(after) in _MARK_CATEGORIES or _within(after, _ATTACHING):
        return True
    if before == _ZWJ:  # an emoji ZWJ sequence
        return True
    if _within(after, _REGIONAL_INDICATORS):  # flags are pairs of regional indicators
        run = 0
        while position - run > 0 and _within(text[position - run - 1], _REGIONAL_INDICATORS):
            run += 1
        return run % 2 == 1
    if _within(after, _HANGUL_FOLLOWING) or (_within(before, _HANGUL_LEADING) and _within(after, _HANGUL)):
        return True  # decomposed Hangul syllables
    # A virama joins consonants into a conjunct, as in Devanagari.
    return unicodedata.combining(before) == _VIRAMA and unicodedata.category(after).startswith("L")


def _within(char: str, ranges: tuple[tuple[int, int], ...]) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in ranges)
