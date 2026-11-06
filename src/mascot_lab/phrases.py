"""Text normalization for navigation phrases.

The allowlist (``mascot_lab.routes``) and the router (``mascot_lab.router``)
compare text only through these functions, so a configured alias and a
visitor's message are always normalized the same way.

Normalization removes differences that do not change a name: letter case,
accents, full-width and other compatibility forms, invisible format
characters such as zero-width spaces, punctuation, and spacing. It never
corrects spelling: "Satire" does not become "Satira".
"""

import re
import unicodedata
from collections.abc import Sequence, Set

type Phrase = tuple[str, ...]

# A word (letters and digits, in any script) or a mark that can join a list of
# destinations. Every other character, including an apostrophe, separates words.
_TOKEN = re.compile(r"[^\W_]+|[,;&+|]")

# NFKC turns the full-width and small full stops into ".", but not this one.
_IDEOGRAPHIC_FULL_STOP = str.maketrans({"。": "."})


def normalize_text(text: str) -> str:
    """Return ``text`` in the form used for comparison.

    Compatibility forms such as full-width letters are folded (NFKC),
    invisible format characters such as zero-width spaces and bidirectional
    controls are removed, letter case is folded, and accents are dropped.
    """
    text = unicodedata.normalize("NFKC", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Cf")
    text = unicodedata.normalize("NFKD", text.casefold().translate(_IDEOGRAPHIC_FULL_STOP))
    text = "".join(char for char in text if not unicodedata.category(char).startswith("M"))
    return unicodedata.normalize("NFC", text)


def tokenize(text: str) -> Phrase:
    """Return the words of normalized ``text`` and its list marks (``, ; & + |``).

    Apostrophes separate words, so "QMIND's" gives ``("qmind", "s")`` and
    "let's" gives ``("let", "s")``.
    """
    return tuple(_TOKEN.findall(normalize_text(text)))


def phrase_set(*texts: str) -> frozenset[Phrase]:
    """Return the token form of each phrase in ``texts``."""
    return frozenset(tokenize(text) for text in texts)


def strip_leading(tokens: Sequence[str], phrases: Set[Phrase], *, keep: int = 0) -> Phrase:
    """Remove ``phrases`` from the start of ``tokens``, longest first, keeping ``keep`` tokens."""
    longest = max(map(len, phrases), default=0)
    start = 0
    while True:
        for size in range(min(longest, len(tokens) - start - keep), 0, -1):
            if tuple(tokens[start : start + size]) in phrases:
                start += size
                break
        else:
            return tuple(tokens[start:])


def strip_trailing(tokens: Sequence[str], phrases: Set[Phrase], *, keep: int = 0) -> Phrase:
    """Remove ``phrases`` from the end of ``tokens``, longest first, keeping ``keep`` tokens."""
    longest = max(map(len, phrases), default=0)
    stop = len(tokens)
    while True:
        for size in range(min(longest, stop - keep), 0, -1):
            if tuple(tokens[stop - size : stop]) in phrases:
                stop -= size
                break
        else:
            return tuple(tokens[:stop])


# Words and marks that join destinations, as in "QMIND or Satira".
CONNECTORS = frozenset({",", ";", "&", "+", "|", "and", "or", "then", "plus", "nor", "vs", "versus"})

# What a possessive "'s" leaves behind, since apostrophes separate words.
_POSSESSIVE = "s"

# Words around a destination that do not name it. core_words() removes them in
# this order, never down to nothing: leading determiners, trailing
# descriptions, then leading descriptions. Removing trailing words first keeps
# "project" in "the project page" but drops it from "the project Satira".
_DETERMINERS = phrase_set(
    "a", "an", "the", "this", "his", "your", "rastin", "rastins", "aghighi", "aghighis",
    "featured", "either", "both",
)
_TRAILING_DESCRIPTIONS = phrase_set(
    "page", "webpage", "web page", "section", "project", "featured", "work", "research"
)
_LEADING_DESCRIPTIONS = phrase_set(
    "project", "page for", "page about", "page on", "section for", "section about", "section on"
)

# Words that cannot name a destination by themselves.
GENERIC_WORDS = (
    frozenset(
        phrase[0]
        for phrase in _DETERMINERS | _TRAILING_DESCRIPTIONS | _LEADING_DESCRIPTIONS
        if len(phrase) == 1
    )
    | CONNECTORS
    | {_POSSESSIVE}
)


def core_words(tokens: Sequence[str]) -> Phrase:
    """Return the words of a destination phrase that name it.

    Possessive "s" tokens and the surrounding words above are removed; at
    least one word is kept, so "the project page" keeps "project".
    """
    words = tuple(token for token in tokens if token != _POSSESSIVE)
    words = strip_leading(words, _DETERMINERS, keep=1)
    words = strip_trailing(words, _TRAILING_DESCRIPTIONS, keep=1)
    return strip_leading(words, _LEADING_DESCRIPTIONS, keep=1)


def phrase_key(tokens: Sequence[str]) -> str:
    """Return the comparison key of a destination phrase: its core words, joined.

    Joining without spaces makes "Q MIND", "Q-MIND", and "QMIND" one key.
    """
    return "".join(core_words(tokens))
