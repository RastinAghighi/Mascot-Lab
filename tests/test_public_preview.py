"""Tests for the public API's thought-cloud preview and continuation.

``public_preview`` is a pure text function. Nothing here contacts Ollama or
opens a socket; the few engine-level tests use the in-process ``FakeOllama``.
These are not HTTP integration tests: the HTTP adapter does not exist yet.
``gateway_r4_failures`` repeats the gateway's R4 checks from the public
contract (chat-v1.md, section 18.2), which compare strings only. All texts
are synthetic.
"""

import dataclasses
import random
import socket
import textwrap
import unicodedata
from typing import Any

import pytest
from ollama_fakes import TAG, FakeOllama

from mascot_lab import ConfigError
from mascot_lab.chat_config import ChatConfig, load_chat_config
from mascot_lab.engine import ChatEngine, ChatStatus, Continuation, ContinuationReason
from mascot_lab.ollama import OllamaClient
from mascot_lab.output import cloud_preview
from mascot_lab.public_preview import (
    PUBLIC_CONTINUATION_ROUTE,
    PUBLIC_PREVIEW_MAX_CHARACTERS,
    PublicPreview,
    _wrap_spans,
    public_preview,
    reached_output_limit,
)

U2026 = "…"
ALLOWED_REASONS = (["exceeds_cloud"], ["output_limit_reached"], ["exceeds_cloud", "output_limit_reached"])
TRIMMED_PUNCTUATION = ",;:.-–—"

SHORT = "A short synthetic answer."
SENTENCE = "This synthetic sentence exists only to fill the thought cloud."
LONG = " ".join([SENTENCE] * 4)
FULL_CLOUD = " ".join(letter * 45 for letter in "abc")  # three full lines
CHINESE = "这是一个用于界面测试的合成句子。"
JAPANESE = "これは画面の表示を確かめるための合成の文です。"
QUESTION = "Please describe the synthetic test topic."


@pytest.fixture(scope="module")
def chat() -> ChatConfig:
    return load_chat_config()


@pytest.fixture
def engine(ollama: FakeOllama) -> ChatEngine:
    return ChatEngine.from_files(client=OllamaClient(transport=ollama))


def preview(answer: str, chat: ChatConfig, *, limit: bool = False) -> PublicPreview:
    return public_preview(answer, output_limit_reached=limit, chat=chat)


def gateway_r4_failures(answer_text: str, cloud_preview: str, continuation: dict[str, Any] | None) -> list[str]:
    """The numbers of the gateway's R4 checks that a response fails.

    The preview counts as shortened exactly when it differs from the answer,
    so check 1 holds by definition. No normalization, case folding or
    trimming: exact code-point comparison, as the contract requires.
    """
    reasons = [] if continuation is None else continuation["reasons"]
    shortened = cloud_preview != answer_text
    failures = []
    if shortened and "exceeds_cloud" not in reasons:
        failures.append("2: reason present")
    if shortened and not cloud_preview.endswith(U2026):
        failures.append("3: ellipsis")
    start = cloud_preview[: -len(U2026)] if cloud_preview.endswith(U2026) else cloud_preview
    if shortened and not (start and answer_text.startswith(start) and len(start) < len(answer_text)):
        failures.append("4: exact prefix")
    if ("exceeds_cloud" in reasons) != shortened:
        failures.append("5: reason only when shortened")
    return failures


def assert_public(answer: str, result: PublicPreview, *, max_lines: int = 3, width: int = 45) -> None:
    """``result`` meets the response schema's preview and continuation rules, R4, and the cloud."""
    shown = result.cloud_preview
    body = result.continuation_json()
    assert 1 <= len(shown) <= PUBLIC_PREVIEW_MAX_CHARACTERS
    assert len(shown) <= max_lines * width + max_lines - 1
    assert shown.strip()
    if body is not None:
        assert set(body) == {"route_id", "reasons"}
        assert body["route_id"] == "mascot_chat"
        assert body["reasons"] in ALLOWED_REASONS
    assert gateway_r4_failures(answer, shown, body) == []
    assert len(textwrap.wrap(shown, width=width, break_on_hyphens=False)) <= max_lines
    if shown != answer:  # the cut is at a word boundary, unless one word fills a whole line
        cut = len(shown) - len(U2026)
        if not answer[cut:].lstrip(TRIMMED_PUNCTUATION).startswith(" "):
            word_start = answer.rfind(" ", 0, cut) + 1
            word_end = answer.find(" ", cut)
            if word_end == -1:
                word_end = len(answer)
            assert word_end - word_start >= width, (answer, shown)


# --------------------------------------------------------- the R4 checks


R4_VIOLATIONS = [
    pytest.param(LONG, "An unrelated synthetic preview." + U2026, ["exceeds_cloud"], ["4: exact prefix"], id="unrelated"),
    pytest.param(LONG, LONG[:40], ["exceeds_cloud"], ["3: ellipsis"], id="no-ellipsis"),
    pytest.param(LONG, U2026, ["exceeds_cloud"], ["4: exact prefix"], id="only-ellipsis"),
    pytest.param(LONG, LONG + U2026, ["exceeds_cloud"], ["4: exact prefix"], id="whole-text-and-ellipsis"),
    pytest.param(LONG, LONG[:40] + U2026, [], ["2: reason present", "5: reason only when shortened"], id="no-reason"),
    pytest.param(SHORT, SHORT, ["exceeds_cloud"], ["5: reason only when shortened"], id="reason-not-shortened"),
    pytest.param(LONG, LONG[:40].upper() + U2026, ["exceeds_cloud"], ["4: exact prefix"], id="letter-case"),
    pytest.param(LONG, " " + LONG[:40] + U2026, ["exceeds_cloud"], ["4: exact prefix"], id="whitespace"),
    pytest.param(
        "Café " * 40,
        "Café " * 8 + U2026,
        ["exceeds_cloud"],
        ["4: exact prefix"],
        id="normalization",
    ),
]


@pytest.mark.parametrize(("answer", "shown", "reasons", "failures"), R4_VIOLATIONS)
def test_the_r4_checks_catch_the_contracts_violations(
    answer: str, shown: str, reasons: list[str], failures: list[str]
) -> None:
    continuation = {"route_id": "mascot_chat", "reasons": reasons} if reasons else None
    assert gateway_r4_failures(answer, shown, continuation) == failures


# ------------------------------------------------------ previews that fit


def test_a_short_answer_is_returned_unchanged(chat: ChatConfig) -> None:
    result = preview(SHORT, chat)
    assert result == PublicPreview(SHORT, None)
    assert result.continuation_json() is None
    assert_public(SHORT, result)


def test_an_answer_that_exactly_fills_the_cloud_is_unchanged(chat: ChatConfig) -> None:
    assert len(FULL_CLOUD) == PUBLIC_PREVIEW_MAX_CHARACTERS == 137
    result = preview(FULL_CLOUD, chat)
    assert result == PublicPreview(FULL_CLOUD, None)
    assert_public(FULL_CLOUD, result)


def test_one_more_character_gives_a_preview_of_the_maximum_length(chat: ChatConfig) -> None:
    answer = FULL_CLOUD + "c"  # the last word no longer fits the third line
    result = preview(answer, chat)
    assert result.cloud_preview == "a" * 45 + " " + "b" * 45 + " " + "c" * 44 + U2026
    assert len(result.cloud_preview) == PUBLIC_PREVIEW_MAX_CHARACTERS
    assert result.continuation_json() == {"route_id": "mascot_chat", "reasons": ["exceeds_cloud"]}
    assert_public(answer, result)


def test_a_last_line_that_leaves_room_for_the_ellipsis_is_kept_whole(chat: ChatConfig) -> None:
    answer = "a" * 45 + " " + "b" * 45 + " " + "c" * 44 + " d"
    result = preview(answer, chat)
    assert result.cloud_preview == answer[:136] + U2026
    assert len(result.cloud_preview) == PUBLIC_PREVIEW_MAX_CHARACTERS
    assert_public(answer, result)


@pytest.mark.parametrize(("length", "shortened"), [(135, False), (136, True)])
def test_a_single_word_fits_while_it_fills_at_most_three_lines(
    chat: ChatConfig, length: int, shortened: bool
) -> None:
    answer = "x" * length
    result = preview(answer, chat)
    assert result.cloud_preview == ("x" * 134 + U2026 if shortened else answer)
    assert_public(answer, result)


def test_fitting_depends_on_the_lines_not_only_the_length(chat: ChatConfig) -> None:
    answer = " ".join(["w" * 30] * 4)  # 123 code points, but one word per line
    assert len(answer) < PUBLIC_PREVIEW_MAX_CHARACTERS
    result = preview(answer, chat)
    assert result.cloud_preview == " ".join(["w" * 30] * 3) + U2026
    assert_public(answer, result)


# ------------------------------------------------------ shortened previews


def test_a_long_answer_is_cut_at_a_word_boundary(chat: ChatConfig) -> None:
    result = preview(LONG, chat)
    kept = result.cloud_preview[: -len(U2026)]
    assert result.cloud_preview.endswith(U2026)
    assert LONG.startswith(kept) and kept[-1].isalnum()
    assert LONG[len(kept) :].lstrip(TRIMMED_PUNCTUATION).startswith(" ")
    assert result.continuation == Continuation("mascot_chat", (ContinuationReason.EXCEEDS_CLOUD,))
    # This text has no word longer than a line, so the engine's preview is the same.
    assert result.cloud_preview == cloud_preview(LONG, max_lines=3, line_characters=45)[0]
    assert_public(LONG, result)


@pytest.mark.parametrize("words_before", [0, 1, 2, 5, 9, 14])
def test_a_word_longer_than_a_line_is_kept_exactly(chat: ChatConfig, words_before: int) -> None:
    long_word = "Synthetic" * 6  # 54 code points
    answer = " ".join(["word"] * words_before + [long_word] + ["after"] * 30)
    result = preview(answer, chat)
    assert_public(answer, result)


def test_a_word_boundary_on_the_last_line_is_preferred(chat: ChatConfig) -> None:
    answer = "a" * 45 + " " + "b" * 45 + " c " + "x" * 100
    assert preview(answer, chat).cloud_preview == "a" * 45 + " " + "b" * 45 + " c" + U2026


def test_a_word_that_fills_the_last_line_is_cut_within_the_word(chat: ChatConfig) -> None:
    answer = "Short words first " + "x" * 200
    result = preview(answer, chat)
    assert result.cloud_preview == answer[:134] + U2026
    assert_public(answer, result)


def test_a_long_answer_without_spaces_is_cut_within_it(chat: ChatConfig) -> None:
    answer = "".join(f"{number:03d}" for number in range(200))
    result = preview(answer, chat)
    assert result.cloud_preview == answer[:134] + U2026
    assert_public(answer, result)


@pytest.mark.parametrize("sentence", [CHINESE, JAPANESE], ids=["chinese", "japanese"])
def test_chinese_and_japanese_are_kept_exactly(chat: ChatConfig, sentence: str) -> None:
    assert preview(sentence, chat) == PublicPreview(sentence, None)
    answer = sentence * 12
    result = preview(answer, chat)
    assert " " not in result.cloud_preview
    assert result.cloud_preview == answer[: len(result.cloud_preview) - 1] + U2026
    assert_public(answer, result)


def test_supplementary_code_points_count_as_one_character(chat: ChatConfig) -> None:
    for char in ("\U0001F642", "\U0001D400", "\U00020000"):  # an emoji, a bold A, a CJK ideograph
        answer = char * 200
        result = preview(answer, chat)
        assert result.cloud_preview == char * 134 + U2026
        result.cloud_preview.encode("utf-8")  # no unpaired surrogate
        assert_public(answer, result)


SEQUENCES = [
    pytest.param("\U0001F44D\U0001F3FD", id="skin-tone-modifier"),
    pytest.param("\U0001F1E8\U0001F1E6", id="flag"),
    pytest.param("\U0001F469‍\U0001F469‍\U0001F467", id="zwj-sequence"),
    pytest.param("❤️", id="variation-selector"),
    pytest.param("1️⃣", id="keycap"),
    pytest.param("é", id="combining-accent"),
    pytest.param("क्ष", id="devanagari-conjunct"),
    pytest.param(unicodedata.normalize("NFD", "한"), id="decomposed-hangul"),
]


@pytest.mark.parametrize("offset", range(5))
@pytest.mark.parametrize("sequence", SEQUENCES)
def test_a_cut_within_a_word_never_splits_a_character(chat: ChatConfig, sequence: str, offset: int) -> None:
    answer = "a" * offset + sequence * 100
    result = preview(answer, chat)
    kept = result.cloud_preview[: -len(U2026)]
    assert answer.startswith(kept)
    assert (len(kept) - offset) % len(sequence) == 0  # whole sequences only
    assert len(kept) > 134 - len(sequence)  # and no more is dropped than needed
    assert_public(answer, result)


def test_a_letter_with_hundreds_of_marks_is_still_cut_within_the_bound(chat: ChatConfig) -> None:
    answer = "a" + "́" * 300
    result = preview(answer, chat)
    assert result.cloud_preview == answer[:134] + U2026
    assert_public(answer, result)


def test_previews_are_never_normalized(chat: ChatConfig) -> None:
    composed = "Café résumé naïve façade"
    decomposed = unicodedata.normalize("NFD", composed)
    assert decomposed != composed
    for answer in (composed, decomposed):
        assert preview(answer, chat).cloud_preview == answer
    answer = " ".join([decomposed] * 8)
    result = preview(answer, chat)
    assert_public(answer, result)
    # A preview of the normalized text would fail the gateway's exact-prefix check.
    normalized = preview(unicodedata.normalize("NFC", answer), chat)
    assert gateway_r4_failures(answer, normalized.cloud_preview, normalized.continuation_json()) == [
        "4: exact prefix"
    ]


# ------------------------------------------------------------ continuation


CONTINUATIONS = [
    pytest.param(SHORT, False, None, id="none"),
    pytest.param(LONG, False, ["exceeds_cloud"], id="exceeds-cloud"),
    pytest.param(SHORT, True, ["output_limit_reached"], id="output-limit"),
    pytest.param(LONG, True, ["exceeds_cloud", "output_limit_reached"], id="both"),
]


@pytest.mark.parametrize(("answer", "limit", "reasons"), CONTINUATIONS)
def test_continuation_reasons(chat: ChatConfig, answer: str, limit: bool, reasons: list[str] | None) -> None:
    result = preview(answer, chat, limit=limit)
    expected = None if reasons is None else {"route_id": "mascot_chat", "reasons": reasons}
    assert result.continuation_json() == expected
    assert (result.cloud_preview == answer) == (answer == SHORT)  # the limit never shortens a preview
    assert_public(answer, result)


def test_both_reasons_are_in_the_contracts_order(chat: ChatConfig) -> None:
    result = preview(LONG, chat, limit=True)
    reasons = (ContinuationReason.EXCEEDS_CLOUD, ContinuationReason.OUTPUT_LIMIT_REACHED)
    assert result.continuation == Continuation("mascot_chat", reasons)


def test_each_continuation_body_is_a_new_object(chat: ChatConfig) -> None:
    result = preview(LONG, chat, limit=True)
    body = result.continuation_json()
    assert body is not None
    body["reasons"].append("changed")
    assert result.continuation_json() == {
        "route_id": "mascot_chat",
        "reasons": ["exceeds_cloud", "output_limit_reached"],
    }


@pytest.mark.parametrize("flag", [None, 0, 1, "length", "stop"])
def test_the_output_limit_must_be_a_bool(chat: ChatConfig, flag: Any) -> None:
    with pytest.raises(TypeError, match="output_limit_reached"):
        public_preview(SHORT, output_limit_reached=flag, chat=chat)


def test_the_answer_must_be_text_with_a_visible_character(chat: ChatConfig) -> None:
    with pytest.raises(TypeError, match="answer_text"):
        public_preview(b"bytes", output_limit_reached=False, chat=chat)  # type: ignore[arg-type]
    for blank in ("", " ", "   "):
        with pytest.raises(ValueError, match="whitespace"):
            preview(blank, chat)


# ------------------------------------------------- the engine's metadata


LIMIT_CASES = [
    pytest.param(SHORT, "stop", None, id="short-stop"),
    pytest.param(LONG, "stop", ["exceeds_cloud"], id="long-stop"),
    pytest.param(SHORT, "length", ["output_limit_reached"], id="short-length"),
    pytest.param(LONG, "length", ["exceeds_cloud", "output_limit_reached"], id="long-length"),
]


@pytest.mark.parametrize(("answer", "done_reason", "reasons"), LIMIT_CASES)
def test_the_output_limit_comes_from_the_engines_report(
    engine: ChatEngine, ollama: FakeOllama, answer: str, done_reason: str, reasons: list[str] | None
) -> None:
    ollama.reply(answer, done_reason=done_reason)
    response = engine.respond(QUESTION, TAG)
    assert response.status is ChatStatus.ANSWERED and response.answer_text == answer
    assert reached_output_limit(response) is (done_reason == "length")
    result = public_preview(
        response.answer_text, output_limit_reached=reached_output_limit(response), chat=engine.chat_config
    )
    assert len(ollama.requests) == 1  # only the engine's request: the helper sends nothing
    expected = None if reasons is None else {"route_id": "mascot_chat", "reasons": reasons}
    assert result.continuation_json() == expected
    assert_public(answer, result)


def test_an_error_at_the_output_limit_has_no_limit_reason(engine: ChatEngine, ollama: FakeOllama) -> None:
    ollama.reply("", done_reason="length")
    response = engine.respond(QUESTION, TAG)
    assert response.status is ChatStatus.ERROR
    assert reached_output_limit(response) is False
    result = public_preview(response.answer_text, output_limit_reached=False, chat=engine.chat_config)
    assert result == PublicPreview(response.answer_text, None)


# ----------------------------------------------------------- configuration


def test_the_configured_cloud_gives_the_public_bound(chat: ChatConfig) -> None:
    # The public bound is derived from these settings (chat-v1.md, section 7).
    lines, width = chat.cloud_max_lines, chat.cloud_line_characters
    assert lines * width + lines - 1 == PUBLIC_PREVIEW_MAX_CHARACTERS
    assert chat.continuation_route == PUBLIC_CONTINUATION_ROUTE == "mascot_chat"


@pytest.mark.parametrize("cloud", [{"cloud_max_lines": 4}, {"cloud_line_characters": 46}])
def test_a_cloud_larger_than_the_public_bound_is_refused(chat: ChatConfig, cloud: dict[str, int]) -> None:
    with pytest.raises(ConfigError, match="public API allows 137"):
        preview(SHORT, dataclasses.replace(chat, **cloud))


def test_another_continuation_route_is_refused(chat: ChatConfig) -> None:
    with pytest.raises(ConfigError, match="mascot_chat"):
        preview(SHORT, dataclasses.replace(chat, continuation_route="project_qmind"))


def test_a_smaller_cloud_gives_smaller_previews(chat: ChatConfig) -> None:
    small = dataclasses.replace(chat, cloud_max_lines=2, cloud_line_characters=20)
    for answer in (LONG, "x" * 300, CHINESE * 12):
        result = preview(answer, small)
        assert len(result.cloud_preview) <= 41
        assert_public(answer, result, max_lines=2, width=20)


# --------------------------------------------- the engine's preview, kept


INSERTED_SPACE_CASES = [
    pytest.param("x" * 300, id="one-long-word"),
    pytest.param("Short words first " + "Synthetic" * 20, id="long-word-after-words"),
    pytest.param("Hi " + "Synthetic" * 7 + " " + " ".join(["after"] * 30), id="long-word-on-the-first-line"),
    pytest.param(CHINESE * 12, id="chinese"),
    pytest.param(JAPANESE * 12, id="japanese"),
]


@pytest.mark.parametrize("answer", INSERTED_SPACE_CASES)
def test_the_engine_preview_would_fail_the_gateway_where_this_one_passes(chat: ChatConfig, answer: str) -> None:
    """The engine's preview, which stays unchanged, inserts a space where it splits a word."""
    engine_preview, cut = cloud_preview(answer, max_lines=3, line_characters=45)
    assert cut
    engine_body = {"route_id": "mascot_chat", "reasons": ["exceeds_cloud"]}
    assert gateway_r4_failures(answer, engine_preview, engine_body) == ["4: exact prefix"]
    result = preview(answer, chat)
    kept = result.cloud_preview[: -len(U2026)]
    assert kept.count(" ") == answer[: len(kept)].count(" ")  # no space inserted
    assert_public(answer, result)


def test_the_line_layout_is_the_engines_word_wrap() -> None:
    rng = random.Random(2026)
    pieces = ["a", "word", "x" * 44, "y" * 45, "z" * 46, "w" * 100, " ", "  ", " ", "　"]
    pieces += [CHINESE, "é", "\U0001F642"]
    for _ in range(3000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 40)))
        width = rng.randint(10, 60)
        lines = [text[start:end] for start, end in _wrap_spans(text, width)]
        assert lines == textwrap.wrap(text, width=width, break_on_hyphens=False), (text, width)


def test_the_preview_equals_the_engines_wherever_that_is_an_exact_prefix(chat: ChatConfig) -> None:
    rng = random.Random(45)
    words = ["synthetic", "words", "fill", "the", "cloud,", "then;", "text.", "a", "Q-test", "z" * 44, "x" * 45]
    words += ["y" * 60]
    compared = 0
    for _ in range(2000):
        answer = " ".join(rng.choice(words) for _ in range(rng.randint(1, 50)))
        engine_preview, engine_cut = cloud_preview(answer, max_lines=3, line_characters=45)
        result = preview(answer, chat)
        assert (result.cloud_preview != answer) == engine_cut, answer  # the same decision to shorten
        if engine_cut and not gateway_r4_failures(answer, engine_preview, result.continuation_json()):
            assert result.cloud_preview == engine_preview, answer
            compared += 1
    assert compared > 200


# ---------------------------------------------------------------- invariants


def test_random_answers_meet_the_public_contract(chat: ChatConfig) -> None:
    rng = random.Random(137)
    vocabulary = ["synthetic", "words", "fill", "the", "cloud,", "and", "then;", "text.", "a", "Q-test"]
    vocabulary += ["x" * 45, "y" * 60, "これは", "été", "\U0001F642"]
    vocabulary += ["\U0001F44D\U0001F3FD", "\U0001F1E8\U0001F1E6", "क्ष", CHINESE]
    for _ in range(2000):
        answer = " ".join(rng.choice(vocabulary) for _ in range(rng.randint(1, 60)))
        if rng.random() < 0.2:
            answer = answer.replace(" ", "")
        limit = rng.random() < 0.5
        result = preview(answer, chat, limit=limit)
        assert_public(answer, result)
        reasons = () if result.continuation is None else result.continuation.reasons
        assert (ContinuationReason.EXCEEDS_CLOUD in reasons) == (result.cloud_preview != answer)
        assert (ContinuationReason.OUTPUT_LIMIT_REACHED in reasons) == limit


def test_the_answer_text_is_not_modified(chat: ChatConfig) -> None:
    pieces = ["Synthetic", " ", "é", " ", CHINESE, " ", "\U0001F44D\U0001F3FD", " "] * 20
    answer = "".join(pieces)
    copy = "".join(pieces)  # equal, but a separate object
    for limit in (False, True):
        result = preview(answer, chat, limit=limit)
        assert answer == copy
        assert copy.startswith(result.cloud_preview[: -len(U2026)])
    short = "Synthetic é answer"
    assert preview(short, chat).cloud_preview is short


def test_the_helper_opens_no_socket(chat: ChatConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("public_preview opened a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    for answer in (SHORT, LONG, CHINESE * 12, "x" * 300):
        preview(answer, chat, limit=True)
