"""Machine measurements of one response: measured or reported values only.

``measure`` collects the values the evaluation records for one
``ChatResponse``, using the field names of the handoff's test protocol
(section 5) where it names them. Each value is measured by this harness,
reported by Ollama, or computed exactly from those. A value that was not
measured or not reported is None (JSON null); nothing is estimated.

=========================  ===================================  ==========================================
Field                      Source                               None when
=========================  ===================================  ==========================================
``pipeline_ms``            the runner's clock around the whole   never
                           ``ChatEngine.respond`` call (steps
                           1 to 6), in milliseconds
``elapsed_ms``             the Ollama client's clock around the  no request was sent to Ollama
                           HTTP request, in milliseconds; also
                           set for failed requests
``time_to_first_token_ms`` not measured: requests are not        always
                           streamed
``*_duration_ns``          Ollama's ``load_duration``,           Ollama did not report it, or no response
                           ``prompt_eval_duration``,
                           ``eval_duration``, and
                           ``total_duration``, in nanoseconds
``prompt_tokens``          Ollama's ``prompt_eval_count``        Ollama did not report it, or no response
``output_tokens``          Ollama's ``eval_count``               Ollama did not report it, or no response
``tokens_per_second``      ``output_tokens`` divided by          either is None, or ``eval_duration_ns``
                           ``eval_duration_ns`` / 10^9, rounded  is 0
                           to 3 decimals
``answer_characters``,     the displayed answer: characters,     the status is not ``answered``, so the
``word_count``,            whitespace-separated words, and       displayed text is a fixed application
``answer_lines``           lines when word-wrapped at the        reply, not a model answer
                           cloud's configured line width
=========================  ===================================  ==========================================

``answer_lines`` wraps text exactly as ``mascot_lab.output.cloud_preview``
does, so an answer exceeds the cloud preview exactly when it has more lines
than ``cloud.max_lines``. Like the preview, it approximates the thought
cloud at the text level; the real fit depends on the cloud's width and font.

Ollama's ``load_duration`` is included in its ``total_duration`` and in
``elapsed_ms``, so a request that had to load the model first takes longer.
``prompt_tokens`` counts the prompt tokens Ollama evaluated for the request;
how Ollama reports it when it reuses cached prompt tokens depends on its
version and has not been verified here.
"""

import textwrap
from typing import Any

from mascot_lab.engine import ChatResponse, ChatStatus

MEASUREMENT_FIELDS = (
    "pipeline_ms",
    "elapsed_ms",
    "time_to_first_token_ms",
    "load_duration_ns",
    "prompt_eval_duration_ns",
    "eval_duration_ns",
    "total_duration_ns",
    "prompt_tokens",
    "output_tokens",
    "tokens_per_second",
    "answer_characters",
    "word_count",
    "answer_lines",
)


def measure(response: ChatResponse, *, pipeline_ms: float, line_characters: int) -> dict[str, Any]:
    """Return the measurements of ``response``, keyed by ``MEASUREMENT_FIELDS``.

    ``pipeline_ms`` is the runner's measurement of the engine call, and
    ``line_characters`` the cloud's configured line width.
    """
    output, error = response.output, response.error
    metrics = None if output is None else output.metrics
    if output is not None:
        elapsed_ms = output.elapsed_ms
    elif error is not None:
        elapsed_ms = error.elapsed_ms  # set when a request was sent and failed
    else:
        elapsed_ms = None

    def reported(name: str) -> int | None:
        return None if metrics is None else getattr(metrics, name)

    output_tokens = reported("eval_count")
    eval_duration_ns = reported("eval_duration")
    answer = response.answer_text if response.status is ChatStatus.ANSWERED else None
    return {
        "pipeline_ms": pipeline_ms,
        "elapsed_ms": elapsed_ms,
        "time_to_first_token_ms": None,  # requests are not streamed
        "load_duration_ns": reported("load_duration"),
        "prompt_eval_duration_ns": reported("prompt_eval_duration"),
        "eval_duration_ns": eval_duration_ns,
        "total_duration_ns": reported("total_duration"),
        "prompt_tokens": reported("prompt_eval_count"),
        "output_tokens": output_tokens,
        "tokens_per_second": tokens_per_second(output_tokens, eval_duration_ns),
        "answer_characters": None if answer is None else len(answer),
        "word_count": None if answer is None else len(answer.split()),
        "answer_lines": None if answer is None else wrapped_line_count(answer, line_characters),
    }


def tokens_per_second(output_tokens: int | None, eval_duration_ns: int | None) -> float | None:
    """Output tokens per second of generation, or None when it cannot be computed."""
    if output_tokens is None or eval_duration_ns is None or eval_duration_ns <= 0:
        return None
    return round(output_tokens * 1_000_000_000 / eval_duration_ns, 3)


def wrapped_line_count(text: str, line_characters: int) -> int:
    """Lines of ``text`` word-wrapped as ``cloud_preview`` wraps it."""
    return len(textwrap.wrap(text, width=line_characters, break_on_hyphens=False))
