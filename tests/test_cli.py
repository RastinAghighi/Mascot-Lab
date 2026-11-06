"""Tests for the command line (``python -m mascot_lab``).

Commands that would contact Ollama receive ``FakeOllama`` as their
transport, so no test contacts a real server or runs a model.
"""

import json
import os
import subprocess
import sys
from typing import Any

import pytest
from ollama_fakes import TAG, FakeOllama

from mascot_lab import ModelsConfig
from mascot_lab.cli import main
from mascot_lab.ollama import OllamaError, OllamaErrorCode
from mascot_lab.paths import PROJECT_ROOT

QUESTION = "Where does Rastin study?"
ANSWER = "Rastin studies at Queen's University."


def run(capsys: pytest.CaptureFixture[str], *argv: str, transport: FakeOllama | None = None) -> tuple[int, str, str]:
    status = main(list(argv), transport=transport)
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def run_json(capsys: pytest.CaptureFixture[str], *argv: str, transport: FakeOllama | None = None) -> tuple[int, Any]:
    status, out, err = run(capsys, *argv, transport=transport)
    assert err == ""
    return status, json.loads(out)


# ------------------------------------------------------------------ validate


def test_validate_checks_the_configuration_without_contacting_ollama(
    capsys: pytest.CaptureFixture[str], ollama: FakeOllama, models_config: ModelsConfig
) -> None:
    status, out, err = run(capsys, "validate", transport=ollama)
    assert (status, err) == (0, "")
    assert out.startswith("Configuration is valid.")
    assert "baseline: think=false; num_ctx=4096, temperature=0.0, seed=42, top_p=1.0, num_predict=160" in out
    assert "documented exceptions: none" in out
    for tag in models_config.tags:
        assert tag in out
    assert "continuation route mascot_chat" in out
    assert "Ollama was not contacted." in out
    assert ollama.requests == []


def test_validate_ollama_reads_model_metadata_only(
    capsys: pytest.CaptureFixture[str], ollama: FakeOllama, models_config: ModelsConfig
) -> None:
    for tag in models_config.tags:
        capabilities = ["completion", "thinking"] if tag.startswith("qwen") else ["completion"]
        ollama.json(200, {"capabilities": capabilities, "parameters": 'stop "<|end|>"\nrepeat_penalty 1'})
    status, out, err = run(capsys, "validate", "--ollama", transport=ollama)
    assert (status, err) == (0, "")
    assert [request.url.rsplit("/", 1)[1] for request in ollama.requests] == ["show"] * 6
    assert [request.payload for request in ollama.requests] == [{"model": tag} for tag in models_config.tags]
    assert "qwen3.5:2b-q4_K_M: installed; capabilities: completion, thinking" in out
    assert 'Modelfile parameters: stop "<|end|>" | repeat_penalty 1' in out
    assert "thinking capability reported by: qwen3.5:2b-q4_K_M" in out


def test_validate_ollama_reports_missing_models(
    capsys: pytest.CaptureFixture[str], ollama: FakeOllama, models_config: ModelsConfig
) -> None:
    for tag in models_config.tags:
        if tag == "phi4-mini:3.8b-q4_K_M":
            ollama.json(404, {"error": f"model '{tag}' not found"})
        else:
            ollama.json(200, {"capabilities": ["completion"]})
    status, out, _ = run(capsys, "validate", "--ollama", transport=ollama)
    assert status == 1
    assert "phi4-mini:3.8b-q4_K_M: NOT FOUND" in out
    assert "not installed: phi4-mini:3.8b-q4_K_M" in out
    assert "thinking capability reported by: none" in out


def test_validate_ollama_when_the_server_is_down(capsys: pytest.CaptureFixture[str], ollama: FakeOllama) -> None:
    ollama.fail(OllamaError(OllamaErrorCode.UNAVAILABLE, "cannot reach Ollama"))
    status, _, err = run(capsys, "validate", "--ollama", transport=ollama)
    assert status == 1
    assert "cannot reach Ollama" in err


# --------------------------------------------------------------------- route


def test_route_shows_a_navigation_response(capsys: pytest.CaptureFixture[str], ollama: FakeOllama) -> None:
    status, record = run_json(capsys, "route", "take me to QMIND", transport=ollama)
    assert status == 0
    assert record["status"] == "routed"
    assert record["actions"] == [{"type": "navigate", "route_id": "project_qmind"}]
    assert record["model"] is None
    assert ollama.requests == []


def test_route_shows_an_open_question_as_not_navigation(
    capsys: pytest.CaptureFixture[str], ollama: FakeOllama
) -> None:
    status, record = run_json(capsys, "route", "What does Rastin study?", transport=ollama)
    assert status == 0
    assert record["status"] == "not_navigation"
    assert record["message"] == "What does Rastin study?"
    assert record["route"]["status"] == "not_navigation"
    assert ollama.requests == []


def test_route_reports_invalid_input(capsys: pytest.CaptureFixture[str]) -> None:
    status, record = run_json(capsys, "route", "   ")
    assert status == 0
    assert record["status"] == "invalid_input"
    assert record["error"]["code"] == "empty_input"


# ---------------------------------------------------------------------- chat


def test_chat_sends_one_request_and_prints_the_response(
    capsys: pytest.CaptureFixture[str], ollama: FakeOllama
) -> None:
    ollama.reply(ANSWER)
    status, record = run_json(capsys, "chat", "--model", TAG, QUESTION, transport=ollama)
    assert status == 0
    assert record["status"] == "answered"
    assert record["answer_text"] == ANSWER
    assert "system_message" not in record["prompt"]
    (request,) = ollama.requests
    assert request.url == "http://127.0.0.1:11434/api/chat"
    assert request.timeout == 120.0
    assert request.payload["think"] is False
    assert request.payload["options"]["num_predict"] == 160


def test_chat_accepts_a_configured_name_url_and_timeout(
    capsys: pytest.CaptureFixture[str], ollama: FakeOllama
) -> None:
    ollama.reply(ANSWER, model="qwen3.5:2b-q4_K_M")
    argv = ("chat", "--model", "Qwen3.5 2B", "--ollama-url", "http://localhost:11500", "--timeout", "30", QUESTION)
    status, record = run_json(capsys, *argv, transport=ollama)
    assert status == 0
    assert record["model"] == "qwen3.5:2b-q4_K_M"
    (request,) = ollama.requests
    assert (request.url, request.timeout) == ("http://localhost:11500/api/chat", 30.0)


def test_chat_dry_run_prints_the_request_and_sends_nothing(
    capsys: pytest.CaptureFixture[str], ollama: FakeOllama
) -> None:
    status, record = run_json(capsys, "chat", "--model", TAG, "--dry-run", QUESTION, transport=ollama)
    assert status == 0
    assert record["dry_run"] == "nothing was sent"
    payload = record["payload"]
    assert payload["model"] == TAG
    assert payload["think"] is False
    assert payload["messages"][0]["content"].startswith("<system message: 7670 characters")
    assert payload["messages"][1] == {"role": "user", "content": QUESTION}
    assert ollama.requests == []


def test_chat_dry_run_can_show_the_full_prompt(capsys: pytest.CaptureFixture[str], ollama: FakeOllama) -> None:
    status, record = run_json(
        capsys, "chat", "--model", TAG, "--dry-run", "--show-prompt", QUESTION, transport=ollama
    )
    assert status == 0
    assert record["payload"]["messages"][0]["content"].startswith("You answer questions about Rastin Aghighi.")
    assert ollama.requests == []


def test_chat_navigation_never_reaches_the_model(capsys: pytest.CaptureFixture[str], ollama: FakeOllama) -> None:
    status, record = run_json(capsys, "chat", "--model", TAG, "show me Satira", transport=ollama)
    assert status == 0
    assert record["status"] == "routed"
    assert ollama.requests == []


def test_chat_failure_exits_with_status_one(capsys: pytest.CaptureFixture[str], ollama: FakeOllama) -> None:
    ollama.fail(OllamaError(OllamaErrorCode.TIMEOUT, "no complete response within 120 s"))
    status, record = run_json(capsys, "chat", "--model", TAG, QUESTION, transport=ollama)
    assert status == 1
    assert record["status"] == "error"
    assert record["error"]["code"] == "timeout"
    assert record["answer_text"] == "Sorry, I can't answer right now. Please try again later."


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (("chat", "--model", "llama3:8b", QUESTION), "not a configured model"),
        (("chat", "--model", TAG, "--ollama-url", "http://example.com:11434", QUESTION), "loopback"),
        (("chat", "--model", TAG, "--timeout", "0", QUESTION), "timeout"),
        (("validate", "--ollama-url", "https://127.0.0.1:11434"), "loopback"),
    ],
)
def test_invalid_settings_exit_with_status_two(
    capsys: pytest.CaptureFixture[str], ollama: FakeOllama, argv: tuple[str, ...], message: str
) -> None:
    status, out, err = run(capsys, *argv, transport=ollama)
    assert status == 2
    assert out == ""
    assert err.startswith("error: ") and message in err
    assert ollama.requests == []


def test_a_missing_command_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main([])
    assert caught.value.code == 2
    assert "COMMAND" in capsys.readouterr().err


# ------------------------------------------------------------ python -m entry


def test_the_module_entry_point_runs_offline_commands() -> None:
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    completed = subprocess.run(
        [sys.executable, "-m", "mascot_lab", "route", "take me to Satira"],
        capture_output=True,
        cwd=PROJECT_ROOT,
        encoding="utf-8",
        env=environment,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["actions"] == [{"type": "navigate", "route_id": "project_satira"}]
