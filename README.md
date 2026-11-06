# Mascot-Lab

Development and evaluation environment for the small language model (SLM) powering the planned mascot chatbot on rastinaghighi.dev.

## Objective

Evaluate six local Ollama models using one shared Python implementation, consistent instructions, an approved knowledge baseline, deterministic navigation, and reproducible tests.

No winning model has been selected.

## Candidate models

- granite4:micro-h
- gemma:2b-instruct-q4_K_M
- gemma:2b-instruct-v1.1-q4_K_M
- qwen3.5:2b-q4_K_M
- phi4-mini:3.8b-q4_K_M
- granite4.2:3b-q4_K_M

## Architecture

- Shared system instructions for all candidates
- Server-side knowledge injection from knowledge/facts.md
- Deterministic navigation using semantic route IDs
- One configurable Ollama integration
- Repeatable evaluation and human-reviewed results
- No fine-tuning or character-persona training in v1

## Documentation

- knowledge/facts.md — owner-approved knowledge baseline
- docs/slm-development-handoff.md — development requirements and completed hardware observations
- docs/design-reference.md — reference to the authoritative website design specification

## Development setup

Requires Python 3.12. From the repository root in Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```

## Foundation (Part 1)

- `configs/models.yaml`, loaded by `src/mascot_lab/config.py`: the six exact Ollama tags and one shared option baseline (`num_ctx: 4096`). A model may differ from the baseline only through an `overrides` block with a written `reason`; prompts and knowledge cannot be set per model. Documented exceptions: none.
- `prompts/system.md`: the one system instruction shared by all six models. Its `{{MASCOT_KNOWLEDGE}}` placeholder is replaced with the knowledge block. It asks for grounded answers of about three short lines; it is not a security control.
- `src/mascot_lab/knowledge.py`: returns only the text between the `<!-- BEGIN MASCOT KNOWLEDGE -->` and `<!-- END MASCOT KNOWLEDGE -->` lines of `knowledge/facts.md`, fails closed on missing, duplicated, out-of-order, malformed, or empty boundaries, and records the block's SHA-256.
- `src/mascot_lab/prompt.py`: builds identical system and user messages for every model and records the SHA-256 of the instruction, the knowledge block, and the assembled system message.
- `tests/`: automated tests for configuration, knowledge extraction, and prompt consistency.

Hashes cover UTF-8 text with LF line endings, so they do not depend on the platform or Git line-ending settings.

## Status

Foundation (Part 1) implemented with automated tests. Navigation routing, the Ollama adapter, the evaluation runner, and the evaluation cases are not implemented yet, and no model has been run through this code. Model quality, latency, robustness, hosting compatibility, and deployment security have not yet been established.

## Repository policy

This repository is private during development. The authoritative design PDF, local evaluation outputs, and unapproved private research details are excluded from version control.

Git commits and pushes are performed manually by the repository owner.
