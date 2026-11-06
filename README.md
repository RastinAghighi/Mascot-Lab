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

## Deterministic navigation (Part 2)

Navigation commands are resolved by application code before any model sees them, and only to semantic route IDs. No page paths or URLs exist yet, so none appear in the code or configuration.

- `configs/routes.yaml`, loaded by `src/mascot_lab/routes.py`: the navigation allowlist. It routes the five featured projects by their IDs from the navigation registry in `knowledge/facts.md` (`project_qmind`, `project_ingenuity`, `project_satira`, `project_deeplense`, `project_website`), lists the aliases that name each one, and lists ambiguous terms (`project`, `research`) that name several. The schema has no place for a path or URL; unknown keys such as `url` are rejected. Aliases that normalize to the same text collide, and the file is rejected.
- `src/mascot_lab/phrases.py`: the single normalization used for both aliases and messages. Letter case, accents, full-width forms, zero-width characters, punctuation, spacing, a possessive `'s`, and surrounding words such as "the", "page", or "project" do not matter; spelling does. A misspelling matches only if it is listed (`Ingenuity Labs`, `DeepLens`).
- `src/mascot_lab/router.py`: `Router.route(message)` returns a `RouteResult` with a `status`, the `route_id` when routed, the `candidates` when ambiguous, and the configured phrases that matched. It never returns text from the message, a URL, or a path. When a router is built, it checks that every configured phrase routes as configured and fails closed otherwise.
- `tests/test_routes.py` and `tests/test_router.py`: valid commands, spelling and case variants, near misses that must not match, alias collisions, ambiguous and unknown destinations, URL and path injection, result invariants, and a fixed-seed randomized safety check.

| Status | Meaning |
| --- | --- |
| `routed` | A navigation command for exactly one allowlisted route. |
| `ambiguous_destination` | The destination names several routes, as in "QMIND or Satira" or "the research project". |
| `unknown_destination` | A navigation command whose destination is not allowlisted, as in "take me to the Papers page". |
| `url_rejected` | A navigation command containing a URL, domain name, IP address, email address, or path. The allowlisted name `rastinaghighi.dev` is accepted only on its own. |
| `not_navigation` | Not a direct navigation command; the message is left for the model. |

Go commands such as "take me to", "go to", or "navigate to" always count as navigation. Show commands such as "show me", "open", or "visit" count only when the destination is allowlisted, contains a URL, or ends with a page word such as "page" or "website", so "show me what Satira does" stays a question for the model.

```python
from mascot_lab.router import Router

router = Router.from_file()
router.route("Can you take me to Q-MIND?")  # status routed, route_id 'project_qmind'
router.route("take me to https://example.com")  # status url_rejected, no route_id
```

Configuration decisions, recorded in `configs/routes.yaml`: `mascot_chat` has no text alias, so it is not routable by command, and "build log" is not an alias of `project_website`.

## Status

Foundation (Part 1) and deterministic navigation (Part 2) are implemented with automated tests. The Ollama adapter, the evaluation runner, and the evaluation cases are not implemented yet, and no model has been run through this code. Model quality, latency, robustness, hosting compatibility, and deployment security have not yet been established.

## Repository policy

This repository is private during development. The authoritative design PDF, local evaluation outputs, and unapproved private research details are excluded from version control.

Git commits and pushes are performed manually by the repository owner.
