# Mascot-Lab

Development and evaluation environment for the small language model (SLM) powering the planned mascot chatbot on rastinaghighi.dev.

## Objective

Evaluate six local Ollama models using one shared Python implementation, consistent instructions, an approved knowledge baseline, deterministic navigation, and reproducible tests.

A v1 model has been selected provisionally after the first real evaluation (`docs/model-selection-v1.md`, `configs/selection.yaml`). Nothing is deployed.

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
- docs/architecture.md — the complete architecture: components, request and evaluation pipelines, record formats, hashes, safety controls, and limitations
- eval/rubric.md — the provisional human-review rubric
- eval/cases.jsonl — the 24-case v1 evaluation set
- docs/model-selection-v1.md — the first real evaluation, the provisional v1 model selection, and its deployment readiness
- docs/future-improvements.md — deferred work, separating what blocks a safe public launch from optional improvements

## Development setup

Requires Python 3.12. From the repository root in Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```

## Foundation (Part 1)

- `configs/models.yaml`, loaded by `src/mascot_lab/config.py`: the six exact Ollama tags and one shared baseline (its current settings are listed under Part 3). A model may differ from the baseline only through an `overrides` block with a written `reason`; prompts and knowledge cannot be set per model. Documented exceptions: none.
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

## Shared chat engine (Part 3)

One engine handles a message for any of the six models, in a fixed order, and stops at the first step that settles it:

1. **Input validation**: line endings become LF; control characters other than line breaks and tabs, and invisible format characters (zero-width spaces, bidirectional controls, Unicode tag characters) are removed; surrounding whitespace is trimmed. An empty message, or one longer than `input.max_characters`, is refused.
2. **Deterministic routing**: the Part 2 router. Navigation commands are answered by the application, never by a model.
3. **Approved knowledge loading**: the bounded knowledge block is read from `knowledge/facts.md` for every request that reaches this step, so an edited fact applies to the next request.
4. **Shared prompt construction**: the Part 1 prompt builder, identical for every model.
5. **Ollama request**: one `POST /api/chat` with the model's exact tag, the messages, `think`, and the configured options.
6. **Output validation**: the display text is derived from the raw answer, which is kept unchanged.

Files:

- `src/mascot_lab/engine.py`: `ChatEngine` and the structured `ChatResponse`.
- `src/mascot_lab/ollama.py`: the single Ollama adapter (standard library only). It sends non-streamed requests only to a loopback address, because the prompts carry the private knowledge block. It ignores proxies configured in the environment, never follows redirects, limits response size, and rejects option names it does not know: Ollama ignores unknown options with only a server-side warning, so a misspelled option would otherwise silently fall back to the model's default.
- `src/mascot_lab/output.py`: output validation and the cloud preview.
- `src/mascot_lab/chat_config.py` and `configs/chat.yaml`: the input limit, the cloud approximation, the continuation route, reply labels, and fixed replies, validated as strictly as the other configuration files.
- `src/mascot_lab/cli.py`: the command line.

### Responses

A `ChatResponse` holds what a visitor would see and, separately, the record of what ran:

| Field | Content |
| --- | --- |
| `answer_text` | The validated model answer, or a fixed reply from `configs/chat.yaml`. Plain text. |
| `cloud_preview` | The start of `answer_text` that fits about 3 short lines, cut at a word with an ellipsis when it runs over. |
| `continuation` | Set when the answer runs past the preview (`exceeds_cloud`) or was cut at the output limit (`output_limit_reached`). It points to `mascot_chat`, the mascot's page, for the design plan's "continue on the mascot's page" link (p. 7). |
| `actions` | Semantic navigation actions, only from the router's allowlist: `navigate` for a direct request such as "show me Satira", which goes straight to the page, and `suggest`, one per candidate, for an ambiguous destination (the design plan's "take me there" button, p. 7). |
| `output` | The raw model output, unmodified: `content`, separate `thinking`, `done_reason`, and Ollama's reported counts and durations, plus the client-measured `elapsed_ms`. |
| `output_issues` | What output validation found or changed. |
| `error` | Why no usable answer was produced. Its message is for records and is never shown. |
| `message`, `route`, `prompt`, `request`, `configs` | The validated message, the router's result, the prompt hashes, the request settings, and the SHA-256 of each configuration file. |

| Status | Meaning | Model called |
| --- | --- | --- |
| `answered` | The model's validated answer. | Yes |
| `routed`, `ambiguous_destination`, `unknown_destination`, `url_rejected` | A navigation command, answered with a fixed reply. | No |
| `invalid_input` | Empty or too long (`error.code` `empty_input` or `input_too_long`). | No |
| `withheld` | The answer copied the instruction; the fixed `withheld` reply is shown instead. | Yes |
| `error` | `knowledge_unavailable`, `model_not_found`, `ollama_unavailable`, `timeout`, `ollama_error`, `invalid_response`, or `empty_output`. The fixed `unavailable` reply is shown. | Depends |

Nothing a model writes can navigate or add a link: actions come only from the router and the continuation route only from `configs/chat.yaml`. Output validation also removes hidden reasoning (`<think>` blocks), links and contact details (URLs, `www.` hosts, email and IP addresses, social media handles, phone numbers, paths, Markdown and HTML links, and domain names absent from the approved knowledge, so `rastinaghighi.dev` and `Three.js` stay as plain text), and Markdown or HTML formatting. It withholds an answer that copies 12 or more consecutive words of the instruction. It detects verbatim copies only, not paraphrases, and it is not a guarantee against prompt injection.

### Command line

From the repository root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m mascot_lab validate
.\.venv\Scripts\python.exe -m mascot_lab route "take me to QMIND"
.\.venv\Scripts\python.exe -m mascot_lab chat --model granite4:micro-h --dry-run "Where does Rastin study?"
```

- `validate` checks every configuration file, the instruction, and the knowledge block, and prints their hashes. It does not contact Ollama. `validate --ollama` also reads each configured model's metadata from the local server (`/api/show`): whether it is installed, its capabilities (including `thinking`), and its Modelfile parameters. No model is run.
- `route` runs input validation and routing only and prints JSON.
- `chat --model TAG_OR_NAME` sends one message to one configured model and prints the response as JSON. `--dry-run` prints the exact request instead of sending it, and `--show-prompt` includes the full system message. `--ollama-url` (loopback only) and `--timeout` (default 120 seconds) set the connection.

Exit status: 0 on success, 1 when a request fails or a configured model is not installed, 2 for invalid configuration or arguments.

### Configuration exceptions and known incompatibilities

The shared baseline in `configs/models.yaml`, set by the owner for Part 3: `num_ctx` 4096, `temperature` 0.0, `seed` 42, `top_p` 1.0, `num_predict` 160, and `think: false`. No model has an override.

- `think` is Ollama's top-level request field, not an option, so the configuration schema now has a `baseline.think` key. It is sent to all six models. For a model without Ollama's thinking capability there is no thinking mode to switch off, so the field is expected to change nothing; this has not been verified against the installed Ollama version. Which of the six report the capability has not been checked yet: `validate --ollama` lists it. Reasoning that a model returns anyway, in Ollama's `thinking` field or in `<think>` tags, is kept in the raw output, never displayed, and flagged (`thinking_returned`, `thinking_removed`). If Ollama rejects `think` for a model, the request fails with `ollama_error`; a per-model exception for `think` would then be needed, which the `overrides` schema does not support yet.
- Options outside the baseline (for example `top_k`, `min_p`, `repeat_penalty`, and stop sequences) come from each model's Modelfile and Ollama's defaults, so they can differ between models. `validate --ollama` prints each model's Modelfile parameters. Whether any of them changes greedy decoding (a repeat penalty, for example) is not verified here.
- With `temperature` 0.0 decoding is greedy, so the seed does not change the chosen tokens; it is set for completeness. Identical settings do not guarantee identical outputs across runs; repeated runs measure variability (handoff, section 5).
- Gemma's published chat format has only user and model turns. How Ollama's templates for the two Gemma tags present the shared system message has not been checked (`ollama show <tag> --template`).
- Requests are not streamed, so time to first token is not measured; the planned website streams answers (design plan, pp. 6 and 9). Counts and durations are recorded only as Ollama reports them; missing values stay empty.
- The system message is 7,670 characters. Its token count depends on each model's tokenizer and has not been measured. Ollama reports `prompt_eval_count` per request.

The first real run checked several of these points: which models report thinking, their Modelfile parameters, the Gemma templates, prompt token counts, and model loading. The findings are in `docs/model-selection-v1.md`, sections 2 and 3.

### Provisional values awaiting owner approval

The design plan does not fix these, so `configs/chat.yaml` marks them provisional:

- The input limit of 1,000 characters (the input cloud grows to about 3 small lines, then scrolls, p. 7).
- The cloud approximation of 3 lines of 45 characters. Pixel fit must still be checked visually once the cloud exists.
- The route labels and the wording of every fixed reply, including the `(link removed)` placeholder.
- Offering one `suggest` action per candidate for an ambiguous destination.
- Model answers carry no `suggest` action yet. The design plan lets a reply carry a "take me there" button but does not say when.

Off-topic filtering, rate limits, and the other planned safeguards come after the model is chosen (design plan, p. 8) and are not part of this engine.

## Evaluation infrastructure (Part 4)

The evaluation runner sends the same version-controlled cases through the shared chat engine for every configured model and records the machine results. Human grades are kept apart from those results, in review sheets graded with a provisional rubric. No code grades, scores, ranks, or selects a model. `docs/architecture.md` describes every file format in full.

`eval/cases.jsonl` holds the 24-case v1 evaluation set, written to the owner's specification: 20 cases answered by models and 4 navigation cases answered by the application. `tests/test_eval_cases.py` keeps it frozen. The first real evaluation ran on November 6, 2026; its results and the provisional selection are in `docs/model-selection-v1.md`. The other tests use synthetic cases and mocked Ollama responses only.

Files:

- `src/mascot_lab/cases.py`: loads and validates the JSON Lines case file and fails closed with line numbers.
- `src/mascot_lab/runner.py`: plans a run, asks Ollama (`/api/show`, which runs no model) whether every selected model is installed, sends every case through `ChatEngine.respond` for each model, and stops early when continuing could not produce comparable records.
- `src/mascot_lab/measurements.py`: the measured values of each response; unavailable values are null.
- `src/mascot_lab/recorder.py`: writes each run's directory under `results/`.
- `src/mascot_lab/review.py` and `eval/rubric.md`: the blank, blind review sheet and the provisional rubric it follows.

### Cases

`eval/cases.jsonl` holds one JSON object per line. `id`, `category`, and `message` are required. `expected_behavior` is optional guidance for reviewers, `expected_route` is an optional routing expectation that the runner checks without any model, and `notes` is for case authors. Reviewer guidance and author notes never reach a model. Format illustration only, not an evaluation case:

```json
{"id": "example-nav", "category": "navigation", "message": "take me to QMIND", "expected_route": {"status": "routed", "route_id": "project_qmind"}}
```

### Running

```powershell
.\.venv\Scripts\python.exe -m mascot_lab evaluate --dry-run
.\.venv\Scripts\python.exe -m mascot_lab evaluate --note "first baseline pass"
.\.venv\Scripts\python.exe -m mascot_lab evaluate --model granite4:micro-h --case CASE_ID --repeat 3
```

- `--dry-run` validates the cases and prints the plan: how many records and Ollama requests the run would make, which cases the application answers without a model, and whether each `expected_route` holds. It contacts nothing and writes nothing.
- `--model` (tag or configured name) and `--case` (ID) can be repeated to select a subset. Models always run in configuration order and cases in file order. `--repeat N` runs every selected case N times per model, to observe variability.
- `--cases` reads another case file. `--results-dir` writes somewhere other than `results/`: inside this repository only `results/` is accepted, because only it is ignored by Git. `--note` is stored in the manifest. `--ollama-url` and `--timeout` work as they do for `chat`.
- Each model's requests run back to back. Requests are sequential and never retried. An error response such as a timeout is recorded and the run continues. The run stops early (`aborted`) when Ollama cannot be reached or the knowledge block becomes unavailable or changes. Ctrl+C (`interrupted`) keeps every record written before it.

Exit status: 0 when every planned record was written (whatever the responses were), 1 when a model is not installed, Ollama cannot be reached, or the run stopped early, and 2 for invalid configuration, cases, or arguments.

### Results

Each run creates `results/<run_id>/`, for example `results/20261106T150000Z-3fa91c/`, and never reuses a directory:

| File | Content |
| --- | --- |
| `manifest.json` | Settings, every configuration hash, the knowledge hash, each model's effective options and `/api/show` capabilities and Modelfile parameters, the exact system message (stored once, keyed by its SHA-256), the rubric hash, counts, and the final status. |
| `records.jsonl` | One complete record per response: the case, the model, the status, the route check, the measurements, and the engine's full `ChatResponse` record (raw and validated output, prompt hashes, request settings, errors). |
| `records.csv` | The same records as flat columns for spreadsheets. |
| `review.csv` | One blank row per record for human grading. |

| Measurement | Source | Null when |
| --- | --- | --- |
| `pipeline_ms` | The runner's clock around the whole engine call | never |
| `elapsed_ms` | The Ollama client's clock around the HTTP request | no request was sent |
| `time_to_first_token_ms` | Not measured: requests are not streamed | always |
| `load_duration_ns`, `prompt_eval_duration_ns`, `eval_duration_ns`, `total_duration_ns` | Reported by Ollama | not reported |
| `prompt_tokens`, `output_tokens` | Ollama's `prompt_eval_count` and `eval_count` | not reported |
| `tokens_per_second` | `output_tokens` ÷ (`eval_duration_ns` ÷ 10⁹) | either input is missing, or the duration is zero |
| `answer_characters`, `word_count`, `answer_lines` | The displayed answer; lines wrap at the cloud's configured width | the answer is a fixed application reply, not a model answer |

A null is `null` in JSON and an empty cell in CSV; nothing is estimated. CSV files are UTF-8 with a byte-order mark. A text cell that begins with `=`, `+`, `-`, `@`, a tab, or a carriage return gets a leading apostrophe so that spreadsheets do not run model output as a formula; `records.jsonl` keeps the exact text. The results hold raw model output, and the manifest holds the private knowledge block: keep them local.

### Human review

`review.csv` lists, for each record, the message, the reviewer guidance, the status, the actions, the route check, the displayed answer, the raw model output, and the output issues, followed by one blank column for each rubric dimension: `factual_accuracy`, `scope`, `unknown_handling`, `disclosure_boundaries`, `safety`, `navigation`, and `brevity`. Then come `reviewer` and `notes`. It is blind: it shows no model name or tag and no timing or token measurement, and each case's rows are ordered by an opaque `record_id`, not by model. Grade a copy of the sheet (for example `review-<initials>.csv`) using `eval/rubric.md`, and never edit the machine records. `record_id` joins the grades to `records.csv` afterwards.

The rubric is provisional: its dimensions, its scale (`pass`, `partial`, `fail`, `n/a`), and its criteria await owner approval, and its open questions are listed at its end. It defines no weights, totals, thresholds, or decision rule.

### Limitations

- Navigation and invalid-input cases never reach a model, so all six records for such a case are identical. They test the application.
- The first request to each model can include loading it (`load_duration`). Ollama decides when to load and unload models, and CPU/GPU placement is not recorded.
- How Ollama reports `prompt_eval_count` when it reuses cached prompt tokens is unverified.
- The Ollama server version and model digests are not recorded, because the adapter has no request for them yet.
- Cases are single messages: the engine has no conversation history, so multi-turn steering cannot be tested yet.

## Status

Foundation (Part 1), deterministic navigation (Part 2), the shared chat engine (Part 3), and the evaluation infrastructure (Part 4) are implemented with automated tests, which use synthetic inputs and mocked Ollama responses.

On November 6, 2026 the 24-case evaluation set ran against all six models through this code, one pass per case and model. Phi-4-mini (`phi4-mini:3.8b-q4_K_M`) was selected provisionally, with Granite 4.2 3B (`granite4.2:3b-q4_K_M`) as the backup (`docs/model-selection-v1.md`). The grades behind the choice are an AI reviewer's provisional grades and await the owner's review.

Nothing is deployed. Hosting, production integration, and the planned safeguards are not built, and they block a public launch (`docs/future-improvements.md`).

## Repository policy

This repository is private during development. The authoritative design PDF, local evaluation outputs, and unapproved private research details are excluded from version control.

Git commits and pushes are performed manually by the repository owner.
