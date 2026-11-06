# Mascot-Lab architecture

**Status, November 6, 2026:** Parts 1–4 are implemented and covered by automated tests that use mocked Ollama responses only. No model has been run through this code, no evaluation case set exists yet, no evaluation has been run, and no model has been selected.

**Authoritative sources:** the *rastinaghighi.dev Design Plan* of November 4, 2026 (kept privately in `private/`, never committed) for every website and mascot decision; `docs/slm-development-handoff.md` for the evaluation requirements; and `knowledge/facts.md` for the approved facts. This document describes the code. It does not change those documents or decide anything they leave open.

## 1. Purpose and boundaries

Mascot-Lab is one shared, local Python harness that compares six candidate Ollama models for the planned mascot chatbot. All six get the same instruction, the same approved knowledge, the same deterministic navigation, the same output validation, and the same decoding baseline, and any exception has to be written into the configuration with a reason.

- **Version 1 is a plain small language model:** no persona, no fine-tuning, no training (design plan, pp. 7–8).
- **Facts are supplied server-side on every request:** only the bounded `MASCOT KNOWLEDGE` block of `knowledge/facts.md`, read again for each request.
- **Navigation belongs to the application:** commands resolve to allowlisted semantic route IDs before any model is called, and nothing a model writes can navigate or add a link.
- **Evaluation measures and records; people grade:** machine records and human grades live in separate files. No code grades, scores, weights, ranks, or selects a model.

## 2. Repository layout

```text
Mascot-Lab/
├── configs/
│   ├── models.yaml        six exact tags, one shared baseline, documented overrides (none)
│   ├── routes.yaml        navigation allowlist: semantic route IDs, aliases, ambiguous terms
│   └── chat.yaml          input limit, cloud approximation, labels, fixed replies (provisional values)
├── prompts/system.md      the one shared system instruction, with {{MASCOT_KNOWLEDGE}}
├── knowledge/facts.md     owner-approved facts; only the marked block reaches a model
├── eval/
│   ├── rubric.md          provisional human-review rubric
│   └── cases.jsonl        the evaluation set: not written yet (the owner writes and freezes it)
├── src/mascot_lab/        the package (section 3)
├── tests/                 automated tests; synthetic inputs and mocked Ollama only
├── docs/                  handoff, design reference, this document
├── results/               created by evaluation runs; ignored by Git
└── private/               the design PDF; ignored by Git
```

## 3. Components

| Part | Module | Responsibility |
| --- | --- | --- |
| 1 | `config.py` | Loads `configs/models.yaml` with a strict YAML loader (no duplicate, merge, or unknown keys). Each model gets the shared baseline plus an optional `overrides` block that needs a `reason`. |
| 1 | `knowledge.py` | Returns only the text between the knowledge markers and fails closed on missing, duplicated, out-of-order, or malformed markers. |
| 1 | `prompt.py` | Builds the same system message (instruction plus knowledge) and user message for every model. |
| 1 | `_textfile.py`, `paths.py`, `errors.py` | Strict UTF-8 reading with LF line endings, SHA-256 hashing, project paths, and the error types. |
| 2 | `phrases.py`, `routes.py`, `router.py` | One normalization for aliases and messages, the validated route allowlist, and `Router.route()`, which returns a `RouteResult` holding only allowlisted route IDs. |
| 3 | `engine.py` | `ChatEngine`: the six-step request pipeline (section 4) and the structured `ChatResponse`. |
| 3 | `ollama.py` | The single Ollama adapter: non-streamed `/api/chat` and `/api/show`, loopback only, no proxies or redirects, size-limited responses, and an allowlist of option names. |
| 3 | `output.py` | Derives display text from the raw answer: removes reasoning, links and contacts, and formatting, withholds verbatim copies of the instruction, and builds the cloud preview. |
| 3 | `chat_config.py` | Loads `configs/chat.yaml`. |
| 3, 4 | `cli.py` | `python -m mascot_lab validate | route | chat | evaluate`. |
| 4 | `cases.py` | Loads and validates the JSON Lines case file. |
| 4 | `runner.py` | Plans an evaluation, checks the models with Ollama, runs every case through `ChatEngine.respond`, and applies the stop rules. |
| 4 | `measurements.py` | The measured values of one response, null when unavailable. |
| 4 | `recorder.py` | Writes the run directory: manifest, JSONL and CSV records, and the review sheet. Holds the results-directory guard. |
| 4 | `review.py` | The rubric's dimension columns and the blind, blank review sheet. |

Dependencies point one way: configuration and knowledge, then the prompt, router, adapter, and output modules, then the engine, then the evaluation modules, with the command line on top. The evaluation modules use the engine exactly as the `chat` command does. They never build prompts, call Ollama, or validate output themselves, so an evaluated answer is produced the same way an answer to a visitor would be.

## 4. Request pipeline: one message, one model

```text
visitor message
  │
  ├─1─ input validation ── empty or over input.max_characters ──► invalid_input (fixed reply)
  │      (LF line endings; control and invisible format characters removed; trimmed)
  ├─2─ deterministic routing (configs/routes.yaml)
  │      navigation command ──► routed / ambiguous_destination / unknown_destination / url_rejected
  │                             (fixed reply + semantic actions; no model is called)
  ├─3─ knowledge loading (knowledge/facts.md, marked block only, read for every request)
  ├─4─ prompt construction (prompts/system.md + knowledge; identical for every model)
  ├─5─ Ollama request (/api/chat: exact tag, messages, think, options from configs/models.yaml)
  └─6─ output validation ──► answered / withheld / error (fixed reply)
                                   │
                                   ▼
                              ChatResponse
          display: answer_text, cloud_preview, continuation, actions
          record:  model, message, route, prompt hashes, request settings,
                   raw output (content, thinking, done_reason, Ollama counts and
                   durations, client-measured elapsed_ms), output_issues, error,
                   configuration hashes
```

`ChatResponse.to_dict()` is the engine's record format, and the evaluation stores it unchanged inside each record. The README describes every response status and error code.

## 5. Evaluation pipeline

```text
eval/cases.jsonl ──► cases.load_cases ──► CaseSet (validated; route IDs checked against the allowlist)
                                             │
                     runner.plan_evaluation  (steps 1–2 per case; no knowledge read, no Ollama contact)
                                             │
                                       EvaluationPlan ──► `evaluate --dry-run` prints it and stops
                                             │
EvaluationRunner.run
  1. /api/show for every selected model: installed? capabilities, Modelfile parameters
     (no model is run; skipped when no case needs a model; a missing model stops the run before any file is written)
  2. results/<run_id>/manifest.json, status "running"
  3. for each model (configuration order)
       for each repetition
         for each case (file order):  ChatEngine.respond(message, model)
            ──► record = run and case identity, model, status, route_check,
                         measurements, ChatResponse.to_dict()
            ──► records.jsonl and records.csv (appended and flushed one record at a time)
     stops early:  Ollama unreachable, or knowledge unavailable or changed  ──► "aborted"
                   Ctrl+C                                                 ──► "interrupted"
                   an unexpected exception                                ──► "failed" (then raised)
  4. review.csv: one blank, blind row per record
  5. manifest.json replaced: final status, stop reason, counts, finish time
                                             │
human review: grade a copy of review.csv with eval/rubric.md; grades never go into the machine files
```

The schedule runs one model's requests back to back, so Ollama loads each model once per run instead of once per case. On a GPU with 4 GB of memory that matters (handoff, section 3). A response that is an error, such as a timeout, is recorded and the run continues. Requests are never retried, because a retry would hide the failure and distort the timing.

## 6. Data contracts

### 6.1 Case file (`eval/cases.jsonl`)

One JSON object per line, in a stable order; blank lines are ignored. Unknown keys, duplicate keys, duplicate IDs, non-finite numbers, lone surrogates, and values of the wrong type are rejected with their line number.

| Key | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Unique; lowercase letters and digits joined by single `.`, `_`, or `-`; at most 64 characters. |
| `category` | yes | Same form; groups cases, for example by the handoff's protocol areas. |
| `message` | yes | The exact visitor message, sent through the engine unchanged. Must not be the empty string; a whitespace-only message tests the empty-input reply. |
| `expected_behavior` | no | Guidance for reviewers, shown on the review sheet and never sent to a model. |
| `expected_route` | no | `{"status": <router status>, "route_id": ..., "candidates": [...]}`, checked by the runner without a model. `route_id` is required with `routed` and not allowed otherwise; `candidates` (a set of at least two) is allowed only with `ambiguous_destination`. IDs must be allowlisted. |
| `notes` | no | Notes for case authors. Recorded, never shown on the review sheet, never sent to a model. |

Format illustration only, not an evaluation case:

```json
{"id": "example-nav", "category": "navigation", "message": "take me to QMIND", "expected_route": {"status": "routed", "route_id": "project_qmind"}}
```

The engine is single-turn, like the cloud, so cases are single messages. Multi-turn steering would need conversation history, which neither the engine nor the case format has.

### 6.2 Run directory (`results/<run_id>/`)

The run ID is the UTC start time plus a random suffix, such as `20261106T150000Z-3fa91c`. A run never reuses a directory or overwrites a file.

| File | Content |
| --- | --- |
| `manifest.json` | `schema_version`; `run_id`; `status` (`running`, `completed`, `interrupted`, `aborted`, `failed`) and `stop_reason`; `note`; `started_at` and `finished_at` (UTC); `software` (package, Python, and platform versions); `ollama` (base URL, timeout); `order`; `cases` (source, SHA-256, count, selected IDs); `repetitions`; `models` (name, tag, `think`, effective options, override, and the `/api/show` capabilities and Modelfile parameters); `configs` (SHA-256 of each configuration file and the instruction); `knowledge` (source, SHA-256); `system_messages` (the exact system message of every request, keyed by SHA-256); `rubric` (source, SHA-256); `planned_records`, `planned_model_requests`, and `records_written`; `status_counts` (records by status); `route_checks` (records by route check); `files` (the file names). |
| `records.jsonl` | One complete record per response (section 6.3), exact, UTF-8. |
| `records.csv` | The same records as flat columns (section 6.4), UTF-8 with a byte-order mark. |
| `review.csv` | The blank, blind review sheet (section 6.6). |

The system message, which contains the private knowledge block, is stored once in the manifest instead of in every record, so a single record can be quoted without the knowledge block. Each record's `system_message_sha256` keys into the manifest, so the exact request is reproducible: the system message, the record's `user_message`, and its request settings.

### 6.3 Records (`records.jsonl`)

| Field | Content |
| --- | --- |
| `schema_version`, `run_id`, `record_id`, `sequence`, `started_at` | Identity and order. `record_id` is a 16-hex-digit SHA-256 prefix of the run ID, model tag, case ID, and repetition: deterministic, but it does not reveal the model. |
| `repetition` | 1 to N. |
| `case` | The case's canonical form and its SHA-256. |
| `model` | Configured name and exact tag. |
| `status` | The `ChatResponse` status. |
| `request_sent` | Whether the engine sent (or attempted) an Ollama request. |
| `route_check` | `match` or `mismatch` against the case's `expected_route`; null when the case sets none. |
| `measurements` | Section 6.5. |
| `response` | `ChatResponse.to_dict()`, unchanged: validated answer, cloud preview, continuation, actions, validated message, route, prompt (user message and hashes), request settings, raw output (content, thinking, done reason, Ollama's counts and durations, `elapsed_ms`), output issues, error, and configuration hashes. |

### 6.4 Flat records (`records.csv`)

One row per record. The columns are identity (`run_id`, `sequence`, `record_id`, `started_at`, `case_id`, `category`, `repetition`, `model_name`, `model_tag`), outcome (`status`, `request_sent`, `route_status`, `route_id`, `route_candidates`, `route_check`, `actions`), text (`message`, the case's message; `answer_text`; `continuation`; `output_issues`; `raw_output`; `raw_thinking`; `done_reason`), errors (`error_code`, `error_message`, `http_status`), the measurements of section 6.5, and hashes (`models_sha256`, `routes_sha256`, `chat_sha256`, `system_instruction_sha256`, `knowledge_sha256`, `system_message_sha256`, `case_sha256`). An empty cell means null. Lists are joined with `"; "`, and booleans are `true` or `false`. A text cell that begins with `=`, `+`, `-`, `@`, a tab, or a carriage return gets a leading apostrophe, so a spreadsheet program does not run model output as a formula. `records.jsonl` keeps the exact text.

### 6.5 Measurements and the null policy

Every value is measured by the harness, reported by Ollama, or computed exactly from those. A value that was not measured or not reported is null, never an estimate (`measurements.py`).

| Field | Source | Null when |
| --- | --- | --- |
| `pipeline_ms` | Runner's clock around the whole `ChatEngine.respond` call | never |
| `elapsed_ms` | Ollama client's clock around the HTTP request; also set for failed requests | no request was sent |
| `time_to_first_token_ms` | Not measured: requests are not streamed | always |
| `load_duration_ns`, `prompt_eval_duration_ns`, `eval_duration_ns`, `total_duration_ns` | Ollama, in nanoseconds | not reported, or no response |
| `prompt_tokens`, `output_tokens` | Ollama's `prompt_eval_count`, `eval_count` | not reported, or no response |
| `tokens_per_second` | `output_tokens` ÷ (`eval_duration_ns` ÷ 10⁹), 3 decimals | either input null, or zero duration |
| `answer_characters`, `word_count`, `answer_lines` | The displayed answer; lines are word-wrapped at the cloud's configured width, exactly as the cloud preview wraps them | status is not `answered`: the display text is then a fixed application reply, not a model answer |

### 6.6 Review sheet (`review.csv`) and human grades

The columns are `record_id`, `case_id`, `category`, `message`, `expected_behavior`, `status`, `actions`, `route_check`, `answer_text`, `continuation`, `raw_output`, `output_issues`, one blank column per rubric dimension (`factual_accuracy`, `scope`, `unknown_handling`, `disclosure_boundaries`, `safety`, `navigation`, `brevity`), then `reviewer` and `notes`.

The sheet is blind: no model name or tag, no timing or token measurement, and within each case the rows are ordered by `record_id` rather than by model. The provisional grade values are `pass`, `partial`, `fail`, and `n/a`, defined in `eval/rubric.md`. Reviewers grade a copy of the sheet, and `record_id` joins the grades to `records.csv` afterwards. The code never fills in, reads, or aggregates grades.

### 6.7 Identity and hashes

Every hash is the SHA-256 of UTF-8 text with LF line endings, so it does not depend on the platform or on Git's line-ending settings.

| What | Hashed form | Where recorded |
| --- | --- | --- |
| `configs/models.yaml`, `routes.yaml`, `chat.yaml` | Canonical JSON of the parsed file, so comments and layout do not count | manifest `configs`; every record's `response.configs`; CSV |
| `prompts/system.md` | The instruction template | same |
| Knowledge block | The text between the markers | manifest `knowledge`; record `prompt.knowledge_sha256`; CSV |
| System message | Instruction plus knowledge, exactly as sent | manifest `system_messages` (with the text); record prompt; CSV |
| Case file / one case | Canonical JSON of all cases in order / of the case | manifest `cases`; record `case.sha256`; CSV |
| `eval/rubric.md` | The file text | manifest `rubric` |

## 7. Trust boundaries and safety controls

- **The knowledge block is private.** The adapter talks only to a loopback Ollama, ignores proxies, never follows redirects, and limits response size. Only the marked block of `facts.md` is injected, never the navigation registry or the provenance notes.
- **Evaluation output is private.** The manifest holds the knowledge block and the records hold raw model output. Inside the repository, the recorder writes only under `results/`, which `.gitignore` excludes, and it refuses any other directory in the repository. Results are never uploaded, deployed, or committed by this code.
- **Navigation is allowlisted.** Route IDs come only from `configs/routes.yaml`, the continuation link only from `configs/chat.yaml`, and output validation removes any link a model writes. No URL or page path exists anywhere in the configuration.
- **Output validation is not a security guarantee.** It removes links, contact details, formatting, and reasoning, and it withholds verbatim copies of 12 or more words of the instruction, but it does not catch paraphrased leaks. Prompt-injection resistance is what the evaluation observes, not something the code promises.
- **Untrusted model output stays inert.** CSV cells that could run as spreadsheet formulas are prefixed with an apostrophe. JSONL escapes the characters that some line splitters treat as line breaks.
- **Evaluation integrity.** Every model gets the same messages, options, and `think` field. Only the tag differs, plus any documented override in `configs/models.yaml` (there are none). Reviewer guidance and author notes never reach a model. A knowledge block that changes or disappears mid-run stops the run, because later models would otherwise get different facts.

## 8. Configuration exceptions and known limitations

The shared baseline is `num_ctx` 4096, `temperature` 0.0, `seed` 42, `top_p` 1.0, `num_predict` 160, and `think: false`. **Documented exceptions: none.** The evaluation adds no per-model settings.

- **Thinking:** `think: false` is sent to all six models. Which of them report Ollama's thinking capability has not been checked yet; every run's manifest records the `/api/show` capabilities. Reasoning returned anyway is kept in the raw output, never displayed, and flagged.
- **Modelfile parameters** outside the baseline (stop sequences, `repeat_penalty`, and so on) can differ between models. Every run records them, but whether any of them affects greedy decoding is not verified.
- **Gemma templates:** how the two Gemma tags present the system message is unverified (`ollama show <tag> --template`).
- **No streaming,** so time to first token is not measured; the planned site streams answers (design plan, pp. 6 and 9).
- **Model loading:** the first request to each model can include loading it. Ollama reports that time as `load_duration`, which is included in `total_duration` and in `elapsed_ms`. When models are loaded and unloaded is up to Ollama (its `keep_alive`), not to the runner. CPU/GPU placement (`ollama ps`) is not recorded.
- **Prompt caching:** with an identical system message on every request, Ollama may reuse cached prompt tokens. How that affects `prompt_eval_count` and `prompt_eval_duration` depends on the Ollama version and is unverified.
- **Not recorded:** the Ollama server version and model digests, because the adapter has no request for them yet. Recording them would need an adapter change.
- **Sequential and unretried:** a model that times out on every case takes about the timeout (120 seconds by default) per case. Ctrl+C keeps everything recorded so far.
- **Text-level cloud fit:** the cloud preview approximates "about 3 short lines" with 3 lines of 45 characters (provisional). The real fit must be checked visually.
- **Blind review** depends on reviewers not opening `records.csv` or `records.jsonl` before grading.
- **Identical results across models:** navigation and invalid-input cases never reach a model, so all six records for such a case match. They test the application, not the models.

## 9. Decisions left to the owner

- The final ~20–30 evaluation cases in `eval/cases.jsonl`, written and frozen by the owner. None exist yet.
- Approval or revision of `eval/rubric.md`: the dimensions, the scale, the anchors, blind review, and the reviewer process. Its open questions are listed at its end.
- The provisional values in `configs/chat.yaml` and the navigation decisions recorded in `configs/routes.yaml` (README, Parts 2 and 3).
- How results are compared, which models are repeated, and which model is chosen. Per the handoff (section 7), that happens only after real runs and human grading. No code here ranks or selects.
- Licensing, hosting, Cloudflare Workers AI compatibility, and deployment (handoff, Phase 2).
