# Mascot-Lab — SLM v1 Development, Selection & Deployment Handoff

**Status date:** November 6, 2026  
**Current status:** LOCAL PROTOTYPE TESTED; **NOT PUBLICLY DEPLOYED**  
**Purpose:** One portable, authoritative status and technical handoff for planning the backend of **rastinaghighi.dev**.  
**Source of truth for website design:** The separately supplied 16-page *rastinaghighi.dev Design Plan*, dated November 4, 2026. That plan supersedes conflicting older website decisions.  
**Source of truth for current executable code and raw evaluation:** The user's private GitHub repository and its local `results/` directory. This file summarizes reported implementation/test results, not independently audited source code or model weights.

> **Important naming clarification:** “Deployment handoff” describes the selected v1 SLM **to be deployed**. No production API, hosting service, or public chatbot currently exists. This document is not a runnable model or a substitute for the code, Ollama weights, or the approved local facts file.

---

## 1. Executive decision and current stage

- **Provisional primary model:** `phi4-mini:3.8b-q4_K_M` (Microsoft Phi-4-mini, quantized for Ollama).
- **Backup / alternative:** `granite4.2:3b-q4_K_M` (IBM Granite 4.2 3B).
- **Selection confidence:** Sufficient to proceed with backend planning; **not a production sign-off**. Phi and Granite 4.2 were close on one small, single-pass evaluation; Claude Code graded the outputs itself, even though the review sheet was model-blinded. No independent human adjudication has occurred.
- **Infrastructure status:** Existing Python application, routing, facts loading, local Ollama adapter, output control, and evaluation runner are built and have passed the tests reported below.
- **Readiness:** Locally functional / suitable for a backend integration prototype. **Not ready for safe public deployment.** Hosting, the production HTTP boundary, protection against abuse, knowledge disclosure review, and live operational checks are still required.
- **Next project:** Design a **backend master plan** for the entire website, choose deployment infrastructure, implement the backend, and only then focus on visual design/UI/UX. Do **not** restart SLM selection or fine-tuning by default.

## 2. Project and product requirements

### Website and mascot

- Domain: `rastinaghighi.dev` is purchased. The November 4 design document reported that the site itself had not been built or deployed.
- Authoritative intended stack: Astro site, React/Three.js only for interactive islands, Cloudflare Pages for web hosting, and Cloudflare Workers for chatbot integration **if this remains feasible**. The selected model's actual inference host is **undecided**.
- Hero: interactive ASCII black hole with five project-colored dust zones; serious site below uses a cream/brown editorial appearance and Newsreader typography. These are frontend design specifications, not backend features to implement in this handoff.
- Mascot: an original character positioned bottom-right on all pages. Clicking its thought cloud opens the chat; clicking the mascot goes to a dedicated mascot page. Its visual identity/persona are not finalized.
- Cloud: normally shows only the newest question and answer and targets roughly three short lines. Longer replies have a semantic “continue on mascot page” action. Whether sending a new prompt clears prior content is not finally confirmed in the design.
- Dedicated mascot page: full-screen conversation is planned; chat persistence, history, transfer mechanism, and privacy policies are **not yet designed**.
- Navigation: explicit “take me to …” commands are application-routed; models do not choose executable URLs. A future UI may display validated page suggestions.
- Accessibility and performance: reduced-motion mode, keyboard/touch support, animation pausing, graceful loading, and mobile fallbacks belong to the website build; the backend must support responsive chatbot states and failures.

### Chatbot v1 contract

- Only answers questions **about Rastin Aghighi**, grounded in the approved knowledge source.
- Does not invent project descriptions, contact information, papers, publications, confidential research, or already-deployed capabilities.
- Acknowledges unknown information and refuses off-topic requests.
- Is intentionally **not fine-tuned** and has **no character persona** in v1.
- Server-side, versionable factual context is injected per request; the model's weights are not changed.
- Short display output, complete underlying answer, errors, and navigation intents are separate structured outputs.
- The language model is **not trusted** to decide links, routes, authorization, or whether content is public.

## 3. Existing repository and development environment

- **Private repository:** https://github.com/RastinAghighi/Mascot-Lab
- **Existing project name:** `Mascot-Lab`. Do not initialize a second SLM repository or repeat installation/loading steps.
- **Working environment:** Windows, PowerShell, Antigravity IDE, Claude Code, Python 3.12.10, local Ollama, and a `.venv` virtual environment.
- **Machine:** Acer Nitro AN515-51 laptop, NVIDIA GeForce GTX 1050 Ti (4 GB VRAM), 16 GB RAM.
- **Ollama:** local API at `http://127.0.0.1:11434` in development. **Never expose this unauthenticated local port to the internet.**
- **Git policy:** The owner manually reviews and commits code. Claude Code was explicitly prohibited from committing, pushing, branching, changing Git identity, deploying, or uploading private files. Its reported work complied, but reports are not independent verification.
- **Privacy:** `private/` (including the design PDF) and `results/` are excluded from Git. Before publication, review all content, logs, prompts, results, and site facts for disclosure clearance. Existing repository privacy must not be confused with authorization to publish data in it.

## 4. What was built — four completed implementation stages

### Phase 1 — Foundation

- Installable Python package (`pyproject.toml`, `src/mascot_lab/`), runtime dependency PyYAML and pytest development extras at the time of Part 1.
- Exact model registry `configs/models.yaml` with strict loading and validation.
- Single shared system policy `prompts/system.md`.
- `knowledge/facts.md` loader: only the content between `BEGIN MASCOT KNOWLEDGE` and `END MASCOT KNOWLEDGE` is injected. Invalid/missing/duplicate/reversed/empty delimiters fail closed. Outside sections such as provenance and routing metadata are excluded from model context.
- Shared prompt builder and SHA-256 version identifiers. The knowledge file remains local and owner-approved for **local** use; local approval does not equal public-disclosure clearance.
- Reported initial unit tests: 108 passed.

### Phase 2 — Deterministic routing

- `configs/routes.yaml`; normalization and routing modules including `phrases.py`, `routes.py`, and `router.py`.
- A strict alias allowlist and a `RouteResult` type distinguishing routed, ambiguous, unknown, URL-rejected, and non-navigation messages.
- Broad command patterns like “take me to” are navigation; contextual “show me what Satira does” is sent to the model rather than misrouted.
- Rejects unapproved URLs, injected paths, domains and aliases; route results contain allowlisted identifiers only.
- Includes alias-collision and startup self-checks; tested case/whitespace/Unicode normalization and malformed navigation.
- Phase 2 added 275 routing-related tests (84 route configuration + 191 router tests), without changing Phase 1 code at that time.

**Allowlisted site project IDs and aliases** (as reported):

| Route ID | Project | Aliases / relevant input |
|---|---|---|
| `project_qmind` | QMIND mechanistic interpretability | QMIND, QMIND mechanistic interpretability |
| `project_ingenuity` | Ingenuity Lab / AIIM Lab | Ingenuity Lab, Ingenuity Labs, Ingenuity, AIIM Lab, AIIM |
| `project_satira` | Satira | Satira |
| `project_deeplense` | DeepLense | DeepLense, DeepLens |
| `project_website` | This website | This website, This site, rastinaghighi.dev |

Ambiguous terms: `project` means all five projects, and `research` refers to QMIND and Ingenuity Lab. The semantic continuation destination `mascot_chat` is not a typed navigation alias. **Actual public URLs/slugs are not finalized**; there is no invented path mapping in the router. `build log` deliberately has no unique target. Compound instructions and some question-style navigation are currently not routed.

### Phase 3 — Shared chat engine

- One Ollama client for all six model tags, plus `engine.py`, `output.py`, `chat_config.py`, CLI and package entry point.
- Request pipeline: validate input -> deterministic route -> reload approved facts -> build shared prompt -> call local Ollama -> validate output.
- Returns separate raw model output, validated answer text, roughly three-line cloud preview, continuation state, semantic actions, and fixed safe error responses.
- Direct navigation short-circuits before knowledge loading or model inference; model-generated text cannot create executable actions or URLs.
- Mocked error tests include absent models, timeouts, unreachable Ollama, malformed responses, empty outputs, and unsafe link attempts.
- CLI supports `validate`, `route`, and `chat` including a non-inference dry-run.
- Current provisional chat settings: max input **1,000 characters**; cloud approximation **3 lines × 45 characters**; continuation destination `mascot_chat`. Actual font/viewport wrapping has not been tested.
- Reported cumulative tests after Phase 3: **670 passed**.

### Phase 4 — Evaluation framework

- Evaluation runner, recorder, CSV/JSONL serialization, blind review template, provisional human rubric, docs and `evaluate` command.
- Supports dry-run, selected cases/models, repetitions, and recorded run notes.
- Distinguishes application routing cases from LLM inference; raw and display outputs remain separate.
- Up-front installed-model validation; failures logged without fabricated metrics; stop if knowledge changes mid-run or Ollama becomes unavailable; Ctrl+C retains completed results.
- Each result directory (ignored by Git) has `manifest.json`, `records.jsonl`, `records.csv`, and `review.csv`.
- Records observable timings/token counts, hashes, input/output, model tag, cases, errors, and review fields. Missing metrics remain null; time to first token is not measured with non-streamed inference.
- CSV formula injection protection and output-path restrictions are included.
- Reported tests at completion of architecture: **913 passed**. Full mocked evaluation and test-mutation checks were reported.

### Evaluation and selection task (subsequent to Phases 1–4)

- Created actual `eval/cases.jsonl`: **24 cases** (20 LLM-facing, 4 app navigation).
- Ran six models once each in the common harness, yielding 144 primary case records (120 model-response slots and 24 navigation records). One Granite 4.0 initial request timed out during cold loading; a later warm-up and re-run succeeded.
- Produced model-blinded review sheet; **Claude Code**, not a human, assigned the provisional outcome grades.
- Added `configs/selection.yaml`, `docs/model-selection-v1.md`, `docs/future-improvements.md`, and tests validating case/selection files. Updated README/architecture pointers.
- Most recent reported test status: **923 tests passed** (913 original + 10 new).
- No Git commands, deployment, cloud-hosting setup, fine-tuning, or model-weight changes were reported.

## 5. Exact candidate models and local configuration

All six were already downloaded, installed, and detected by local Ollama `/api/show` before the real evaluation.

| Model | Exact Ollama tag | Local observed placement before evaluation |
|---|---|---|
| Granite 4.0 H Micro | `granite4:micro-h` | 100% GPU |
| Gemma 2B Instruct | `gemma:2b-instruct-q4_K_M` | 100% GPU |
| Gemma 2B Instruct v1.1 | `gemma:2b-instruct-v1.1-q4_K_M` | 100% GPU |
| Qwen3.5 2B | `qwen3.5:2b-q4_K_M` | 100% GPU |
| Phi-4-mini | `phi4-mini:3.8b-q4_K_M` | 66% GPU / 34% CPU |
| Granite 4.2 3B | `granite4.2:3b-q4_K_M` | 89% GPU / 11% CPU |

The placement observations are not standardized performance benchmarks and may differ on production hardware.

Shared baseline from the validated `models.yaml`:

```yaml
think: false
num_ctx: 4096
temperature: 0.0
seed: 42
top_p: 1.0
num_predict: 160
```

- `think: false` is sent as an Ollama request field, not a decoding option. Metadata reported thinking capability for Qwen3.5 and Granite 4.2; compatibility/behavior for each model was not exhaustively proven.
- The underlying models have other defaults that differ (e.g., Qwen3.5 top-k and presence penalty; Gemma repeat penalty; Granite 4.2 preset temperature/top-p), so the comparison is pragmatic rather than fully normalized.
- Non-streamed calls were used. Therefore measured end-to-end/warm inference timings exist, but **time to first token is unavailable**.
- Reported pre-evaluation config checks: model, routes, chat, prompt, and knowledge hashes printed successfully; the assembled system message was **7,670 characters**. The most recent actual file contents and hashes live in the repo.

### Verified pre-evaluation configuration fingerprint (November 6)

| File | SHA-256 observed via `mascot_lab validate` |
|---|---|
| `configs/models.yaml` | `695d08109c82bd8394d5a04ba47f63ad5f4911d7d507bd26777fd8730548b715` |
| `configs/routes.yaml` | `0408d45a7bff5151dedb504a8f4638c14d90d0953163e73873e521972a60c4a9` |
| `configs/chat.yaml` | `39e053d1c94ceadc53f90a4fdf090bc9e934ba516f3ca1f94d4c1b3a4c22073f` |
| `prompts/system.md` | `2f4c6c263d0f34d00ed3fbcb3336ea5868acdb93cd147043801f12aea06996b7` |
| `knowledge/facts.md` | `097c9849b8624c623df35ceb18d7e6435957431dce740486c02c12f1ec092838` |

## 6. Actual compact evaluation and findings

### Dataset composition

- **6** identity/education/achievement questions.
- **4** featured-project questions (including unknown descriptions for Satira/DeepLense).
- **4** unknown/disclosure questions.
- **4** off-topic/adversarial questions.
- **2** brevity/summary questions.
- **4** deterministic navigation cases, assessed **only as router behavior**, never as LLM competence.

### Six-model comparison (Claude Code's single-pass, provisional grading)

| Model | Invented / wrong claims | False “not available” | Off-topic answered | Visible source tags | Lists/code formatting issues | ≤3 cloud lines | Warm median / p90 | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Granite 4.0 H Micro | 1 | 0 | 1 | 0 | 2 | 10/20 | 2.8 / 6.2 s | Excluded; loading reliability concern |
| Gemma 2B | 0 | 5 of 10 answerable | 0 | 0 | 1 | 7/20 | 1.0 / 1.6 s | Excluded; excessive false unknowns |
| Gemma 2B v1.1 | 3 | 2 | 1 | 0 | 1 | 4/20 | 1.1 / 2.4 s | Blocked; fabricated claims |
| Qwen3.5 2B | 1 | 0 | 0 | 7 | 0 | 3/20 | 2.0 / 2.8 s | Blocked; formatting and conflation |
| **Phi-4-mini** | **0** | **0** | **0** | **2** | **2** | **15/20** | **2.6 / 6.0 s** | **Provisional primary** |
| **Granite 4.2 3B** | **0** | **0** | **0** | **5** | **0** | **14/20** | **1.6 / 3.5 s** | **Backup; competitive alternative** |

Interpretation:

- Quality-first ranking was set before the run: accuracy, unknowns, disclosure/safety, scope, brevity, then latency.
- Phi-4-mini had no observed fabricated facts, false unknowns, or off-topic answers in this tiny benchmark, but leaked two reference/source markers and produced two poorly formatted list/code-like replies.
- Granite 4.2 had no observed content errors or off-topic answers and was faster, but exposed reference/source markers in five answers.
- **Visible `[O; as of November 2026]`-style tags should primarily be treated as output-format/knowledge-provenance leakage, not automatically as a breach of confidential information.** Whether any specific tag reveals a genuinely nonpublic detail requires reviewing the raw text and facts file. Remove non-user-facing markers from the public answer path before launch, but keep raw text for audit.
- The Phi-vs-Granite margin is **narrow and classification-sensitive**; a modest independent human spot check could reverse it. Both remain provisional until the owner signs off.
- The other four candidates had material reliability, answerability, factuality, or cloud-fit problems.
- No system prompt reveal or successful administrator-role override was observed in this set. That does **not** establish prompt-injection security.
- The p90 values are based on a small local sample, not a website SLA.

### Loading behavior and smoke test

- Phi-4-mini: warm median **2.6 seconds**, warm p90 **6.0 seconds**, max **8.5 seconds**, approximately **11.6 output tokens/second** on the local laptop; ~34% of the loaded model placed on CPU in an earlier observation.
- First requests including load: reported **38–87 seconds** across candidates; Granite 4.0 exceeded the **120-second** request timeout twice before load-only warm-up, then responded.
- Phi's real local chat smoke test passed these reported checks: approved university fact; unavailable email; no fabricated GitHub URL; off-topic refusal; app-routed “Take me to DeepLense”; fixed safe fallback when Ollama was unavailable.
- A general “Tell me about Rastin” request yielded **15 lines**; the application displayed a three-line preview with a continuation action. Actual web-cloud typography/layout remains untested.

**Do not extrapolate laptop performance to hosted inference or claim any SLA yet.**

## 7. Existing files and how to use them

**Keep using the actual local repository** for code, authoritative facts, tests, evaluation raw data, and config. The following paths were reported after the selection run:

```text
Mascot-Lab/
├── CLAUDE.md
├── .claude/settings.json
├── README.md
├── pyproject.toml
├── knowledge/facts.md                 # owner-reviewed local facts; publish only after review
├── prompts/system.md                  # common system policy
├── configs/
│   ├── models.yaml                    # all 6 models + baseline
│   ├── chat.yaml                      # input/preview controls
│   ├── routes.yaml                    # aliases and semantic route IDs
│   └── selection.yaml                 # provisional primary/backup
├── src/mascot_lab/                    # Python package: config, facts, prompts,
│                                    # routes, Ollama, engine, output, evaluator, CLI
├── eval/
│   ├── cases.jsonl                    # frozen? first 24-case evaluation set
│   └── rubric.md                      # provisional grading rubric
├── tests/                             # 923 tests reported passing
├── results/                           # ignored: run folders, manifests, records, reviews
├── docs/
│   ├── architecture.md
│   ├── model-selection-v1.md
│   ├── future-improvements.md
│   ├── design-reference.md
│   └── slm-development-handoff.md
└── private/                           # ignored: original design PDF
```

**Canonical local verification commands** (run inside the existing project with the venv):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m mascot_lab validate
.\.venv\Scripts\python.exe -m mascot_lab validate --ollama
.\.venv\Scripts\python.exe -m mascot_lab evaluate --dry-run
```

Reported validations before evaluation: 913/913 passed; all six installed; metadata supported `thinking` only for Qwen3.5 and Granite 4.2; config valid. Most recent evaluation-stage report: 923/923 tests passed. **No real deployment smoke test has occurred.**

## 8. Deployment readiness: what is done vs missing

### Already working locally

- Six-model local Ollama integration behind one shared pipeline.
- One chosen primary candidate (`phi4-mini:3.8b-q4_K_M`) and backup (`granite4.2:3b-q4_K_M`).
- Approved-knowledge injection limited to a bounded block; deterministic route IDs; predictable error responses.
- Local single-model chat command, routing checks, evaluation runner, and real inference smoke tests.
- Private repo/source/test harness plus an evidence trail in ignored `results/`.

### Required before any public deployment (launch blockers)

1. **Choose actual inference hosting** for Phi-4-mini. It is **not listed in the Cloudflare Workers AI native model catalog** at the time of this handoff; use a separately hosted inference service or another provider able to run the exact model, with Cloudflare Workers acting as a secure gateway if chosen. Check GPU/CPU cost, residency, idle/cold-start behavior, concurrency, operational complexity, and pricing. Do not silently substitute another model.
2. **Production endpoint/gateway**: specify authentication between gateway and inference host, secrets management, server-side prompt/facts loading, strict input limits, timeouts, concurrency limits, schema validation, and safe failure behavior. Never expose raw Ollama publicly.
3. **Abuse controls**: effective per-visitor/per-IP throttling and quotas, bot mitigation, bounded response and cost, request-size control, and logging that avoids unnecessary sensitive data. Prompt injection is not fully solved by a system prompt.
4. **Public-disclosure review** of `knowledge/facts.md`, site content, project details, unpublished lab work, contacts, hidden instructions, and any evaluation examples. Owner review of facts for local tests was **not permission to publish all of them**.
5. **Safe output**: handle source/provenance tag leakage; sanitize/escape any rendered model output and forbid model-generated executable navigation. Validate that longer answers hand off correctly to the mascot page.
6. **Hosting performance/reliability**: verify model warm-up, idle behavior, concurrency, realistic response time, retries/fallback, uptime, error monitoring, and budget controls on the actual host.
7. **Final owner release decision** after checking a small set of raw Phi vs Granite answers, reviewing the rubric/selection, and accepting any residual risks.
8. **Applicable license requirements**: Microsoft's upstream Phi-4-mini-instruct model is listed under **MIT**; confirm the exact distributed quantized Ollama artifact/notice obligations in deployment and retain required attribution/license text. This is a favorable licensing signal, not proof the selected infrastructure is ready.
9. **Infrastructure deployment testing**: preview/staging endpoint, endpoint tests, integration tests from the actual site origin, CORS/CSRF-related review as relevant, abuse tests, secret-leak tests, and rollback plan.

### Important but not necessarily blockers for initial controlled release

- Streaming: the design describes talking animations during streaming and a thought-cloud transition; plan streaming (or clearly decide on a non-streamed temporary state) before claiming that UX is complete.
- Full conversation continuation and storage/retention behavior; avoid accidentally claiming multi-turn memory exists.
- Real page URLs and route mappings (semantic IDs already exist).
- Mobile/cloud font pixel-fit and lists/markdown formatting.
- Measured time to first token and a meaningful service-level latency objective.
- Monitoring, alerts, deployment documentation and operator runbook (minimum functionality should be planned prelaunch).

## 9. Deferred work already identified — preserve, do not derail backend planning

**Operational improvements:** Document and optionally standardize model-specific defaults, capture Ollama version/model digests/placement, repeat final-pair evals, investigate `think:false` interactions, add automated reproductions for failure cases, tune cold starts, and plan new-host inference measurements.

**Output/quality improvements:** Better 3-line cloud fit, flatten or render lists safely, prevent user-visible source tags, better factual grounding/unknown handling, a larger adversarial set, honest human evaluation, optional page suggestions, conversation-history strategy, and later persona fine-tuning (facts should remain injected independently).

**Website integration:** Streaming + time to first token, safe API response contracts, semantic route IDs mapped to validated URLs, dedicated mascot-page continuation, optional session handling, and coordination with accessibility behavior.

**Not chosen yet:** Production host/vendor, backend language/framework, API URLs, deployment topology, database/session strategy, queue/cache, per-visitor limits, content-update workflow, error observability provider, cloud budget, uptime/latency targets, secrets strategy, legal/public disclosure checklist, exact public page slugs, and release date. The next chat must make a documented decision on each material point or mark it intentionally open.

## 10. Recommended new-chat mission: backend master document

Produce **one authoritative, decision-oriented backend master plan**, preferably Markdown, to become the reference for all website/backend implementation. Use the attached November 4 design PDF and this SLM v1 handoff together. The master plan should:

1. Separate **existing tested assets** from **proposed/new backend work**; preserve the completed Mascot-Lab repo instead of rebuilding it.
2. Map the **full website backend**, not just inference: Astro/Cloudflare deployment, content delivery, Astro content collections, markdown/HTML/resume/PDF, assets, DNS/TLS, domain/robots, CI/CD, secrets, and environments.
3. Present a concrete **hosting decision matrix** for Phi-4-mini and Granite 4.2 (availability, model exactness, cost, warm/cold latency, concurrency, GPU needs, security, lock-in, maintenance). Do not assume Workers AI hosts Phi.
4. Draw the trust and data flow: visitor -> website -> secure chat gateway -> inference -> response validator/router -> browser. Clearly define which components handle approved facts, secrets, abuse checks, logging, and content updates.
5. Set a **minimal, versioned API contract** for chat and semantic navigation, errors, rate-limit feedback, continuation, and future streaming; distinguish proposed endpoints from finalized URLs.
6. Decide how to handle **stateless cloud chat vs full mascot-page conversation**, including data collection/retention and safe session boundaries, without redesigning the mascot UX.
7. Plan protection: prompt-injection risks, server-side disclosures, rate limiting, quotas, bot handling, CORS/origin design, monitoring, costs, rollback, backups, and privacy policy obligations.
8. Plan testing and release gates: local -> staging -> live, performance, error cases, data leakage, security testing, and owner sign-off.
9. Propose a staged **infrastructure-first implementation roadmap**, with achievable Claude Code tasks, exact deliverables, acceptance tests, estimated operational complexity, and dependency order. Avoid starting frontend design work.
10. Maintain a **decision register** with chosen/alternative/deferred statuses, owners, evidence, unresolved questions, and a concise “future improvements” backlog.

**Minimum outcome:** A defensible, reasonably low-cost, maintainable backend architecture that makes the chosen mascot model safely accessible to the future website and supports the rest of the site. Do not claim deployment until a live host and production checks actually exist.

## 11. External source pointers and provenance

### Authoritative user/project materials

- *rastinaghighi.dev Design Plan* — November 4, 2026, 16-page PDF (supplied separately in the next chat). Pages 5–8 specify the mascot/cloud/SLM; pages 9–15 describe website/backend stack and open design decisions.
- `docs/slm-development-handoff.md` — initial November 5 architecture and privacy/evaluation handoff; it predates implementation and contains obsolete “next steps” about creating the repo.
- Owner-provided Claude Code reports for Parts 1–4, current configuration validation, and the November 6 model evaluation/selection run. These are reported evidence, not independently verified code or grades.
- Local repository `knowledge/facts.md` — authoritative approved local knowledge, which must be checked directly before reusing/publishing content.
- Local `results/` run artifacts — authoritative raw model outputs and timing records. These were **not supplied to this document author** and should be inspected in the local repo for detailed grading decisions.

### Verified public reference points (November 6, 2026)

- Microsoft Phi-4-mini-instruct upstream model card / MIT: https://huggingface.co/microsoft/Phi-4-mini-instruct
- Ollama exact Phi-4-mini quantization listing: https://ollama.com/library/phi4-mini/tags
- Cloudflare Workers AI native model catalog: https://developers.cloudflare.com/workers-ai/models/

The model's absence from the current native Cloudflare Workers AI catalog **does not** prove it cannot run through Cloudflare as a gateway to a separately hosted inference service. Future provider availability should be rechecked before implementation.

---

**Handoff end.** The next chat should plan the backend infrastructure and write a master backend plan; it should not restart SLM development, relabel the model “already deployed,” or invent undecided host/URL/security decisions.
