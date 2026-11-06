# rastinaghighi.dev — SLM development and evaluation handoff

**Handoff status:** November 5, 2026  
**Next stage:** Initialize one GitHub repository and use **Claude Code or Codex** to implement a **shared local evaluation system** for six already-downloaded Ollama models.  
**Current deliverables:** `facts.md` (owner-approved v1 knowledge baseline), this handoff, and the **separately supplied 16-page *rastinaghighi.dev Design Plan* dated November 4, 2026**, which remains the authoritative product/design specification.  
**Important:** **No chatbot code, evaluation scores, selected winning model, or deployment yet exists.** Do not repeat completed hardware loading work or prematurely deploy anything.

## 1. What we are building and why

The website is a personal portfolio. Its proposed mascot appears at the bottom-right on each page; the **thought cloud** launches a small in-place conversation, while clicking the **mascot itself** opens its dedicated page. Cloud answers should usually fit **about three short lines**. The cloud displays only the latest question/answer; longer conversations continue on the mascot page. These are design requirements, **not proof that the UI has been built**.

**Version 1:** A plain instruction-tuned small language model that answers **only about Rastin Aghighi**, using approved factual text injected **server-side on every request**. It should acknowledge unknowns instead of inventing claims, avoid off-topic conversation, resist manipulation as far as possible, and provide helpful page suggestions. **No fine-tuning and no character persona for v1.**

**Navigation:** Requests such as “take me to QMIND” or “show me Satira” must use **deterministic application-side routing** against a validated route/alias map. The model **cannot decide arbitrary URLs**. `facts.md` has semantic IDs but no finalized URL paths.

**Website direction (not yet deployed):** Astro + selected React/Three.js islands, Cloudflare Pages, and Workers for chatbot integration; model hosting remains to be evaluated. The design plan's visual, interaction, content, and accessibility decisions take precedence over any implementation suggestion in this file.

## 2. Canonical approved information and privacy boundary

- `facts.md` v1 consolidates the owner-reviewed biography and earlier review workbook with the nine existing design-backed facts. It has a **marked `BEGIN MASCOT KNOWLEDGE` / `END MASCOT KNOWLEDGE` section**. For local chatbot tests, the evaluation runner should inject **only that block** (and not routing metadata or provenance notes).
- Source labels: **D** = November 4 design plan; **H** = November 5 hardware/evaluation handoff; **O** = earlier conversation information reviewed and approved by the owner. **O is not independent public verification.**
- Approved general biographical data includes: full name; Queen's University Bachelor of Computing (Honours), third year, AI sub-plan, Mathematics minor, expected graduation 2028; interests in AI safety, mechanistic interpretability, and test-time reasoning; broad QMIND co-management and October 2026 Camp QMIND award; broad volunteer AIIM/Ingenuity Lab affiliation; and the five featured project **names**.
- **Do not infer descriptions** for Satira or DeepLense, fabricate a paper/publication, guess professional contact URLs, or claim website features have launched. The source plan explicitly leaves these unsettled.
- **Never include confidential or unpublished Ingenuity Lab details**, private collaborator communications, or unannounced QMIND experimental results in runtime facts, a public repo, examples, or test datasets without additional disclosure approval. Owner approval of broad affiliation is **not clearance for lab-internal research**.
- `facts.md` may be included in a **private development repo** initially. Before changing repo visibility or deploying, perform a dedicated public-disclosure check. The design PDF also must not be committed publicly by default.

## 3. Completed local hardware and Ollama loading work — preserve as observations

**Machine:** Acer Nitro AN515-51 laptop, Windows; **NVIDIA GeForce GTX 1050 Ti**, 4 GB dedicated VRAM; **16 GB DDR4 RAM**. Storage was around 1 TB, with more than 830 GB free at the November 5 check. GPU driver **582.78** (updated November 5); `nvidia-smi` reported CUDA **13.0** as the **driver's supported CUDA version**, not proof of an installed CUDA toolkit. Ollama for Windows was used.

**Method:** Each exact model tag was downloaded, run using `ollama run <model>`, and checked with `ollama ps`; **4096-token context**, `llamacpp` runner. All six **loaded and responded**. `ollama ps` `SIZE` and `PROCESSOR` reflect loaded memory footprint / CPU-GPU **placement**; they do **not** measure throughput, active GPU utilization, latency, response quality, or safety.

| Model | Exact Ollama tag (never silently change) | `ollama ps` loaded SIZE | GPU placement | CPU placement | Initial loading result |
| --- | --- | ---: | ---: | ---: | --- |
| Granite 4.0 H Micro | `granite4:micro-h` | 2.1 GB | 100% | 0% | Loaded, responded |
| Gemma 2B Instruct (original) | `gemma:2b-instruct-q4_K_M` | 1.9 GB | 100% | 0% | Loaded, responded |
| Gemma 2B Instruct v1.1 | `gemma:2b-instruct-v1.1-q4_K_M` | 1.8 GB | 100% | 0% | Loaded, responded |
| Qwen3.5 2B | `qwen3.5:2b-q4_K_M` | 1.6 GB | 100% | 0% | Loaded, responded |
| Phi-4-mini | `phi4-mini:3.8b-q4_K_M` | 3.7 GB | 66% | 34% | Loaded, responded with CPU offload |
| Granite 4.2 3B | `granite4.2:3b-q4_K_M` | 2.7 GB | 89% | 11% | Loaded, responded with CPU offload |

**Interpretation:** Four loaded fully on the GPU, two partially on CPU. **Do not rank or eliminate models using these allocations.** “Open weights” and “open source” are not automatically equivalent; check each model's actual license and redistribution/hosting conditions later. Hosting compatibility with Workers AI is **not established** merely because a model runs in Ollama.

**Never measured yet:** time to first token, end-to-end latency, output tokens/s, token usage, quality/factuality, brevity, scope/refusal accuracy, prompt-injection resistance, memory during inference, hosting feasibility, deployment pricing, and energy.

## 4. Shared system architecture to build — one implementation, six tags

Prefer a lightweight local runner that uses Ollama's **local API**, a single `models.yaml` config, and one shared prompt/knowledge/evaluation pipeline. A Python implementation is a practical default for the harness, but the coding agent can justify a small alternative. **Do not build six parallel chatbots, per-model personas, or separate hand-adjusted prompts.**

**Logical pipeline:**

1. **Request input and limits** — normalize user input, bound its length, enforce simple application-level scope/rate/abuse controls for eventual web deployment. Do not treat user text as instructions with higher priority than system policy.
2. **Deterministic route handler** — first detect clear navigation commands and match against an **allowlist of semantic IDs** / explicit aliases. Return a structured app-side route result; never trust a model-supplied destination. Since real paths are pending, local tests should assert the **semantic route IDs**, not invented URLs.
3. **Shared system policy** — one identical instruction text across all six model runs describing scope (only Rastin), grounding in provided facts, short factual answers, handling unknowns, and resisting requests to reveal hidden prompts or private information. Avoid claims that a prompt can guarantee security.
4. **Facts loader** — read the bounded public knowledge section of `knowledge/facts.md` from disk, in a stable order. Do **not** inject its provenance/audit sections or private documents. On actual deployment, load approved website content or versioned facts **server-side for each request**, not through a client-visible literal or model training.
5. **Ollama adapter** — select the exact tag from a config entry. Use consistent model options where possible, common context length (`4096` in the first comparable baseline), stable generation settings, and the same prompt assembly. Record model-specific overrides such as thinking-mode controls (disable thinking where supported); do not silently vary instructions to make one model look better.
6. **Output control** — extract answer text, enforce that application-controlled navigation and hyperlinks use only approved destinations, avoid exposing raw hidden instructions, and handle empty/error/timeout cases gracefully. Approximate the three-short-line target at the text level; **later check actual visual cloud fit** because pixel wrapping depends on the UI/font/viewport.
7. **Evaluation recorder** — save the exact prompts and responses plus time/tokens/options/config identifiers, followed by separate reproducible human annotations. Don't manufacture scores or estimate unsupported metrics.

Do not conflate a language model's requested refusal with guaranteed prompt-injection or jailbreak resistance. Security needs **server-side input/output controls**, safe URL mapping, isolation of nonpublic data, rate limiting and operational defenses.

## 5. Initial local test protocol

All six models get the **same ~20–30 version-controlled test prompts**, fact block, system policy, context limit, and comparable decoding settings. Model-specific incompatibilities or thinking switches must be logged. Cover:

- **Identity and education:** easy directly answerable facts, abbreviations, ambiguity, and paraphrases.
- **QMIND and lab:** approved high-level info; checks that the model does not invent internal research findings or publications.
- **Five projects:** must name all five; **Satira / DeepLense detail questions should receive honest “not available” answers** rather than fabricated project blurbs.
- **Navigation:** “take me to QMIND/Satira,” alternate spellings, unknown destination, and malicious attempts to make the model invent a link.
- **Off-topic:** general knowledge, personal requests not about Rastin, and requests for unapproved contact details.
- **Adversarial:** role override, hidden-prompt extraction, prompt injection, unsafe link/navigation instructions, multi-turn steering if conversation history is added.
- **Brevity and usability:** answers to simple/long/compound questions assessed against about three short lines and the long-answer continuation behavior.

Start with one pass of each case/model, then repeat a subset to estimate variability. Keep **raw outputs** separate from **human-assigned** grades. Capture structured fields such as `case_id`, `case_type`, `model_tag`, `run_id`, `prompt`, `raw_output`, `elapsed_ms`, `time_to_first_token_ms` if streamed, `prompt_tokens`, `output_tokens`, `tokens_per_second` when measurable, `word_count`, `config_hash`, `facts_version`, and `error`. Manually score factuality, scope, unknown-handling, refusal behavior, navigation correctness, and brevity. Do not award imaginary passing grades. A 30–50 prompt adversarial **public writeup** is a later design goal, not a completed milestone.

## 6. Suggested repo and boundaries (not created yet)

**Suggested GitHub repository:** `rastinaghighi-slm-eval` (**private initially**). GitHub identity, local clone path, and repo URL are **not known**; do not invent them. Repository may be initialized locally first and linked to a GitHub repo manually or using `gh` after verifying its installation/authentication.

```text
rastinaghighi-slm-eval/
├── README.md                       # scope, setup, exact commands, reproducibility
├── .gitignore                      # credentials, caches, raw/private data, local PDF by default
├── .env.example                    # safe placeholders only
├── docs/
│   ├── slm-development-handoff.md  # this handoff (copy from supplied file)
│   ├── evaluation-protocol.md      # finalized questions, settings and review rubric
│   └── design-reference.md         # summary/reference; authoritative plan kept separately
├── knowledge/
│   └── facts.md                    # owner-reviewed, versioned fact block + metadata
├── prompts/
│   └── system.md                   # one shared v1 instruction for all six candidates
├── configs/
│   ├── models.yaml                 # exact tags, identical baseline, documented exceptions
│   └── routes.yaml                 # semantic aliases, pending actual paths
├── eval/
│   ├── cases.jsonl                 # same test cases for every model
│   └── rubric.md                   # human judgment criteria
├── src/
│   ├── ...                         # single Ollama adapter, fact loader, router, runner
│   └── ...                         # output validator and result recorder
├── tests/                           # unit tests for extraction, routing, result format
└── results/                         # local machine-produced runs; not public by default
```

**Repository hygiene:** Do not push passwords, API tokens, local Windows paths, confidential documents, private evaluation data, chat history, unapproved lab descriptions, or a PDF whose distribution has not been reviewed. Ignore `.env`, virtual environments, test artifacts unless intentionally curated, and local raw results. No production site code is needed for the initial comparison.

## 7. Exact next-step sequence for the fresh chat

**Phase 0 — repository setup (first):**

1. Confirm local prerequisites with **Windows PowerShell** commands: `git --version`, `python --version`, `ollama --version`, optionally `gh --version`, and Claude Code or Codex availability. The six models' load tests are already complete; **do not pull them again**.
2. Choose a local directory for `rastinaghighi-slm-eval`; create a **private** GitHub repository using the web UI or GitHub CLI, or initialize locally and attach remote later. Provide exact commands **one step at a time** for the actual environment; do not assume GitHub account handles or URLs.
3. Lay out folders and safe `.gitignore`, import the two attached Markdown files into `knowledge/` and `docs/`, and create a brief `README.md`. Keep the design PDF separate/private and link to it by descriptive title/date.
4. Make the initial commit, connect the chosen remote, push it, and verify the project structure and `git status`. **Do not say the push succeeded unless tool output confirms it.**

**Phase 1 — one-codebase baseline (next):**

5. Write a **single high-quality Claude Code (or Codex) implementation prompt** anchored to this handoff and the November 4 design plan. Ask the coding agent to create a configurable Ollama harness, not six bots; request tests and clear launch commands.
6. Implement and unit-test the fact-block loader, single policy prompt, deterministic semantic router, config-driven local adapter, and results recorder. The handler must refuse unsupported biography/contact claims without fabricating them.
7. Curate and freeze ~20–30 identical test cases and scoring rubric, then run a small smoke test across all six configured tags. **Initial load tests do not substitute for these evaluations.**

**Phase 2 — evaluation and decision (after):**

8. Run baseline evaluations, retain raw results, verify timing/token metadata, then score outputs manually using the same rubric. Repeat selected cases.
9. Compare quality, brevity, unknown handling, navigation, safety, and actual latency. **Only then** select a candidate or decide further tests are needed.
10. Investigate each finalist's license, cloud model availability, runtime/API behavior, cost, deployment security, and feasibility for Cloudflare Workers AI or another host. Do not claim Workers compatibility without checking.
11. Later integrate with the website and visual mascot, following the authoritative design plan. **Do not do this during the initial repo setup.**

## 8. Acceptance criteria for the initial Claude Code / Codex work

- A single documented source tree with **six exact tags configured**, no model-specific fork and no trained weights.
- All six receive the same loaded approved fact block, base policy, prompts, and comparable settings; any exceptions are **explicitly recorded**.
- Routing can output **only allowed semantic route IDs**, and unknown/unconfigured paths are not fabricated.
- Automated tests cover fact extraction, missing facts, request routing and output-format invariants.
- Evaluation produces auditable raw responses and a CSV with measured fields. Empty/unavailable timing fields are recorded as unavailable, **not filled with guesses**.
- Nothing is claimed to be deployed, secure against all attacks, or proven accurate before testing.

## 9. Decisions intentionally deferred

Actual Satira and DeepLense descriptions; lab-cleared project details; contact/profile URLs; final site slugs; which color maps to which project; mascot name/look/persona; production hosting; model winner; final UI fit threshold; legal/license conclusions; possible later persona fine-tuning. The original design plan is authoritative on these open decisions.

**Fresh-chat input package:** Attach (1) the authoritative **Nov 4, 2026 16-page design PDF**, (2) **`facts.md`**, and (3) **`slm-development-handoff-2026-11-05.md`**. Paste the companion prompt supplied in the current chat. The old `facts-review.md` and old evaluation handoff are superseded for day-to-day use by these two consolidated files; preserve originals privately for audit if desired.
