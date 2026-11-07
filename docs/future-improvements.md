# Mascot-Lab — future improvements

**Status:** November 6, 2026, written after the first real evaluation and the provisional v1 selection (`docs/model-selection-v1.md`). **Nothing here is implemented.** The November 4, 2026 design plan stays authoritative, and every item that touches one of its open decisions remains the owner's.

Each item is in one of three groups:

- **A. Blocks a safe public launch.** It must be done before any visitor can reach the chatbot.
- **B. Needed for the designed website experience.** It is not a safety blocker, but the design plan's interaction depends on it.
- **C. Optional.** It improves confidence, reproducibility, or quality; the launch does not depend on it.

## A. Blocks a safe public launch

### A1. Production hosting and integration

- **Hosting is unresolved.** Phi-4-mini is not in Cloudflare's Workers AI model catalog (the page dated August 12, 2026, read on November 6). The design plan allows "Cloudflare Workers AI or self-hosted" within free or cheap limits (p. 8). Choose a self-hosted inference server that only the backend can reach, or a third-party host, after checking cost, data retention, and availability. None of those has been verified.
- **Build the production request path.** It is an HTTP endpoint that runs the same pipeline as `ChatEngine` and returns only the visitor fields (`answer_text`, `cloud_preview`, `continuation`, `actions`), never the record or `error.message`. A Cloudflare Worker cannot run the Python code, so either port the logic (`routes.yaml`, `chat.yaml`, phrase normalization, output validation) or have the Worker call a Python service.
- **Re-run the evaluation against the production artifact and runtime.** A hosted build or a different quantization can behave differently from the local Q4_K_M GGUF.
- **Never expose Ollama or any model server to the internet.** The adapter is loopback-only by design, so a remote model server needs a new, authenticated transport. That is a deliberate change, not a configuration setting.

### A2. Server-side input protection, rate limiting, and operational security

- **The design plan's safeguards** (p. 8): answers only about Rastin; an off-topic filter; prompt-injection defenses; per-visitor rate limits; and facts kept on the server.
- **Also needed:**
  - limits on message size, request time, and concurrency, and a cost ceiling;
  - bot and abuse protection and origin restrictions;
  - secrets for any link between the Worker and the model host;
  - logging that keeps no unneeded visitor data, with a retention rule;
  - monitoring and a graceful fallback;
  - telling visitors they are talking to an AI, as the model card asks.
- **Why the model alone is not enough:** two of the six candidates wrote code for an off-topic request. The selected model passed the 4 scope and adversarial cases here, which is far too little evidence for public exposure. Output validation removes links and verbatim instruction copies, but it does not catch paraphrased leaks.

### A3. Expanding the adversarial test suite

- The first evaluation had 2 off-topic and 2 injection cases.
- **Before launch, add:**
  - role-play and persona attacks;
  - instructions embedded inside questions;
  - encoded or obfuscated text;
  - other languages, since the model card notes weaker non-English behavior and language-specific jailbreaks;
  - inputs near the 1,000-character limit;
  - requests for links and contact details;
  - repeated attempts, and multi-turn steering if conversation history is added.
- Run the suite against the production stack, not only locally.
- The public "how this works" page with 30 to 50 adversarial prompts and their results is a design-plan goal (p. 8) that builds on this suite.

### A4. Model-license confirmation

- **Established:** Phi-4-mini is MIT-licensed, according to the license text bundled with the Ollama tag and Microsoft's model card (`docs/model-selection-v1.md`, section 11).
- **Remaining:** the owner's sign-off, because legal conclusions are deferred to the owner (handoff, section 9). Also confirm the terms of whatever build production uses, if it is not this artifact.
- **Backup:** Granite 4.2's bundled text is Apache 2.0, not yet checked against the upstream model card.

### A5. Public-disclosure review of the facts, and launch-time fact updates

- Review the knowledge block for public disclosure before deployment (handoff, section 2; the disclosure note in `knowledge/facts.md`). Check the repository again before changing its visibility.
- At launch, update the website-status facts (W01, C01), which currently say the site is not built, following the update protocol in `facts.md`.

### A6. Owner review of the evaluation and the choice

- Grade or confirm the grades: blind, from `review.csv`, or by checking the provisional AI grades in `review-claude.csv`.
- Settle the rubric's open questions and approve the rubric.
- Approve or change the provisional selection, in particular how source tags are classified, which decides the order of the winner and the backup.

## B. Needed for the designed website experience

### B1. Streaming and time-to-first-token measurement

- The mascot plays its talking animation "while an answer streams in" (design plan, pp. 6–7). Requests are not streamed now, so `time_to_first_token_ms` is null in every record.
- Output validation currently checks a complete answer. A streaming design must not show unvalidated text, so how to validate incrementally is a design decision of its own.
- Measure time to first token once streaming exists.

### B2. Response length and actual visual fit

- Open-ended questions run long: "Tell me about Rastin." took 15 cloud lines in the smoke test.
- Two winner answers used Markdown lists, which display as run-together text, and two showed internal source tags.
- **Possible output-validation changes:** strip source tags and entry codes, and turn list items into a sentence. Re-test after either change.
- **Prompt changes** apply to all six models, because the instruction is shared, and would need a full re-evaluation.
- The 3-line × 45-character cloud approximation is provisional. Check the real fit visually once the cloud exists.

### B3. Navigation destinations

- Map the semantic route IDs to real paths once the site's slugs exist (all "Pending" in the `facts.md` registry).
- Decide whether model answers may carry a "take me there" suggestion (README, provisional values).

### B4. Cold starts on the chosen host

- Locally, Phi-4-mini's first request after a load took 75 to 87 seconds, and Granite 4.0 H Micro could not finish a cold load within the 120-second timeout.
- Keep the production model loaded, or design the first-request behavior and its fallback.

## C. Optional improvements

### C1. Standardizing model-specific generation defaults

- Settings outside the baseline still come from each Modelfile: Qwen3.5's `presence_penalty 1.5`, which changes greedy decoding, and `top_k 20`; Gemma's `repeat_penalty` and `penalize_newline`.
- Pinning them in the shared baseline would make comparisons cleaner. It changes the evaluation conditions, so it needs the owner's approval and a re-run.

### C2. Recording Ollama version, model digests, and CPU/GPU placement

- The adapter has no request for the server version, the model digests, or the running models' placement, so this run recorded them by hand: Ollama 0.40.14, the `ollama list` IDs, and `ollama ps` placement.
- Recording them automatically needs an adapter change.

### C3. Repeating evaluations to measure variability

- First, repeat the two finalists (`--model phi4-mini:3.8b-q4_K_M --model granite4.2:3b-q4_K_M --repeat 3`) to settle the close call.
- Then repeat a subset across all models. Even with `temperature` 0, outputs can vary between runs.

### C4. More robust factual grounding and unknown-question handling

The winner's remaining issues:

- a recurring blur ("specializing in Artificial Intelligence and Mathematics" for the AI sub-plan and Mathematics minor);
- false-premise questions answered with "not available" instead of a correction;
- some refusals with no route back to Rastin.

Possible fixes, each followed by a re-evaluation:

- clearer wording of the facts, with owner approval;
- instruction refinements, which are shared and would need a full re-run;
- fixed application answers for the most common questions.

### C5. Model-specific thinking compatibility

- All six accepted `think: false`, and the two models that report thinking (Qwen3.5 and Granite 4.2) returned no reasoning. Phi-4-mini reports no thinking capability, so nothing is needed for the winner.
- The `overrides` schema cannot set `think` per model. That matters only if a thinking model is chosen or a new Ollama version behaves differently.

### C6. Updating approved facts as projects evolve

- Add owner-approved wording for Satira and DeepLense, contact links, and the résumé URL when they exist, following the update protocol in `knowledge/facts.md`.
- Re-run the factual and safety cases after any material change, because facts are injected per request, not trained in.

### C7. Separating model loading from timed requests

- An optional warm-up before the timed requests, or a longer timeout for a model's first request, would keep cold loads from turning into unusable records.
- It changes how runs behave, so it needs the owner's approval.

### C8. Conversation history

- The engine is single-turn, like the cloud. The mascot's page can show the whole conversation (design plan, p. 7). Multi-turn steering can be tested only once history exists.

### C9. Optional later persona fine-tuning

- The design plan puts persona and fine-tuning after v1 (p. 8): voice only, with facts still injected on the server.
- Phi-4-mini's open MIT weights allow it. Workers AI does not host Phi-4-mini, so a fine-tune would also need another host.
- Nothing is trained in v1.
