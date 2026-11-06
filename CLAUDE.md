# Mascot-Lab — Development Instructions

## Project
Build one shared local Python evaluation system for six Ollama
models powering the planned rastinaghighi.dev mascot.

Read docs/slm-development-handoff.md and knowledge/facts.md.
The November 4, 2026 design PDF in private/ is authoritative
for website design decisions. Do not reinterpret those decisions.

## Git ownership
- Never run Git or GitHub operations.
- Never stage, commit, push, create branches, or open PRs.
- Never modify Git author/committer settings.
- Never add Co-authored-by or other AI attribution trailers.
- Never use alternate tools, scripts, APIs, or IDE actions to
  perform Git operations.
- The human owner reviews and performs all Git operations.

## Implementation boundaries
- One shared codebase, not six independent implementations.
- No model fine-tuning or personality training in v1.
- Load only the bounded MASCOT KNOWLEDGE section at runtime.
- Never invent facts, research findings, publications, or URLs.
- Navigation must use application-controlled semantic IDs.
- Do not modify the canonical facts or design documents
  without explicit owner approval.
- Never deploy or upload private materials.
- Write automated tests and document configuration exceptions.
- Never manufacture evaluation scores or performance measurements.
