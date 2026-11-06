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

## Status

Repository setup in progress. Model quality, latency, robustness, hosting compatibility, and deployment security have not yet been established.

## Repository policy

This repository is private during development. The authoritative design PDF, local evaluation outputs, and unapproved private research details are excluded from version control.

Git commits and pushes are performed manually by the repository owner.
