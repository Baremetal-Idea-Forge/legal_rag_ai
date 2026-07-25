# Changelog

Version-wise record of notable changes.

## 0.0.2

**Fix: intermittent LLM refusals ("I cannot answer" while citing sources).**

Root cause: the RAG call used a hardcoded `temperature=0.2`, so the same
retrieved context was non-deterministically sampled — sometimes into a grounded
answer, sometimes into the strict "not found" refusal. Retrieval was never at
fault (citations were present), so the fix targets the generation layer.

### Changed
- **Deterministic generation.** Added `LLM_TEMPERATURE` setting (default `0.0`)
  in `core/config.py`; `RagService` now forwards it to both `chat()` and
  `stream_chat()` (`services/rag_service.py`). Same context → same answer.
- **Prompt hardened against false refusals.** Rule 3 in `services/prompts.py`
  now answers from any relevant context and reserves the exact "could not find
  information…" reply for topics genuinely absent from the context. The
  anti-hallucination guard (answer only from context, cite every claim) is
  unchanged.
- Documented `LLM_TEMPERATURE` in `.env.example`.

### Added
- Debug instrumentation (Phase 1): DEBUG/INFO logging across
  `services/rag_service.py`, `services/search_service.py`,
  `services/prompts.py`, `clients/ollama_client.py`, `clients/gemini_client.py`
  covering retrieval hits/scores, context selection + cutoff, and LLM
  request/response sizes.
- Consistency harness (Phase 2): `tests/test_article21_consistency.py` — runs
  the same query N times and reports answer/retrieval variance (needs live
  Typesense + LLM).
- Unit tests: `RagService` forwards the configured temperature (sync + stream);
  prompt instructs the model to answer when relevant context is present.

### Tests
- `pytest`: 481 passed, 1 skipped (+3 new; corpus-dependent count).
