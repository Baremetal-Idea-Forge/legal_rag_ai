# Changelog

Version-wise record of notable changes.

## 0.0.3

**Improve semantic search: structure-aware (legal) chunking + a retrieval eval harness.**

The chunker merged consecutive pages into 3,500-char blobs, so a single chunk
spanned many Articles/Sections. Each chunk's embedding was a blurry average of
unrelated provisions, capping retrieval precision — a query for "Article 21"
landed in a smear of the surrounding articles. This splits on legal structure so
one provision ≈ one chunk.

### Changed
- **Structure-aware chunking** (`helpers/pdf_helper.py`). `chunk_pages` now
  segments each page on Article/Section headings (`_HEADING_RE`): a heading-led
  segment starts a fresh chunk, trailing text (a provision continuing onto the
  next page) is appended, and oversized provisions sub-split with the heading
  re-stated on each continuation piece so every chunk stays citable. Heading-less
  text degrades to the previous size-based accumulation (unchanged), so
  non-legal PDFs behave as before.
- **Smaller chunk budget** to keep per-provision embeddings sharp:
  `CHUNK_MAX_CHARS` 3500 → 1200, `CHUNK_OVERLAP_CHARS` 400 → 150
  (`core/config.py`, documented in `.env.example`). **Re-ingest after upgrading.**

### Added
- **Retrieval eval harness** (`tests/eval/`): dependency-free `metrics.py`
  (Hit@k, MRR@k, Recall@k with re-chunk-robust substring relevance), a
  `golden_set.json` template, and `run_eval.py` to score the live search stack
  before/after each retrieval change — so tuning is measured, not guessed.
- Unit tests: structural chunking (`TestStructuralChunking` — Article isolation,
  cross-page continuation, heading-less fallback, oversized-provision prefixing,
  heading-regex boundaries) and the eval metrics (`tests/eval/test_metrics.py`).

### Tests
- `pytest`: 500 passed, 1 skipped (+19 new; corpus-dependent count).

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
