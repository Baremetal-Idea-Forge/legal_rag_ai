# Changelog

Version-wise record of notable changes.

## 0.2.0

**ACTION_PLAN v2 retrieval deltas: L6 conditional lexical routing, Gate-1
abstention floor, abstain-first verification (regenerate loop removed).**

Implements the v2 design's changed decisions in the existing
FastAPI/Typesense stack. Everything is CPU-only rule-based code — no new
models or Apple-specific dependencies; runs unchanged on AMD/Linux.

### Added
- **Exact-term detector** (`services/exact_terms.py`) — rule-based L6 routing
  signal: quoted phrases, §/Section/Article/Rule/Order numbers (arabic or
  uppercase Roman), and mid-query Defined Term capitalization runs.
- **L6 conditional lexical routing** (`EXACT_TERM_ROUTING_ENABLED`, default
  off): with the flag on, the RAG read path is dense-only (`vector`) and the
  lexical (`hybrid`) channel fires only on detector-flagged queries — Reuter
  App. B: BM25 binds documents but costs span precision. Off = pre-v2
  always-hybrid behavior. The chosen mode is reported as
  `ChatResponse.retrieval_mode` and audited.
- **Gate 1 — retrieval-score floor** (`GATE1_MIN_SCORE`, default 0.0 = off):
  when the best dense chunk score (1 − vector_distance) is below the floor,
  the service abstains *before any generation call*, attaching the nearest
  documents (`scoped_documents`) for review. Keyword-only hit sets carry no
  dense plane and skip the gate. One rewrite-and-re-retrieve attempt runs
  first (`GATE1_REWRITE_RETRY`, L11) — now the only place query rewriting
  exists.
- **Doc-level eval metrics** (`tests/eval/`): DocHit@k and DRM@k (document-
  retrieval mismatch, Reuter's headline failure metric) via an optional
  `expected_doc` field in the golden set; `run_eval.py --routed` benchmarks
  per-query L6 routing before the flag is enabled.

### Changed
- **Verification is abstain-first (Gate 2).** On a failed min-of-triad gate
  the service abstains immediately with best-effort sources instead of
  rewriting and regenerating (v2 non-goal R8: regenerate loops doubled
  latency without fixing groundedness). `VERIFY_MAX_RETRIES` is removed from
  config; `VERIFY_ENABLED`/`VERIFY_MIN_TRIAD_SCORE` are unchanged.
- Default behavior is unchanged with all new flags at their defaults (always-
  hybrid retrieval, no Gate-1 floor), matching the 0.1.0 flag discipline.

### Tests
- `pytest`: 618 passed, 1 skipped (+31 net new across exact-terms/routing/
  Gate-1/abstain-first/doc-metrics; corpus-dependent count).

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
