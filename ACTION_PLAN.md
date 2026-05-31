# Legal RAG System — Action Plan

> Living implementation roadmap. Status legend: ☐ not started · ◐ in progress · ☑ done.

---

## 0. Locked Decisions

| Concern | Decision | Rationale |
|---|---|---|
| **Embedding model** | `yuriyvnv/legal-bge-m3` (1024-dim) | Legal-domain tuned, multilingual; existing code + schema + tests already built around it |
| **LLM** | Gemma 3 4B via **Ollama** (`localhost:11434`, OpenAI-compatible API) | Easiest local self-host; swappable behind a client interface |
| **Vector/Search** | Typesense — hybrid (BM25 keyword + vector ANN) | Single engine for both search modes; already integrated |
| **MCP topology** | **Backend orchestrates.** MCP = read-side tool provider wrapping embeddings + Typesense. FastAPI owns the RAG loop. | Simpler, testable; agentic tool-calling can be layered later |
| **Retrieval primitives** | Shared backend module; `mcp-server` imports it. Ingestion (write) stays direct in backend; search (read) routed through MCP from Step 3 | Avoids duplicate Typesense/embedding config (DRY) |
| **Architecture** | Clean layering: routers → services → repositories/clients | Separation of concerns, swappability, testability |

---

## 1. Target Structure

```
legal_rag_ai/
├── backend/
│   ├── core/          config · logging · exceptions · dependencies (DI)
│   ├── routers/       thin HTTP handlers: ingest · search · chat · health
│   ├── services/      ingestion · search · rag (business logic)
│   ├── repositories/  typesense_repo · pdf_storage_repo (data access)
│   ├── clients/       ollama_client · mcp_client (external systems)
│   ├── models/        pydantic DTOs (request/response) + embeddings wrapper
│   ├── helpers/       existing low-level utils (KEPT, wrapped not rewritten)
│   └── main.py        app factory · lifespan · router registration
├── mcp-server/        MCP tools: search_legal_chunks · get_document
├── frontend/          React SPA (TS + Tailwind): upload · search · chat
├── tests/             pytest (existing 913 tests stay green + new per layer)
└── README.md          run instructions · env · architecture summary
```

**Principle:** `helpers/` (pdf_helper, typesense_helper, embeddings_model — currently 100% tested) become the foundation **under** the repository/service layers. Wrap, do not rewrite.

---

## 2. Current State (Baseline)

| Component | Status | Notes |
|---|---|---|
| `backend/core/config.py` | ☑ | Pydantic settings |
| `backend/helpers/pdf_helper.py` | ☑ | Storage, extract, chunk, doc-build — 100% cov |
| `backend/helpers/typesense_helper.py` | ☑ | Schema, collection mgmt, CRUD, hybrid search — 100% cov |
| `backend/models/embeddings_model.py` | ☑ | legal-bge-m3 wrapper — 100% cov |
| `backend/main.py` | ◐ | App shell + `/health`; routers are commented placeholders |
| Clean layering (routers/services/repos) | ☐ | Flat `helpers/` today |
| Ingestion + search endpoints | ☐ | Not wired |
| LLM / generation layer | ☐ | Does not exist |
| `mcp-server/` | ☐ | Does not exist |
| `frontend/` | ☐ | Does not exist |
| `README.md` | ◐ | One line |
| Test suite | ☑ | 913 passing, 100% cov on the 4 core files |

---

## 3. Cross-Cutting Standards (apply to every step)

- **Error handling:** typed domain exceptions in `core/exceptions.py` (`DocumentNotFoundError`, `IngestionError`, `SearchError`, `LLMError`, `UpstreamUnavailableError`); mapped to HTTP via a FastAPI exception handler. No bare `except`. No silent failures.
- **Logging:** structured logging via `core/logging.py`; one logger per module; INFO for lifecycle, WARNING for degraded, ERROR with stack for failures. Correlation/request ID middleware.
- **Config:** all tunables via `core/config.py` (pydantic-settings) + `.env`; no hardcoded hosts/keys/model names.
- **DI:** dependencies wired in `core/dependencies.py` and injected via FastAPI `Depends` — no module-level singletons leaking into handlers.
- **Validation:** pydantic request/response models on every endpoint; explicit limits (file size, query length, top_k bounds).
- **Testing gate:** each step ships with tests; `pytest` stays green before moving on.
- **Security:** tighten CORS off `*`, enforce upload size/type limits, sanitize filenames (already done), prepare an auth dependency seam.

---

## 4. Build Sequence

### Step 1 — Backend Skeleton & Clean Architecture  ☑
**Goal:** Refactor flat helpers into layered architecture; wire real ingestion + search endpoints.

**Tasks**
- [x] `core/logging.py` — structured logging setup + request-ID middleware.
- [x] `core/exceptions.py` — domain exception hierarchy + FastAPI handlers.
- [x] `core/dependencies.py` — DI providers (settings, repos, services).
- [x] `core/config.py` — extended in Step 0 (Ollama, MCP, embedding, upload limits, top_k).
- [x] `repositories/typesense_repository.py` — wraps `helpers/typesense_helper`; `index_chunks`, `search`, `get_document`, `ensure_collection`, `is_ready`.
- [x] `repositories/pdf_storage_repository.py` — wraps `helpers/pdf_helper` save/extract/chunk/build.
- [x] `models/schemas.py` — `IngestResponse`, `SearchRequest/Response`, `ChunkHit`, `ChatRequest/Response`, `Citation`, health DTOs.
- [x] `services/embedding_service.py` — wraps `models/embeddings_model`; lazy model load.
- [x] `services/ingestion_service.py` — store → extract → chunk → embed → index. Idempotency via SHA-256 (= pdf_id).
- [x] `services/search_service.py` — embed query → hybrid Typesense search → ranked `ChunkHit`s.
- [x] `routers/ingest.py` — `POST /ingest` (multipart), `POST /ingest/bulk` (from `data/`).
- [x] `routers/search.py` — `GET/POST /search`.
- [x] `routers/health.py` — `/health` (liveness) + `/health/ready` (Typesense readiness).
- [x] `main.py` — registers routers, exception handlers, request-ID + CORS middleware; lifespan ensures collection.

**Verify** ◐
- [x] `pytest` green — **971 passed** (913 existing + 58 new layer tests, Typesense + embeddings mocked).
- [ ] Live smoke (needs Typesense running): `POST /ingest` a real `data/` PDF → chunk count + pdf_id; `POST /search` → ranked hits. *(deferred to live env)*

---

### Step 2 — RAG Generation (Ollama)  ☑
**Goal:** Turn retrieval into grounded, cited answers.

**Tasks**
- [x] `clients/ollama_client.py` — async client (`/api/chat`); timeouts, retries w/ exp backoff, streaming; all failures → `LLMError` (503).
- [x] `services/rag_service.py` — retrieve (threadpool, non-blocking) → select context → grounded prompt → Ollama → cited answer; no-context path skips the LLM.
- [x] `services/prompts.py` — strict legal system prompt, `[pdf_name p.N]` context headers, `select_hits_within` budget control, `NO_CONTEXT_ANSWER`.
- [x] `models/schemas.py` — `ChatResponse` (answer + citations[] + chunks_used[]) defined in Step 1, used here.
- [x] `routers/chat.py` — `POST /chat` (sync) + `POST /chat/stream` (SSE: token events → citations → done).
- [x] DI: `get_ollama_client`, `get_rag_service`; registered in `main.py`.
- [ ] Pull the model: `ollama pull gemma3:4b` (documented in README at Step 5).

**Verify** ◐
- [x] `pytest` green — **1006 passed** (+35 Step-2 tests; Ollama mocked via httpx MockTransport, search faked).
- [x] Out-of-corpus question → `NO_CONTEXT_ANSWER`, LLM never called (no-hallucination path unit-tested).
- [x] Ollama-down / 5xx / malformed → `LLMError` (503), with retries + backoff (unit-tested).
- [ ] Live smoke with real Ollama + Typesense → grounded answer citing specific PDFs/pages. *(deferred to live env)*

---

### Step 3 — MCP Server  ☑
**Goal:** Expose retrieval as MCP tools; route backend RAG retrieval through MCP.

**Tasks**
- [x] `mcp-server/server.py` — FastMCP (streamable-HTTP) server importing the backend's `SearchService` + Typesense repo (shared primitives).
- [x] Tool `search_legal_chunks(query, top_k, mode, filter_by)` → `{"hits": [...]}` hybrid results.
- [x] Tool `get_document(chunk_id)` → `{"document": {...}}`.
- [x] `clients/mcp_client.py` — streamable-HTTP client; result extraction (structured/text); failures → `UpstreamUnavailableError` (503).
- [x] `services/rag_service.py` — `RETRIEVAL_VIA_MCP` flag swaps direct search for the MCP tool call (falls back to direct search if no client).
- [x] DI: `get_mcp_client`; `mcp-server/README.md`.
- [x] DRY confirmed: mcp-server imports the one backend retrieval stack + one Typesense/embedding config.

**Verify** ◐
- [x] `pytest` green — **1025 passed** (+24 Step-3 tests; tools dispatched in-process, client tested via session seam).
- [x] MCP-down / tool-error → `UpstreamUnavailableError` (503), logged (unit-tested).
- [x] Flag routing unit-tested: `true`→MCP, `false`→direct, `true`+no-client→direct fallback.
- [ ] Live: tool call returns identical hits to direct search; `/chat` end-to-end with `RETRIEVAL_VIA_MCP=true`. *(deferred to live env)*

---

### Step 4 — Frontend (React SPA)  ☑
**Goal:** Usable UI for upload, search, chat.

**Tasks**
- [x] Vite + React 18 + TS (strict) + Tailwind v3 scaffold in `frontend/` (configs, env, README).
- [x] Typed API client (`api/types.ts` mirrors backend DTOs, `api/client.ts`, `lib/sse.ts`) for `/health`, `/ingest`, `/search`, `/chat`, `/chat/stream`.
- [x] Upload view — drag/drop + click, **XHR upload progress**, ingest result (pages/chunks/sha).
- [x] Search view — query box, mode (hybrid/keyword/vector) + top-k, ranked `ChunkCard`s with page refs + **term highlight**.
- [x] Chat view — textarea (Enter to send), **SSE-streamed answer** with live cursor, Stop/abort, **clickable citation chips** → source `file_url`.
- [x] Error/loading/empty states; responsive layout; header health indicator (`/health`).

**Verify** ◐
- [x] JSON/config sanity validated; strict-TS event typing + import interop fixed by inspection (no Node in this env to run `tsc`/`vite build`).
- [ ] Live (needs `npm install` + running backend): upload a PDF → indexed; ask a question → streamed cited answer; citations resolve. *(deferred to a Node env)*

> Note: citation links target `{API}{file_url}`; they resolve once the backend mounts static PDF serving (Step 5 hardening).

---

### Step 5 — README & Hardening  ☑
**Goal:** Reproducible setup + security pass.

**Tasks**
- [x] Root `README.md` — architecture diagram, prerequisites, step-by-step run (Typesense/Ollama/backend/MCP/frontend), API table, env reference, sample legal queries, security notes.
- [x] `.env.example` for backend (root) + frontend; `docker-compose.yml` for Typesense.
- [x] Security: CORS explicit origins (Step 0/1), upload size+MIME enforcement (Step 1), query/top_k bounds (Step 1), **`API_AUTH_TOKEN` auth seam** on the write path, **`RATE_LIMIT_*` rate-limit seam**, secrets via env only.
- [x] Ops: structured logs + request IDs (Step 1), **deep `/health/ready`** (Typesense + Ollama `ping`), graceful lifespan, **static `/pdfs` serving** so citations resolve.
- [x] `core/security.py` (`require_api_key`, `RateLimiter`); `AuthError` (401) + `RateLimitError` (429).

**Verify** ◐
- [x] Final `pytest` — **1043 passed**; new Step-5 modules at 99% (security/ollama-ping/readiness/auth all tested).
- [x] Security checklist reviewed; no secrets committed; corrupted-PDF + adversarial-input (bad MIME, oversize, empty, out-of-range, bad auth) paths handled & tested.
- [ ] Fresh clone → README → working system (live infra). *(deferred to a deployed env)*

---

## Status: all five steps implemented ☑ · backend/MCP fully unit-tested (**1043 tests, ~99%**) · live end-to-end smokes deferred to an environment with running Typesense + Ollama + Node.

---

## 5. Dependencies & Prerequisites

| Need | For | Install/Run |
|---|---|---|
| Typesense Server | Steps 1+ | Docker: `typesense/typesense` on `:8108` |
| Ollama + `gemma3:4b` | Steps 2+ | `ollama pull gemma3:4b`; serves `:11434` |
| Python deps | Backend/MCP | `uv sync` (fastapi, uvicorn, pymupdf, sentence-transformers, typesense, mcp, httpx) |
| Node + pnpm/npm | Frontend | Step 4 |

---

## 6. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Gemma 3 4B context window limits packed legal chunks | Truncated context, weaker answers | Cap top_k + per-chunk chars; rerank; summarize-then-answer fallback |
| Corrupted PDFs in `data/` (2 known: SALE_OF_GOODS, LIMITATION) | Ingestion failures | Already raise `ValueError`; ingestion skips + reports, never crashes batch |
| Embedding cost/latency on large corpus | Slow ingestion | Batch encode; cache; background ingestion task |
| Hallucination beyond retrieved context | Wrong legal info (high stakes) | Strict grounded prompt; "not found" path; always show citations; never answer uncited |
| MCP/Ollama/Typesense down | Endpoint failures | Typed upstream errors → clean 5xx; `/health` deep checks; retries w/ backoff |
| Embedding ↔ schema dim drift (1024) | Index/query mismatch | Single source of truth in config; startup assertion model dim == schema num_dim |

---

## 7. Definition of Done

- All 5 steps verified; `pytest` green with meaningful coverage per new layer.
- End-to-end: upload → ingest → search → cited chat answer, via UI.
- No hardcoded secrets; CORS locked; input limits enforced.
- README reproduces the system from a clean clone.
- Every answer is grounded and cited; out-of-corpus questions decline gracefully.

---

## 8. Next Action

**Start Step 1 — Backend Skeleton & Clean Architecture.** Highest leverage; everything builds on the layered foundation and the wired ingest/search endpoints.
