# Legal RAG AI

A secure, locally-hosted **Retrieval-Augmented Generation** system for legal
documents. Upload PDFs, run hybrid (keyword + vector) search, and ask questions
that are answered **only** from the indexed corpus — every answer cited, nothing
fabricated.

```
┌──────────────┐    HTTP    ┌───────────────────────────┐
│  React SPA   │ ─────────► │        FastAPI backend     │
│ (upload/     │            │  routers → services → repos│
│  search/chat)│ ◄───SSE──  │                            │
└──────────────┘            │   ├─ ingest  (PDF→chunk→   │
                            │   │   embed→index)         │
                            │   ├─ search  (hybrid)      │
                            │   └─ chat    (RAG + cite)  │
                            └───────┬───────────┬────────┘
                                    │           │
                     RETRIEVAL_VIA_MCP?     ┌────▼─────┐
                                    │       │  Ollama  │  Gemma 3 4B
                              ┌─────▼────┐  └──────────┘
                              │ MCP srv  │
                              └─────┬────┘
                                    ▼
                            ┌──────────────┐   ┌──────────────────────┐
                            │  Typesense   │   │ legal-bge-m3 (1024-d) │
                            │ hybrid search│   │ SentenceTransformers  │
                            └──────────────┘   └──────────────────────┘
```

| Layer | Tech |
|---|---|
| Frontend | React 18 · TypeScript · Tailwind · Vite |
| Backend | Python 3.13 · FastAPI · clean architecture (routers → services → repositories/clients) |
| Search | Typesense (BM25 keyword + vector ANN, hybrid) |
| Embeddings | `yuriyvnv/legal-bge-m3` (1024-dim) via SentenceTransformers |
| LLM | Gemma 3 4B via Ollama (OpenAI-compatible local API) |
| Integration | MCP server exposing retrieval as tools (`Model ⟷ MCP ⟷ Embedding/Typesense`) |

---

## Project layout

```
legal_rag_ai/
├── backend/        FastAPI app (core, routers, services, repositories, clients, models)
├── mcp-server/     MCP retrieval server (shares the backend retrieval stack)
├── frontend/       React SPA
├── data/           Source legal PDFs
├── tests/          pytest suite (1000+ tests)
├── docker-compose.yml   Typesense for local dev
├── ACTION_PLAN.md  Implementation roadmap (status per step)
└── .env.example    Backend configuration reference
```

---

## Prerequisites

- **Python 3.13** + [uv](https://docs.astral.sh/uv/)
- **Docker** (for Typesense) — or a native Typesense install
- **Ollama** — https://ollama.com
- **Node 18+** (frontend)

---

## Quick start

### 1. Infrastructure

```bash
docker compose up -d                 # Typesense on :8108
ollama pull gemma3:4b                # LLM (serves :11434)
```

### 2. Backend

```bash
cp .env.example .env                 # adjust if needed
uv sync                              # install deps (first run downloads the embedding model on first query)
uv run python main.py                # http://localhost:8000  (docs at /docs)
```

Ingest the bundled corpus, then try it:

```bash
curl -X POST http://localhost:8000/ingest/bulk        # index everything in data/
curl -X POST http://localhost:8000/search  -H 'content-type: application/json' \
     -d '{"query":"remedies for breach of contract","top_k":5}'
curl -X POST http://localhost:8000/chat    -H 'content-type: application/json' \
     -d '{"query":"What are the remedies for breach of contract?","top_k":5}'
```

### 3. MCP server (optional)

```bash
uv run python mcp-server/server.py    # http://localhost:9000/mcp
# then set RETRIEVAL_VIA_MCP=true in .env and restart the backend
```

### 4. Frontend

```bash
cd frontend
cp .env.example .env
npm install
npm run dev                          # http://localhost:5173
```

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | Liveness |
| `GET`  | `/health/ready` | Readiness (Typesense + Ollama reachable) |
| `POST` | `/ingest` | Upload one PDF (multipart `file`) |
| `POST` | `/ingest/bulk` | Ingest every PDF in `data/` |
| `POST` / `GET` | `/search` | Hybrid / keyword / vector search |
| `POST` | `/chat` | Grounded, cited answer |
| `POST` | `/chat/stream` | Same, streamed via SSE (`token` → `citations` → `done`) |
| `GET`  | `/pdfs/{pdf_id}/{file}` | Stored PDF (citation links) |

Sample legal queries: *“essential elements of a valid contract”*, *“difference
between culpable homicide and murder”*, *“grounds for divorce under Hindu law”*,
*“what is a negotiable instrument”*.

---

## Configuration

All settings live in [`backend/core/config.py`](backend/core/config.py) and are
overridable via `.env` — see [`.env.example`](.env.example). Highlights:

| Var | Default | Notes |
|---|---|---|
| `EMBEDDING_MODEL_NAME` | `yuriyvnv/legal-bge-m3` | 1024-dim; must match schema |
| `OLLAMA_MODEL` | `gemma3:4b` | via `OLLAMA_BASE_URL` |
| `RETRIEVAL_VIA_MCP` | `false` | route retrieval through the MCP server |
| `RAG_TOP_K` / `RAG_MAX_CONTEXT_CHARS` | `5` / `12000` | retrieval + context budget |
| `MAX_UPLOAD_BYTES` | `52428800` | 50 MB upload cap |
| `CORS_ALLOW_ORIGINS` | `localhost:5173,3000` | explicit origins (no `*`) |
| `API_AUTH_TOKEN` | `""` | set to require a key on `/ingest` |
| `RATE_LIMIT_ENABLED` | `false` | per-client request cap |

---

## Security

- **CORS** restricted to explicit origins (no wildcard).
- **Uploads** validated by MIME (`application/pdf`) and size (`MAX_UPLOAD_BYTES`); filenames sanitised.
- **Input bounds** on every endpoint (query length, `top_k` 1–50) via pydantic.
- **Auth seam** — set `API_AUTH_TOKEN` to require `X-API-Key`/`Bearer` on the write path.
- **Rate-limit seam** — `RATE_LIMIT_ENABLED` (single-process; swap store for Redis to scale).
- **Secrets** only via env; nothing committed.
- **Grounding** — the LLM answers strictly from retrieved context and declines (no hallucination) when the corpus lacks the answer; every claim is cited.

---

## Testing

```bash
uv run pytest -q                     # full suite
uv run pytest --cov=backend          # with coverage
```

External services (Typesense, Ollama, the embedding model, MCP) are mocked, so
the suite runs fully offline.

---

## Known limitations / next steps

- Two source PDFs (`SALE_OF_GOODS_ACT.pdf`, `LIMITATION_ACT.pdf`) are corrupt and
  are reported-and-skipped during ingestion (never crash the batch).
- The rate limiter is in-process; use Redis for multi-worker deployments.
- Live end-to-end smokes (real Typesense/Ollama/Node) are deferred to a deployed
  environment — see `ACTION_PLAN.md` for the per-step status.
