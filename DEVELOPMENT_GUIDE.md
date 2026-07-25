# Development Guide

Complete setup and workflow for local development.

## Prerequisites

- **Python 3.13+** (install via [pyenv](https://github.com/pyenv/pyenv) or [python.org](https://www.python.org))
- **uv** (fast Python package manager) — install via `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Node.js 18+** (for frontend, install via [nvm](https://github.com/nvm-sh/nvm) or [node.org](https://nodejs.org))
- **Docker & Docker Compose** (for Typesense, Ollama)
- **Git**

---

## 1. Clone & Initial Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/legal_rag_ai.git
cd legal_rag_ai

# Create .env from template
cp .env.example .env

# (Optional) Adjust .env for local dev
# Default settings are fine for local development with Ollama
nano .env
```

### `.env` for Local Development (Defaults are OK)

```env
# LLM: Use Ollama (local) by default
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434

# Or use Gemini (requires API key)
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=your-api-key

# Disable auth for local dev
API_AUTH_TOKEN=

# Disable rate limiting for local dev
RATE_LIMIT_ENABLED=false

# Storage (create directories as needed)
PDF_STORAGE_DIR=storage/pdfs
TYPESENSE_DATA_DIR=typesense-data
```

---

## 2. Backend Setup

### Install Python Dependencies

```bash
# Uses uv to install in a project venv (fast, deterministic)
uv sync --dev

# Or manually with uv:
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### Create Directories

```bash
mkdir -p storage/pdfs typesense-data
```

---

## 3. Start Services

### Start Typesense (Docker)

```bash
docker-compose up -d typesense
# Typesense runs on http://localhost:8108
# Check status: curl http://localhost:8108/health
```

### Start Ollama (Local LLM) — Optional

If using local Ollama:

```bash
# Install Ollama: https://ollama.ai
# Run Ollama server in background
ollama serve

# In another terminal, pull the model
ollama pull gemma3:4b
# Or run: OLLAMA_BASE_URL=http://localhost:11434 ollama pull gemma3:4b
```

**Skip if using Gemini API** (set `LLM_PROVIDER=gemini` + `GEMINI_API_KEY` in `.env`)

---

## 4. Run Backend

```bash
cd backend

# Development mode (auto-reload on file changes)
fastapi dev main.py
# or: uvicorn main:app --reload

# Production mode (single worker)
python main.py
```

**Output**:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

**Interactive Docs**: Open http://localhost:8000/docs (Swagger UI)

---

## 5. Run Frontend

In a separate terminal:

```bash
cd frontend

# Install dependencies
npm install

# Development server (hot reload)
npm run dev
# or: npx vite

# Open http://localhost:5173 in your browser
```

---

## 6. Quick Test

```bash
# Terminal 1: Backend running on :8000
# Terminal 2: Frontend running on :5173

# Terminal 3: Test API
curl -X POST http://localhost:8000/health

# Upload a test PDF
curl -X POST http://localhost:8000/ingest \
  -F "file=@path/to/test.pdf"

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"test query","top_k":5}'

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What is in this document?","top_k":5}'
```

---

## Project Structure

```
legal_rag_ai/
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── core/
│   │   ├── config.py        # Settings (env vars)
│   │   ├── dependencies.py  # Dependency injection
│   │   ├── exceptions.py    # Error definitions
│   │   ├── logging.py       # Request logging
│   │   └── security.py      # Auth, rate limiting
│   ├── routers/             # HTTP endpoints
│   │   ├── chat.py
│   │   ├── search.py
│   │   ├── ingest.py
│   │   └── health.py
│   ├── services/            # Business logic
│   │   ├── rag_service.py   # Q&A orchestration
│   │   ├── search_service.py
│   │   ├── embedding_service.py
│   │   ├── ingestion_service.py
│   │   └── prompts.py       # LLM prompts
│   ├── repositories/        # Data access
│   │   ├── typesense_repository.py
│   │   └── pdf_storage_repository.py
│   ├── clients/             # External APIs
│   │   ├── gemini_client.py
│   │   ├── ollama_client.py
│   │   └── mcp_client.py
│   ├── models/
│   │   ├── schemas.py       # Pydantic DTOs
│   │   └── embeddings_model.py
│   ├── helpers/
│   │   ├── pdf_helper.py
│   │   └── typesense_helper.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # Main component
│   │   ├── components/      # UI components
│   │   ├── api/             # HTTP client
│   │   └── lib/             # Utilities
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
│
├── tests/
│   ├── conftest.py          # Pytest fixtures
│   ├── test_api.py          # Router tests
│   ├── test_*_service.py    # Service tests
│   └── eval/                # Evaluation suite
│
├── mcp-server/              # MCP server (tools for LLMs)
├── docker-compose.yml       # Local services
├── pyproject.toml           # Project metadata + deps
├── ARCHITECTURE.md
├── DEPLOY.md
└── README.md
```

---

## Development Workflow

### 1. Make Code Changes

Edit any file in `backend/` or `frontend/`. Changes auto-reload:
- **Backend**: `fastapi dev` restarts the app (2–3 sec)
- **Frontend**: Vite updates the browser instantly (HMR)

### 2. Run Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_api.py

# Verbose output
pytest -v

# Coverage report
pytest --cov=backend --cov-report=html
# Open htmlcov/index.html to see coverage
```

### 3. Check Code Quality

```bash
# Linting (requires ruff)
pip install ruff
ruff check backend/

# Type checking (requires pyright)
pip install pyright
pyright backend/

# Format code (auto-fix)
ruff format backend/
```

### 4. Commit & Push

```bash
git add .
git commit -m "feat: add new search filter"
git push origin your-branch
```

---

## Common Tasks

### Add a New Endpoint

1. **Define schema** in `backend/models/schemas.py`:
   ```python
   class MyRequest(BaseModel):
       query: str
   ```

2. **Create router** in `backend/routers/my_feature.py`:
   ```python
   from fastapi import APIRouter, Depends
   router = APIRouter(prefix="/my-feature", tags=["MyFeature"])

   @router.post("")
   async def my_endpoint(req: MyRequest) -> MyResponse:
       ...
   ```

3. **Register in main.py**:
   ```python
   from routers.my_feature import router as my_router
   app.include_router(my_router)
   ```

4. **Add tests** in `tests/test_api.py`:
   ```python
   def test_my_endpoint(client):
       resp = client.post("/my-feature", json={"query": "test"})
       assert resp.status_code == 200
   ```

### Update Configuration

All settings live in `backend/core/config.py`. To add a new setting:

1. Add to `Settings` class:
   ```python
   MY_NEW_SETTING: str = "default_value"
   ```

2. Use in code:
   ```python
   settings = get_settings()
   value = settings.MY_NEW_SETTING
   ```

3. Override in `.env`:
   ```env
   MY_NEW_SETTING=my_value
   ```

### Debug a Test

```bash
# Run with pdb (interactive debugger)
pytest tests/test_api.py -s --pdb

# Or add breakpoint in code:
# import pdb; pdb.set_trace()

# Run with print output visible
pytest tests/test_api.py -s
```

### Add a New Dependency

```bash
# Find package on PyPI
uv pip install some-package

# Update pyproject.toml (uv does this automatically)
# Then commit uv.lock
```

### Profile Performance

```python
# In your code:
import time
start = time.time()
# ... code to profile ...
print(f"Took {time.time() - start:.3f}s")
```

Or use a profiler:

```bash
# Generate profile with cProfile
python -m cProfile -o profile.stats backend/main.py

# Visualize (requires snakeviz)
pip install snakeviz
snakeviz profile.stats
```

---

## Debugging

### Enable Debug Logging

Set `DEBUG=true` in `.env`:

```env
DEBUG=true
```

Or on command line:

```bash
DEBUG=true fastapi dev backend/main.py
```

Logs will include SQL queries, API calls, and stack traces.

### Inspect Typesense Data

```bash
# List all collections
curl http://localhost:8108/collections \
  -H "X-TYPESENSE-API-KEY: xyz"

# Search a collection directly
curl -X POST http://localhost:8108/collections/pdf_chunks/documents/search \
  -H "X-TYPESENSE-API-KEY: xyz" \
  -H "Content-Type: application/json" \
  -d '{"q":"Article 21","query_by":"content"}'
```

### Check Embedding Model

```python
# In Python REPL:
from backend.models.embeddings_model import EmbeddingsModel

model = EmbeddingsModel()
embedding = model.embed_query("Article 21")
print(f"Embedding dim: {len(embedding)}")  # Should be 1024
print(f"First 3 values: {embedding[:3]}")
```

### Frontend Network Inspection

Open browser DevTools (F12):
- **Network tab**: Inspect request/response headers, body, timing
- **Console tab**: View JavaScript errors, logs
- **React DevTools** (Chrome/Firefox extension): Inspect component state

---

## Environment Variables Cheat Sheet

| Variable | Purpose | Example |
|----------|---------|---------|
| `LLM_PROVIDER` | Which LLM to use | `gemini` or `ollama` |
| `GEMINI_API_KEY` | Gemini API key | `AIzaSy...` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `TYPESENSE_HOST` | Typesense server | `localhost` |
| `TYPESENSE_PORT` | Typesense port | `8108` |
| `TYPESENSE_API_KEY` | Typesense auth | (required) |
| `EMBEDDING_MODEL_NAME` | Hugging Face model | `yuriyvnv/legal-bge-m3` |
| `EMBEDDING_DEVICE` | CPU or GPU | `cpu` or `cuda` |
| `CORS_ALLOW_ORIGINS` | Frontend URLs | `http://localhost:5173` |
| `PDF_STORAGE_DIR` | PDF directory | `storage/pdfs` |
| `DEBUG` | Enable debug logging | `true` or `false` |

---

## Troubleshooting

### Typesense won't start

```bash
# Check Docker is running
docker ps

# View logs
docker-compose logs typesense

# Restart
docker-compose restart typesense
```

### Embedding model too large

If the embedding model is too large for your system:

1. Use a smaller model in `.env`:
   ```env
   EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
   EMBEDDING_DIMENSION=384
   ```

2. Ensure Typesense schema matches the new dimension.

### Tests fail with "connection refused"

Ensure services are running:

```bash
# Terminal 1
docker-compose up

# Terminal 2
ollama serve  # If using Ollama

# Terminal 3
pytest
```

### Frontend can't reach backend

Check CORS settings in `.env`:

```env
CORS_ALLOW_ORIGINS=http://localhost:5173,http://localhost:3000
```

And ensure frontend URL matches exactly (including port).

---

## Running in Production Mode Locally

```bash
# Start services in detached mode
docker-compose up -d

# Backend (single worker, production mode)
python backend/main.py

# Frontend (build static files)
cd frontend && npm run build && npm run preview
```

Open http://localhost:4173

