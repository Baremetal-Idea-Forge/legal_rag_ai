# Configuration Guide

Complete reference for all environment variables and settings.

## Quick Start

```bash
# Copy the example file
cp .env.example .env

# Edit for your environment
nano .env

# Verify (restart app to load changes)
```

---

## Environment Variables Reference

All settings are defined in [backend/core/config.py](backend/core/config.py) and loaded from `.env` via `pydantic-settings`.

### App Identity

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `APP_NAME` | string | `"Legal RAG AI"` | Application display name |
| `APP_VERSION` | string | `"1.0.0"` | Version (matches pyproject.toml) |
| `DEBUG` | boolean | `false` | Enable debug logging & error details |
| `DEPLOY_DOMAIN` | string | `"localhost"` | Domain name (used in PDF URLs) |

**Example**:
```env
APP_NAME="Legal RAG AI"
APP_VERSION="1.0.0"
DEBUG=false
DEPLOY_DOMAIN=legal-rag.example.com
```

---

### Search & Indexing (Typesense)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TYPESENSE_HOST` | string | `"localhost"` | Typesense server hostname |
| `TYPESENSE_PORT` | integer | `8108` | Typesense server port |
| `TYPESENSE_PROTOCOL` | string | `"http"` | `http` or `https` |
| `TYPESENSE_API_KEY` | string | `"xyz"` | API key for authentication |
| `TYPESENSE_CONNECTION_TIMEOUT` | integer | `5` | Connection timeout (seconds) |
| `TYPESENSE_STARTUP_RETRIES` | integer | `5` | Retry attempts at startup |
| `TYPESENSE_STARTUP_RETRY_DELAY` | float | `2.0` | Delay between retries (seconds) |
| `TYPESENSE_TASKS_COLLECTION` | string | `"tasks"` | (Deprecated, kept for compatibility) |
| `TYPESENSE_PDF_CHUNKS_COLLECTION` | string | `"pdf_chunks"` | Collection name for indexed chunks |
| `TYPESENSE_DATA_DIR` | string | `"typesense-data"` | Directory for Typesense data files |

**Example (Local Development)**:
```env
TYPESENSE_HOST=localhost
TYPESENSE_PORT=8108
TYPESENSE_PROTOCOL=http
TYPESENSE_API_KEY=xyz
TYPESENSE_CONNECTION_TIMEOUT=5
TYPESENSE_STARTUP_RETRIES=10
TYPESENSE_STARTUP_RETRY_DELAY=2.0
```

**Example (Production)**:
```env
TYPESENSE_HOST=typesense.internal
TYPESENSE_PORT=8108
TYPESENSE_PROTOCOL=https
TYPESENSE_API_KEY=<strong-random-key>
TYPESENSE_CONNECTION_TIMEOUT=10
```

---

### Embeddings Model

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EMBEDDING_MODEL_NAME` | string | `"yuriyvnv/legal-bge-m3"` | Hugging Face model ID |
| `EMBEDDING_DEVICE` | string | `null` | `cpu`, `cuda`, `mps` (auto-detect if null) |
| `EMBEDDING_DIMENSION` | integer | `1024` | Model output dimension |
| `EMBEDDING_NORMALIZE` | boolean | `true` | L2-normalize embeddings |

**Supported Models**:

| Model | Dimension | Size | Latency | Notes |
|-------|-----------|------|---------|-------|
| `yuriyvnv/legal-bge-m3` | 1024 | 1.1 GB | ~500ms | **Recommended** for legal docs |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 80 MB | ~50ms | Lightweight, good for general text |
| `sentence-transformers/paraphrase-mpnet-base-v2` | 768 | 440 MB | ~200ms | Balanced size/quality |

**Example (CPU-only on VPS)**:
```env
EMBEDDING_MODEL_NAME=yuriyvnv/legal-bge-m3
EMBEDDING_DEVICE=cpu
EMBEDDING_DIMENSION=1024
EMBEDDING_NORMALIZE=true
```

**Example (GPU with CUDA)**:
```env
EMBEDDING_MODEL_NAME=yuriyvnv/legal-bge-m3
EMBEDDING_DEVICE=cuda
EMBEDDING_DIMENSION=1024
EMBEDDING_NORMALIZE=true
```

---

### LLM Provider Selection

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LLM_PROVIDER` | string | `"gemini"` | `"gemini"` or `"ollama"` |
| `LLM_TEMPERATURE` | float | `0.0` | Sampling temperature (0–1) |

#### Gemini (Google AI Studio)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GEMINI_API_KEY` | string | `""` | API key (required if LLM_PROVIDER=gemini) |
| `GEMINI_MODEL` | string | `"gemini-2.5-flash"` | Model name |
| `GEMINI_TIMEOUT_SECONDS` | integer | `120` | Request timeout |
| `GEMINI_MAX_RETRIES` | integer | `2` | Retry attempts |

**Setup**:
1. Get API key from https://aistudio.google.com/apikey
2. Set in `.env`:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIzaSy...
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0.0
```

#### Ollama (Local)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OLLAMA_BASE_URL` | string | `"http://localhost:11434"` | Ollama server URL |
| `OLLAMA_MODEL` | string | `"gemma3:4b"` | Model to use |
| `OLLAMA_TIMEOUT_SECONDS` | integer | `120` | Request timeout |
| `OLLAMA_MAX_RETRIES` | integer | `2` | Retry attempts |

**Setup**:
1. Install Ollama from https://ollama.ai
2. Pull a model: `ollama pull gemma3:4b`
3. Set in `.env`:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
OLLAMA_TEMPERATURE=0.0
```

**Available Models on Ollama**:
```bash
ollama pull gemma3:4b       # 4B, ~2.4 GB
ollama pull llama3:8b       # 8B, ~4.7 GB
ollama pull mistral:7b      # 7B, ~4.1 GB
ollama pull neural-chat:7b  # 7B, ~4.1 GB
```

---

### PDF Processing & OCR

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OCR_ENABLED` | boolean | `true` | Enable OCR for scanned PDFs |
| `OCR_MIN_TEXT_LENGTH` | integer | `50` | Min chars to consider OCR successful |
| `PDF_STORAGE_DIR` | string | `"storage/pdfs"` | Directory to store PDFs |
| `PDF_PUBLIC_BASE_URL` | string | `""` | Base URL for PDF links in responses |
| `CHUNK_MAX_CHARS` | integer | `1200` | Max chunk size (characters) |
| `CHUNK_OVERLAP_CHARS` | integer | `150` | Overlap between consecutive chunks |

**Example**:
```env
OCR_ENABLED=true
OCR_MIN_TEXT_LENGTH=50
PDF_STORAGE_DIR=/opt/legal-rag/storage/pdfs
PDF_PUBLIC_BASE_URL=https://legal-rag.example.com/pdfs
CHUNK_MAX_CHARS=1200
CHUNK_OVERLAP_CHARS=150
```

---

### RAG Parameters

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RAG_TOP_K` | integer | `5` | Default number of chunks to retrieve |
| `RAG_MAX_CONTEXT_CHARS` | integer | `12000` | Max context passed to LLM |

**Tuning**:

- **More chunks** (`RAG_TOP_K=10`): Better coverage, slower responses
- **Larger context** (`RAG_MAX_CONTEXT_CHARS=20000`): More information, token limit risk
- **Fewer chunks** (`RAG_TOP_K=3`): Faster, may miss information

**Example**:
```env
RAG_TOP_K=5
RAG_MAX_CONTEXT_CHARS=12000
```

---

### Upload & File Limits

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MAX_UPLOAD_BYTES` | integer | `52428800` | Max upload size (bytes, 50 MB) |
| `ALLOWED_UPLOAD_CONTENT_TYPE` | string | `"application/pdf"` | Accepted MIME type |

**Example**:
```env
MAX_UPLOAD_BYTES=52428800    # 50 MB
ALLOWED_UPLOAD_CONTENT_TYPE=application/pdf
```

---

### Storage & Disk Quotas

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DISK_QUOTA_GB` | integer | `80` | Max disk usage quota (GB) |

**Monitoring**:
- Check `/storage` endpoint to see current usage
- Quotas are **soft limits** (status warnings, not enforcement)

**Example**:
```env
DISK_QUOTA_GB=80
```

---

### Network & CORS

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CORS_ALLOW_ORIGINS` | string | `"http://localhost:5173,http://localhost:3000"` | Comma-separated origins |

**IMPORTANT**: Set explicitly for production (no `*` wildcard).

**Format**: Comma-separated, no spaces:
```env
# Development (multiple local ports)
CORS_ALLOW_ORIGINS=http://localhost:5173,http://localhost:3000

# Production
CORS_ALLOW_ORIGINS=https://legal-rag.example.com,https://app.example.com

# Multiple domains
CORS_ALLOW_ORIGINS=https://example.com,https://app.example.com,https://admin.example.com
```

---

### Authentication & Security

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `API_AUTH_TOKEN` | string | `""` | API key for write endpoints (disabled if empty) |

**Setup**:
1. Generate a strong token (at least 32 characters):
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
2. Set in `.env`:
   ```env
   API_AUTH_TOKEN=<your-generated-token>
   ```
3. Require auth on write endpoints: `/ingest`, `/ingest/bulk`

**Client Usage**:
```bash
# With Bearer token
curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer <your-token>" \
  -F "file=@document.pdf"

# With X-API-Key header
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: <your-token>" \
  -F "file=@document.pdf"
```

---

### Rate Limiting

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RATE_LIMIT_ENABLED` | boolean | `false` | Enable per-IP rate limiting |
| `RATE_LIMIT_PER_MINUTE` | integer | `60` | Max requests per minute |

**Note**: In-process limiter (single worker). For multi-worker deployments, use Redis or API gateway.

**Example**:
```env
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=100  # 100 req/min per IP
```

---

### MCP Server (Optional)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MCP_SERVER_URL` | string | `"http://localhost:9000"` | MCP retrieval server URL |
| `RETRIEVAL_VIA_MCP` | boolean | `false` | Route retrieval through MCP |

**Example**:
```env
MCP_SERVER_URL=http://localhost:9000
RETRIEVAL_VIA_MCP=false  # Set to true to use MCP
```

---

## Configuration Examples

### Local Development (Ollama)

```env
APP_NAME="Legal RAG AI"
APP_VERSION="1.0.0"
DEBUG=true

# Services
TYPESENSE_HOST=localhost
TYPESENSE_PORT=8108
TYPESENSE_API_KEY=xyz
TYPESENSE_DATA_DIR=typesense-data

# Embeddings
EMBEDDING_MODEL_NAME=yuriyvnv/legal-bge-m3
EMBEDDING_DEVICE=cpu
EMBEDDING_DIMENSION=1024

# LLM (local)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
LLM_TEMPERATURE=0.0

# Storage
PDF_STORAGE_DIR=storage/pdfs
MAX_UPLOAD_BYTES=52428800

# Development
CORS_ALLOW_ORIGINS=http://localhost:5173,http://localhost:3000
API_AUTH_TOKEN=
RATE_LIMIT_ENABLED=false
```

### Production (Gemini)

```env
APP_NAME="Legal RAG AI"
APP_VERSION="1.0.0"
DEBUG=false

# Services (Docker internal)
TYPESENSE_HOST=typesense
TYPESENSE_PORT=8108
TYPESENSE_PROTOCOL=https
TYPESENSE_API_KEY=<strong-random-key>
TYPESENSE_DATA_DIR=/data/typesense

# Embeddings (CPU on VPS)
EMBEDDING_MODEL_NAME=yuriyvnv/legal-bge-m3
EMBEDDING_DEVICE=cpu
EMBEDDING_DIMENSION=1024

# LLM (Gemini API)
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIzaSy...
GEMINI_MODEL=gemini-2.5-flash
LLM_TEMPERATURE=0.0

# Storage
PDF_STORAGE_DIR=/opt/legal-rag/storage/pdfs
PDF_PUBLIC_BASE_URL=https://legal-rag.example.com/pdfs
DISK_QUOTA_GB=200

# Security
CORS_ALLOW_ORIGINS=https://legal-rag.example.com
API_AUTH_TOKEN=<strong-random-token>
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=100
```

### High-Traffic Production

```env
APP_NAME="Legal RAG AI"
APP_VERSION="1.0.0"
DEBUG=false

# Services (HA setup)
TYPESENSE_HOST=typesense-lb.internal
TYPESENSE_PORT=8108
TYPESENSE_PROTOCOL=https
TYPESENSE_API_KEY=<strong-random-key>
TYPESENSE_CONNECTION_TIMEOUT=10
TYPESENSE_STARTUP_RETRIES=15

# Embeddings (GPU)
EMBEDDING_MODEL_NAME=yuriyvnv/legal-bge-m3
EMBEDDING_DEVICE=cuda
EMBEDDING_DIMENSION=1024

# LLM
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIzaSy...
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TIMEOUT_SECONDS=180
GEMINI_MAX_RETRIES=3

# Storage (large quota)
PDF_STORAGE_DIR=/mnt/storage/pdfs
PDF_PUBLIC_BASE_URL=https://legal-rag.example.com/pdfs
DISK_QUOTA_GB=500

# RAG (optimized)
RAG_TOP_K=10
RAG_MAX_CONTEXT_CHARS=15000

# Security & Limits
CORS_ALLOW_ORIGINS=https://legal-rag.example.com
API_AUTH_TOKEN=<strong-random-token>
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=200
MAX_UPLOAD_BYTES=104857600  # 100 MB
```

---

## Validation & Troubleshooting

### Check Configuration at Runtime

```bash
# View loaded settings (backend/core/config.py)
python -c "
from backend.core.config import get_settings
settings = get_settings()
print(f'LLM: {settings.LLM_PROVIDER}')
print(f'Embedding model: {settings.EMBEDDING_MODEL_NAME}')
print(f'Typesense: {settings.TYPESENSE_HOST}:{settings.TYPESENSE_PORT}')
"
```

### Common Issues

**Problem**: `GEMINI_API_KEY not set`
- **Solution**: Get key from https://aistudio.google.com/apikey, add to `.env`

**Problem**: `TYPESENSE_API_KEY="xyz"` fails auth
- **Solution**: Generate strong key, update both `.env` and Typesense config

**Problem**: Embedding model too large
- **Solution**: Use smaller model or increase system RAM (see Supported Models table)

**Problem**: CORS errors in frontend
- **Solution**: Verify `CORS_ALLOW_ORIGINS` matches frontend URL exactly

---

## Performance Tuning

### For Latency

```env
# Smaller model, fewer chunks, smaller context
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
RAG_TOP_K=3
RAG_MAX_CONTEXT_CHARS=8000
```

### For Quality

```env
# Larger model, more chunks, larger context
EMBEDDING_MODEL_NAME=yuriyvnv/legal-bge-m3
RAG_TOP_K=10
RAG_MAX_CONTEXT_CHARS=20000
```

### For Resource Usage

```env
# CPU-only, smaller model
EMBEDDING_DEVICE=cpu
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
CHUNK_MAX_CHARS=800  # Smaller chunks = less memory
```

---

## Reloading Configuration

```bash
# After changing .env, restart the app:
# Development
pkill -f "fastapi dev"
fastapi dev backend/main.py

# Production (systemd)
sudo systemctl restart legal-rag-backend
```

