# Legal RAG AI — API Documentation

Complete reference for all HTTP endpoints, request/response schemas, error codes, and usage examples.

## Base URL

```
http://localhost:8000  (development)
https://your-domain.com  (production)
```

## Authentication

### Optional API Key Auth

If `API_AUTH_TOKEN` is set in `.env`, all write endpoints require authentication via either:

1. **Bearer Token** (recommended):
   ```
   Authorization: Bearer <your-api-token>
   ```

2. **API Key Header**:
   ```
   X-API-Key: <your-api-token>
   ```

If `API_AUTH_TOKEN` is empty (default), auth is disabled.

---

## Health & Status Endpoints

### `GET /health`

**Liveness check** — always returns 200 if the process is running.

**Response (200)**:
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

**Use case**: Load balancer liveness probe. Safe to poll frequently.

---

### `GET /health/ready`

**Readiness check** — verifies all required dependencies (Typesense, LLM) are reachable.

**Response (200 — all systems ready)**:
```json
{
  "status": "ready",
  "version": "1.0.0",
  "checks": {
    "typesense": true,
    "llm": true
  }
}
```

**Response (503 — degraded)**:
```json
{
  "status": "degraded",
  "version": "1.0.0",
  "checks": {
    "typesense": false,
    "llm": true
  }
}
```

**Use case**: Load balancer readiness probe. Return 503 if not ready to receive traffic.

---

### `GET /storage`

**Storage info** — disk usage and quota.

**Response (200)**:
```json
{
  "total_bytes": 5368709120,
  "used_bytes": 2147483648,
  "quota_bytes": 85899345920,
  "percent_used": 2.5,
  "status": "ok"
}
```

**Fields**:
- `total_bytes`: Total disk space available on the volume
- `used_bytes`: Bytes used by PDFs + Typesense data
- `quota_bytes`: Configured quota (from `DISK_QUOTA_GB`)
- `percent_used`: Percentage of quota used
- `status`: `"ok"` (< 80%) | `"warning"` (80–95%) | `"critical"` (≥ 95%)

---

## Search Endpoints

### `POST /search`

**Hybrid search** (keyword + vector).

**Request**:
```json
{
  "query": "What are the rights under Article 21?",
  "top_k": 5,
  "mode": "hybrid",
  "filter_by": "pdf_id:=abc123"
}
```

**Request Fields**:
| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `query` | string | **required** | 1–2000 chars | Search text |
| `top_k` | integer | `5` | 1–50 | Max results |
| `mode` | string | `"hybrid"` | `keyword` \| `vector` \| `hybrid` | Search mode |
| `filter_by` | string | `null` | Typesense syntax | Filter expression (see below) |

**Response (200)**:
```json
{
  "query": "What are the rights under Article 21?",
  "mode": "hybrid",
  "count": 3,
  "hits": [
    {
      "id": "abc-001",
      "pdf_id": "sha256abc",
      "pdf_name": "Constitution.pdf",
      "page_start": 15,
      "page_end": 15,
      "chunk_index": 42,
      "content": "Article 21. Protection of life and personal liberty: No person shall be deprived of...",
      "file_url": "http://localhost:8000/pdfs/abc.pdf#page=15",
      "score": 0.9847,
      "text_match": null,
      "vector_distance": 0.0153
    }
  ]
}
```

**Response Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `query` | string | Echo of input query |
| `mode` | string | Search mode used |
| `count` | integer | Number of hits returned |
| `hits` | array | List of matching chunks |

**ChunkHit Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique chunk ID |
| `pdf_id` | string | SHA256 hash of PDF |
| `pdf_name` | string | Original filename |
| `page_start` | integer | Starting page number |
| `page_end` | integer | Ending page number |
| `chunk_index` | integer | Sequence number within PDF |
| `content` | string | Chunk text |
| `file_url` | string | Permalink to PDF + page number |
| `score` | float | Relevance score (0–1) |
| `text_match` | integer | Keyword match score (BM25) |
| `vector_distance` | float | Cosine distance from query embedding |

#### Filter Syntax

Use Typesense filter expressions to narrow results:

```
pdf_id:=abc123                    # Exact match
page_start:[10..20]               # Range (pages 10–20)
pdf_name:="Constitution.pdf"      # String equality
page_start:>15                    # Greater than
page_start:<20 && page_end:>15   # AND condition
```

See [Typesense filtering](https://typesense.org/docs/latest/api/search/#search-results-parameters) for full syntax.

#### Search Modes

| Mode | Description | When to Use |
|------|-------------|-------------|
| `keyword` | BM25 keyword search | Legacy terms, exact phrase matching |
| `vector` | Semantic similarity | Conceptual questions |
| `hybrid` | Keyword + vector combined | General queries, best overall results |

---

### `GET /search`

Same as `POST /search` but via query string.

**Example**:
```
GET /search?q=Article+21&top_k=10&mode=hybrid
```

**Query Parameters**:
- `q` (required): Search text
- `top_k` (optional, default 5): Max results
- `mode` (optional, default "hybrid"): Search mode
- `filter_by` (optional): Filter expression

---

## Chat Endpoints

### `POST /chat`

**Synchronous Q&A** with citations.

**Request**:
```json
{
  "query": "Can the government detain someone without cause under Article 21?",
  "top_k": 5
}
```

**Request Fields**:
| Field | Type | Default | Constraints |
|-------|------|---------|-------------|
| `query` | string | **required** | 1–2000 chars |
| `top_k` | integer | `5` | 1–50 |

**Response (200)**:
```json
{
  "query": "Can the government detain someone without cause under Article 21?",
  "answer": "No. Article 21 states that no person shall be deprived of life or personal liberty except according to the procedure established by law. This implies that detention must be lawful and follow due process.",
  "citations": [
    {
      "pdf_id": "sha256abc",
      "pdf_name": "Constitution.pdf",
      "page_start": 15,
      "page_end": 15,
      "chunk_index": 42,
      "file_url": "http://localhost:8000/pdfs/abc.pdf#page=15"
    }
  ],
  "chunks_used": [/* full ChunkHit objects used for context */]
}
```

**Response Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `query` | string | Echo of input query |
| `answer` | string | Grounded answer from LLM |
| `citations` | array | Source documents cited |
| `chunks_used` | array | Full context chunks (for debugging) |

**Error Responses**:

| Status | Error Code | Description |
|--------|-----------|-------------|
| 400 | `validation_error` | Invalid query format |
| 429 | `rate_limited` | Too many requests |
| 502 | `search_error` | Typesense unavailable |
| 503 | `llm_error` | LLM (Gemini/Ollama) unavailable |
| 503 | `upstream_unavailable` | Required dependency down |

**No Context Scenario**:

If the corpus doesn't contain relevant information, the response is:
```json
{
  "query": "What color is the sky in the Constitution?",
  "answer": "I don't have information about that in the indexed documents.",
  "citations": [],
  "chunks_used": []
}
```

---

### `POST /chat/stream`

**Streaming Q&A** via Server-Sent Events (SSE).

**Request**: Same as `POST /chat`

**Response (200)**: SSE stream

```
event: token
data: "No."

event: token
data: " Article"

event: token
data: " 21"

event: token
data: " states"

...

event: citations
data: [{"pdf_id":"sha256abc","pdf_name":"Constitution.pdf",...}]

event: done
data: {}
```

**Event Types**:

| Event | Data | Description |
|-------|------|-------------|
| `token` | string | A single generated token |
| `citations` | JSON array | Final citations (sent before `done`) |
| `error` | JSON object | Stream error (see below) |
| `done` | `{}` | Stream complete |

**Error Event**:
```
event: error
data: {"error":"llm_error","message":"LLM request timed out"}
```

**Client Example** (JavaScript):

```javascript
const response = await fetch('/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: 'Article 21?', top_k: 5 })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const text = decoder.decode(value);
  const lines = text.split('\n\n');
  
  for (const line of lines) {
    if (!line) continue;
    const [eventLine, dataLine] = line.split('\n');
    const event = eventLine.replace('event: ', '');
    const data = dataLine.replace('data: ', '');
    
    if (event === 'token') {
      console.log(data);  // Append token to display
    } else if (event === 'citations') {
      console.log('Citations:', JSON.parse(data));
    } else if (event === 'error') {
      console.error('Stream error:', JSON.parse(data));
    }
  }
}
```

---

## Ingestion Endpoints

### `POST /ingest`

**Upload and index a single PDF**.

Requires `API_AUTH_TOKEN` if enabled.

**Request** (multipart/form-data):
```
POST /ingest
Content-Type: multipart/form-data

file=@/path/to/legal_doc.pdf
```

**Response (201)**:
```json
{
  "pdf_id": "e6b887ce91f4e199d33a966cdb98662861c2ad27735483493bee0780d6d6dbf2",
  "filename": "legal_doc.pdf",
  "sha256": "e6b887ce91f4e199d33a966cdb98662861c2ad27735483493bee0780d6d6dbf2",
  "num_pages": 45,
  "num_chunks": 112,
  "status": "indexed"
}
```

**Response Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `pdf_id` | string | SHA256 hash of file content (unique ID) |
| `filename` | string | Original filename |
| `sha256` | string | Content hash |
| `num_pages` | integer | Page count |
| `num_chunks` | integer | Number of chunks after splitting |
| `status` | string | Always `"indexed"` on success |

**Error Responses**:

| Status | Code | Description |
|--------|------|-------------|
| 401 | `unauthorized` | Missing/invalid API key |
| 422 | `ingestion_error` | PDF parsing failed, unsupported format |
| 413 | `ingestion_error` | File exceeds `MAX_UPLOAD_BYTES` |
| 429 | `rate_limited` | Too many uploads in short time |

**Constraints**:
- Max file size: `MAX_UPLOAD_BYTES` (default: 50 MB)
- Content-Type: `application/pdf` only
- Max 1000+ chunks per PDF (depends on content)

---

### `POST /ingest/bulk`

**Ingest all PDFs from the data/ directory** (admin operation).

Requires `API_AUTH_TOKEN` if enabled.

**Request**: Empty body

```
POST /ingest/bulk
```

**Response (200)**:
```json
{
  "ingested": [
    {
      "pdf_id": "abc123...",
      "filename": "doc1.pdf",
      "sha256": "abc123...",
      "num_pages": 10,
      "num_chunks": 25,
      "status": "indexed"
    }
  ],
  "failed": [
    {
      "filename": "corrupt.pdf",
      "error": "PDF parsing failed: encrypted document"
    }
  ],
  "total_ingested": 42,
  "total_failed": 2
}
```

**Response Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `ingested` | array | Successfully indexed PDFs |
| `failed` | array | Failed PDFs with error details |
| `total_ingested` | integer | Count of successful ingestions |
| `total_failed` | integer | Count of failures |

---

## Common Error Responses

All errors follow this JSON schema:

```json
{
  "detail": "error message"
}
```

Or (with custom exception handlers):

```json
{
  "error": "error_code",
  "message": "Human-readable message",
  "detail": null
}
```

### HTTP Status Codes

| Status | Meaning |
|--------|---------|
| 200 | Success |
| 201 | Created (ingestion success) |
| 400 | Bad request (validation error) |
| 401 | Unauthorized (missing/invalid auth) |
| 404 | Not found |
| 429 | Rate limited |
| 500 | Internal server error |
| 502 | Bad gateway (upstream error) |
| 503 | Service unavailable (dependency down) |

---

## Rate Limiting

If `RATE_LIMIT_ENABLED=true` in `.env`:

- **Limit**: `RATE_LIMIT_PER_MINUTE` requests per minute per IP
- **Header**: `X-RateLimit-Limit` (optional, not currently exposed)
- **Error Response** (429):
  ```json
  {
    "error": "rate_limited",
    "message": "Rate limit exceeded. Please slow down."
  }
  ```

---

## Testing Endpoints

### Quick Test Script

```bash
#!/bin/bash

BASE_URL="http://localhost:8000"

# 1. Health check
echo "=== Health Check ==="
curl -s "$BASE_URL/health" | jq .

# 2. Search
echo "=== Search ==="
curl -s -X POST "$BASE_URL/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"Article 21","top_k":3}'

# 3. Chat
echo "=== Chat ==="
curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is Article 21?","top_k":5}'

# 4. Upload (with auth)
echo "=== Upload ==="
curl -s -X POST "$BASE_URL/ingest" \
  -H "X-API-Key: your-token" \
  -F "file=@test.pdf"
```

---

## Rate Limit & Quotas

| Limit | Value | Notes |
|-------|-------|-------|
| Max query length | 2000 chars | Applies to search + chat |
| Max top_k | 50 | Prevents resource exhaustion |
| Max file size | 50 MB | Per `MAX_UPLOAD_BYTES` |
| Max requests/min | Configured | If `RATE_LIMIT_ENABLED` |
| Max context length | 12,000 chars | Sent to LLM (truncates if needed) |

---

## API Stability & Versioning

- **Current Version**: 1.0.0
- **Compatibility**: This API is stable. Breaking changes will be announced and versioned as v2.0.
- **Deprecation**: Endpoints deprecated for ≥ 90 days before removal.

