# Legal RAG — MCP Retrieval Server

A read-side [Model Context Protocol](https://modelcontextprotocol.io) server that
exposes the system's retrieval stack as tools:

| Tool | Args | Returns |
|---|---|---|
| `search_legal_chunks` | `query`, `top_k=5`, `mode="hybrid"`, `filter_by=None` | `{"hits": [ChunkHit, …]}` |
| `get_document` | `chunk_id` | `{"document": {…}}` |

It imports the backend's `SearchService`, embedding model, and Typesense config
directly — **one retrieval stack, one config** across the whole system (DRY).

## Topology

```
Backend (FastAPI)  ──MCP──►  mcp-server  ──►  Embedding model + Typesense
        │
        └── when RETRIEVAL_VIA_MCP=true, RagService retrieves via this server
            (default false → backend calls SearchService directly)
```

Ingestion (the write path) always stays in the backend; only search (read) is
routed through MCP.

## Run

```bash
# from the repo root
uv run python mcp-server/server.py
```

Listens on `MCP_SERVER_URL` (default `http://localhost:9000`) at path `/mcp`
using the streamable-HTTP transport. Requires Typesense to be reachable (and,
on first query, downloads/loads the embedding model).

## Enable in the backend

Set in `.env`:

```
RETRIEVAL_VIA_MCP=true
MCP_SERVER_URL=http://localhost:9000
```
