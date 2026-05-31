# Legal RAG AI — Frontend

React + TypeScript + Tailwind SPA (Vite). Three views over the backend API:

- **Upload** — drag/drop a PDF → ingest (store, chunk, embed, index) with progress.
- **Search** — hybrid/keyword/vector search → ranked chunks with page refs + highlight.
- **Chat** — ask a question → streamed (SSE) grounded answer with clickable citations.

## Prerequisites

- Node 18+
- The backend running on `http://localhost:8000` (see repo root README).

## Run

```bash
cd frontend
cp .env.example .env        # optional; defaults to http://localhost:8000
npm install
npm run dev                 # http://localhost:5173
```

`npm run build` type-checks (`tsc -b`) and produces a production bundle in `dist/`.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000` | Backend base URL |

## Notes

- Citation links point at `{API}{file_url}`. They resolve once the backend mounts
  static PDF serving (Step 5 hardening); until then they reference the path the
  backend would serve.
- The dev server port (5173) is already allow-listed in the backend `CORS_ALLOW_ORIGINS`.
