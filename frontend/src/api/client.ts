import { parseSSEStream } from "../lib/sse";
import type {
  ApiError,
  ChatResponse,
  Citation,
  HealthResponse,
  IngestResponse,
  SearchMode,
  SearchResponse,
  StorageResponse,
} from "./types";

const API_BASE = (
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

async function throwApiError(res: Response): Promise<never> {
  let message = `Request failed (${res.status})`;
  try {
    const body = (await res.json()) as ApiError;
    if (body?.message) message = body.message;
  } catch {
    /* non-JSON body — keep the generic message */
  }
  throw new Error(message);
}

export interface StreamHandlers {
  onToken: (token: string) => void;
  onCitations: (citations: Citation[]) => void;
  onDone: () => void;
  onError: (error: Error) => void;
  signal?: AbortSignal;
}

export const api = {
  base: API_BASE,

  fileUrl(path: string | null | undefined): string | null {
    return path ? `${API_BASE}${path}` : null;
  },

  async health(): Promise<HealthResponse> {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) return throwApiError(res);
    return res.json();
  },

  async storage(): Promise<StorageResponse> {
    const res = await fetch(`${API_BASE}/health/storage`);
    if (!res.ok) return throwApiError(res);
    return res.json();
  },

  /** Upload via XHR so we get real upload-progress events. */
  ingest(file: File, onProgress?: (pct: number) => void): Promise<IngestResponse> {
    return new Promise((resolve, reject) => {
      const form = new FormData();
      form.append("file", file);

      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/ingest`);

      xhr.upload.onprogress = (e) => {
        if (onProgress && e.lengthComputable) {
          onProgress(Math.round((e.loaded / e.total) * 100));
        }
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText) as IngestResponse);
        } else {
          let msg = `Upload failed (${xhr.status})`;
          try {
            msg = (JSON.parse(xhr.responseText) as ApiError).message ?? msg;
          } catch {
            /* keep generic */
          }
          reject(new Error(msg));
        }
      };
      xhr.onerror = () => reject(new Error("Network error during upload."));
      xhr.send(form);
    });
  },

  async search(
    query: string,
    opts: { topK?: number; mode?: SearchMode } = {},
  ): Promise<SearchResponse> {
    const res = await fetch(`${API_BASE}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_k: opts.topK ?? 5,
        mode: opts.mode ?? "hybrid",
      }),
    });
    if (!res.ok) return throwApiError(res);
    return res.json();
  },

  async chat(query: string, topK = 5): Promise<ChatResponse> {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK }),
    });
    if (!res.ok) return throwApiError(res);
    return res.json();
  },

  async chatStream(query: string, topK: number, h: StreamHandlers): Promise<void> {
    try {
      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: topK }),
        signal: h.signal,
      });
      if (!res.ok || !res.body) {
        h.onError(new Error(`Stream failed (${res.status})`));
        return;
      }
      for await (const evt of parseSSEStream(res.body)) {
        if (evt.event === "token") {
          h.onToken(JSON.parse(evt.data) as string);
        } else if (evt.event === "citations") {
          h.onCitations(JSON.parse(evt.data) as Citation[]);
        } else if (evt.event === "error") {
          const body = JSON.parse(evt.data) as { message?: string };
          h.onError(new Error(body.message ?? "Streaming failed."));
        } else if (evt.event === "done") {
          h.onDone();
        }
      }
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      h.onError(err as Error);
    }
  },
};
