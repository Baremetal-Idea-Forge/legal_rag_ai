import { type SyntheticEvent, useRef, useState } from "react";
import { api } from "../api/client";
import type { Citation } from "../api/types";
import { Badge, ErrorBanner, pageRef } from "./ui";

export function ChatView() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [asked, setAsked] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  async function ask(e: SyntheticEvent) {
    e.preventDefault();
    const q = query.trim();
    if (!q || streaming) return;

    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setStreaming(true);
    setError(null);
    setAnswer("");
    setCitations([]);
    setAsked(q);

    await api.chatStream(q, 5, {
      signal: ctrl.signal,
      onToken: (t) => setAnswer((prev) => prev + t),
      onCitations: setCitations,
      onError: (err) => {
        setError(err.message);
        setStreaming(false);
      },
      onDone: () => setStreaming(false),
    });
  }

  function stop() {
    abortRef.current?.abort();
    setStreaming(false);
  }

  return (
    <div className="space-y-5">
      <form onSubmit={ask} className="space-y-3">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) void ask(e);
          }}
          rows={3}
          placeholder="Ask a question grounded in the legal corpus…  (Enter to send, Shift+Enter for newline)"
          className="w-full resize-y rounded-lg border border-slate-300 px-4 py-3 text-sm outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
        />
        <div className="flex items-center gap-2">
          <button
            type="submit"
            disabled={streaming || !query.trim()}
            className="rounded-lg bg-brand-600 px-5 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-50"
          >
            Ask
          </button>
          {streaming && (
            <button
              type="button"
              onClick={stop}
              className="rounded-lg border border-slate-300 px-4 py-2 text-sm text-slate-600 hover:bg-slate-50"
            >
              Stop
            </button>
          )}
        </div>
      </form>

      {error && <ErrorBanner message={error} />}

      {(answer || streaming) && (
        <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          {asked && (
            <p className="mb-3 text-sm font-medium text-slate-400">Q: {asked}</p>
          )}
          <div className="whitespace-pre-wrap text-[15px] leading-relaxed text-slate-800">
            {answer}
            {streaming && (
              <span className="ml-0.5 inline-block h-4 w-2 animate-pulse bg-brand-500 align-middle" />
            )}
          </div>

          {citations.length > 0 && (
            <footer className="mt-4 border-t border-slate-100 pt-3">
              <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">
                Sources
              </p>
              <div className="flex flex-wrap gap-2">
                {citations.map((c, i) => (
                  <CitationChip key={`${c.pdf_id}-${c.chunk_index}-${i}`} citation={c} />
                ))}
              </div>
            </footer>
          )}
        </section>
      )}
    </div>
  );
}

function CitationChip({ citation }: { citation: Citation }) {
  const link = api.fileUrl(citation.file_url);
  const label = `${citation.pdf_name} ${pageRef(citation.page_start, citation.page_end)}`;
  if (link) {
    return (
      <a href={link} target="_blank" rel="noreferrer" className="hover:opacity-80">
        <Badge tone="brand">{label}</Badge>
      </a>
    );
  }
  return <Badge tone="brand">{label}</Badge>;
}
