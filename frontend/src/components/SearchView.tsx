import { type FormEvent, useState } from "react";
import { api } from "../api/client";
import type { SearchMode, SearchResponse } from "../api/types";
import { ChunkCard } from "./ChunkCard";
import { EmptyState, ErrorBanner, Spinner } from "./ui";

const MODES: SearchMode[] = ["hybrid", "keyword", "vector"];

export function SearchView() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [topK, setTopK] = useState(5);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [submittedQuery, setSubmittedQuery] = useState("");

  async function run(e: FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.search(query.trim(), { topK, mode });
      setResult(res);
      setSubmittedQuery(query.trim());
    } catch (err) {
      setError((err as Error).message);
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-5">
      <form onSubmit={run} className="space-y-3">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search the legal corpus, e.g. “remedies for breach of contract”"
          className="w-full rounded-lg border border-slate-300 px-4 py-2.5 text-sm outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
        />
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 text-sm text-slate-600">
            Mode
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as SearchMode)}
              className="rounded-md border border-slate-300 px-2 py-1 text-sm"
            >
              {MODES.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 text-sm text-slate-600">
            Top-K
            <input
              type="number"
              min={1}
              max={50}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-20 rounded-md border border-slate-300 px-2 py-1 text-sm"
            />
          </label>
          <button
            type="submit"
            disabled={busy || !query.trim()}
            className="ml-auto rounded-lg bg-brand-600 px-5 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-50"
          >
            Search
          </button>
        </div>
      </form>

      {busy && <Spinner label="Searching…" />}
      {error && <ErrorBanner message={error} />}

      {result && !busy && (
        <div className="space-y-3">
          <p className="text-sm text-slate-500">
            {result.count} result{result.count === 1 ? "" : "s"} ·{" "}
            <span className="font-medium">{result.mode}</span>
          </p>
          {result.count === 0 ? (
            <EmptyState title="No matching chunks" hint="Try different wording or mode." />
          ) : (
            result.hits.map((hit) => (
              <ChunkCard key={hit.id} hit={hit} query={submittedQuery} />
            ))
          )}
        </div>
      )}
    </div>
  );
}
