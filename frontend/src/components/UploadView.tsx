import { type DragEvent, useRef, useState } from "react";
import { api } from "../api/client";
import type { IngestResponse } from "../api/types";
import { Badge, ErrorBanner, Spinner } from "./ui";

export function UploadView() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function upload(file: File) {
    if (file.type !== "application/pdf") {
      setError("Only PDF files are supported.");
      return;
    }
    setBusy(true);
    setError(null);
    setResult(null);
    setProgress(0);
    try {
      const res = await api.ingest(file, setProgress);
      setResult(res);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) void upload(file);
  }

  return (
    <div className="space-y-4">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-14 text-center transition ${
          dragging ? "border-brand-500 bg-brand-50" : "border-slate-300 bg-white hover:bg-slate-50"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void upload(file);
            e.target.value = "";
          }}
        />
        <p className="font-medium text-slate-700">
          Drop a legal PDF here, or click to choose
        </p>
        <p className="mt-1 text-sm text-slate-400">
          Stored, chunked, embedded, and indexed for search & chat.
        </p>
      </div>

      {busy && (
        <div className="space-y-2">
          <Spinner label={progress < 100 ? `Uploading… ${progress}%` : "Indexing…"} />
          <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200">
            <div
              className="h-full bg-brand-600 transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {error && <ErrorBanner message={error} />}

      {result && (
        <div className="rounded-lg border border-green-200 bg-green-50 p-4">
          <div className="mb-2 flex items-center gap-2">
            <Badge tone="green">{result.status}</Badge>
            <span className="font-medium text-slate-800">{result.filename}</span>
          </div>
          <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm text-slate-600 sm:grid-cols-4">
            <Stat label="Pages" value={result.num_pages} />
            <Stat label="Chunks" value={result.num_chunks} />
            <Stat label="PDF id" value={result.pdf_id.slice(0, 12) + "…"} />
            <Stat label="SHA-256" value={result.sha256.slice(0, 12) + "…"} />
          </dl>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-slate-400">{label}</dt>
      <dd className="font-mono">{value}</dd>
    </div>
  );
}
