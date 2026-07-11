import { useEffect, useState } from "react";
import { api } from "./api/client";
import type { StorageResponse } from "./api/types";
import { ChatView } from "./components/ChatView";
import { SearchView } from "./components/SearchView";
import { UploadView } from "./components/UploadView";

type Tab = "chat" | "search" | "upload";

const TABS: { id: Tab; label: string }[] = [
  { id: "chat", label: "Chat" },
  { id: "search", label: "Search" },
  { id: "upload", label: "Upload" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("chat");
  const [health, setHealth] = useState<"up" | "down" | "checking">("checking");
  const [version, setVersion] = useState<string>("");
  const [storage, setStorage] = useState<StorageResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .health()
      .then((h) => {
        if (cancelled) return;
        setHealth("up");
        setVersion(h.version);
      })
      .catch(() => !cancelled && setHealth("down"));
    api.storage().then((s) => !cancelled && setStorage(s)).catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="mx-auto flex min-h-full max-w-3xl flex-col px-4">
      <header className="flex items-center justify-between py-5">
        <div>
          <h1 className="text-xl font-semibold text-slate-900">Legal RAG AI</h1>
          <p className="text-sm text-slate-400">
            Hybrid search & grounded Q&amp;A over legal documents
          </p>
        </div>
        <div className="flex flex-col items-end gap-1.5">
          <HealthDot health={health} version={version} />
          {storage && <StorageBar storage={storage} />}
        </div>
      </header>

      <nav className="mb-5 flex gap-1 rounded-lg bg-slate-200/60 p-1">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex-1 rounded-md px-4 py-2 text-sm font-medium transition ${
              tab === t.id
                ? "bg-white text-brand-700 shadow-sm"
                : "text-slate-500 hover:text-slate-700"
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main className="flex-1 pb-12">
        {tab === "chat" && <ChatView />}
        {tab === "search" && <SearchView />}
        {tab === "upload" && <UploadView />}
      </main>
    </div>
  );
}

function HealthDot({
  health,
  version,
}: {
  health: "up" | "down" | "checking";
  version: string;
}) {
  const config = {
    up: { color: "bg-green-500", text: version ? `API v${version}` : "online" },
    down: { color: "bg-red-500", text: "API offline" },
    checking: { color: "bg-slate-300", text: "…" },
  }[health];
  return (
    <span className="flex items-center gap-2 text-xs text-slate-500">
      <span className={`h-2.5 w-2.5 rounded-full ${config.color}`} />
      {config.text}
    </span>
  );
}

function StorageBar({ storage }: { storage: StorageResponse }) {
  const pct = storage.disk_used_pct;
  const barColor =
    pct > 80 ? "bg-red-500" : pct > 60 ? "bg-yellow-500" : "bg-green-500";

  return (
    <div className="flex items-center gap-2 text-xs text-slate-500">
      <span>
        {storage.pdf_storage.num_files} files &middot;{" "}
        {storage.pdf_storage.total_size_human} / {storage.disk_total_human}
      </span>
      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-200">
        <div
          className={`h-full rounded-full ${barColor}`}
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
    </div>
  );
}
