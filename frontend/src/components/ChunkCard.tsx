import type { ChunkHit } from "../api/types";
import { api } from "../api/client";
import { Badge, pageRef } from "./ui";

/** Highlight whole-word query terms (3+ chars) within text. */
function highlight(text: string, query: string) {
  const terms = query
    .split(/\s+/)
    .map((t) => t.replace(/[^\p{L}\p{N}]/gu, ""))
    .filter((t) => t.length >= 3);
  if (terms.length === 0) return text;

  const alternation = terms.map(escapeRegex).join("|");
  const splitter = new RegExp(`(${alternation})`, "gi");
  const isTerm = new RegExp(`^(${alternation})$`, "i"); // non-global: stateless .test()

  return text
    .split(splitter)
    .map((part, i) =>
      isTerm.test(part) ? <mark key={i}>{part}</mark> : <span key={i}>{part}</span>,
    );
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function ChunkCard({ hit, query }: { hit: ChunkHit; query?: string }) {
  const link = api.fileUrl(hit.file_url);
  return (
    <article className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <header className="mb-2 flex flex-wrap items-center gap-2">
        {link ? (
          <a
            href={link}
            target="_blank"
            rel="noreferrer"
            className="font-medium text-brand-700 hover:underline"
          >
            {hit.pdf_name}
          </a>
        ) : (
          <span className="font-medium text-slate-700">{hit.pdf_name}</span>
        )}
        <Badge>{pageRef(hit.page_start, hit.page_end)}</Badge>
        <Badge tone="slate">chunk #{hit.chunk_index}</Badge>
        {typeof hit.score === "number" && (
          <Badge tone="brand">score {hit.score.toFixed(3)}</Badge>
        )}
      </header>
      <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-700">
        {query ? highlight(hit.content, query) : hit.content}
      </p>
    </article>
  );
}
