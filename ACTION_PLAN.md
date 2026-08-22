# Legal RAG System — Action Plan

# Legal RAG on Apple Silicon — System Design v2

**Status:** Proposed — pre-implementation, for review. **Supersedes v1** (RTX 4090 envelope, brief-only citations).
**Date:** 2026-07-30
**Scope:** 12,000-document contract corpus, +300/month, fully local, no network at inference
**Hardware envelope:** Apple silicon, 64GB unified memory (binding case) or 128GB (comfortable case)

---

## 0. What changed between v1 and v2, and why

Two inputs changed: the six papers arrived and were read in full, and the hardware envelope moved from an RTX 4090 workstation to Apple silicon. Both forced real design changes, not just re-labeling. This section is the audit trail.

### 0.1 The papers, verified

All six are now read directly: **Reuter et al. 2025** (SAC/DRM, arXiv:2510.06999), **Pipitone & Alami 2024** (LegalBench-RAG, arXiv:2408.10343), **Kalra et al. 2024** (HyPA-RAG, CustomNLP4U), **Wahidur et al. 2025** (LQ-RAG, IEEE Access), **Wiratunga et al. 2024** (CBR-RAG, arXiv:2404.04302), and **Hindi et al. 2025** (survey, IEEE Access). The brief's distillation was accurate on every headline claim. But the full texts contain results the brief omitted, and three of them change decisions v1 made. Where a section number is cited below, it is now verified against the paper, not echoed from the brief.

| # | v1 position | What the paper actually shows | v2 action |
|---|---|---|---|
| R1 | Added BM25+RRF hybrid as default (L6), flagged "not in evidence base" | It **is** in the evidence base: Reuter App. B, Table 2 tested dense+sparse. Raising keyword weight improved DRM slightly (19.29→18.18%) but cut text-level precision (11.03→8.23%) and recall (41.8→36.6%); they went dense-only. Mechanism: summaries are identifier-rich (keyword-friendly) but chunk bodies are natural language, so BM25 finds documents and loses spans | **L6 changed.** Dense-only is the default. Lexical becomes a *conditional route* fired only when the router flags an exact-term query (quoted phrase, §-number, defined term). Probe gates even that |
| R2 | Metadata-contrastive summaries as expected winner (L2, medium confidence) | Reuter's **expert-guided** summaries (§5.1–5.3, Ex. 1, App. D) are the nearest tested neighbor of v1's proposal: structured, identifier-rich (parties, term, governing law). Result: matched generic at *document* level but surfaced introductory **boilerplate at span level** — the retrieved chunk was useless for the query. Their hypotheses: narrow-cue overfitting, and embedder capacity strain from dense structured text (§5.3) | **L2 changed.** Generic-150 is now the default summary. Contrastive-delta survives only as a probe arm, with a new **span-precision guard** in its adoption rule and an honestly lowered prior (§3 P1) |
| R3 | Summary length 150 vs 300 framed as open precision-vs-discrimination trade (P2) | App. A, Table 1: at chunk 500, the 150-char summary beat 300 on precision **and** recall **and** DRM (19.29% vs 20.61%). Longer summaries didn't even help document-level discrimination at their scale | P2 prior shifts to the dilution mechanism. Factorial retained, expectation updated |
| R4 | Reranker excluded on Pipitone's negative | Now a genuine **cross-paper conflict**: Pipitone Tables 4 vs 6 — Cohere rerank helped PrivacyQA (P@1 7.86→14.38) but collapsed ContractNLI (16.45→6.63) and hurt the aggregate. Kalra Table 3 — bge-reranker-large gave the *best* faithfulness (0.9098) on LL144 | Conflict resolved by setting-match: our corpus is ContractNLI-shaped (near-duplicate contracts, large pool); Kalra's corpus is a single 15-page statute where DRM cannot exist. **Exclusion stands**, now properly argued (§12) |
| R5 | Kalra chunking cited as "delimiter beats semantic" | Verified with nuance: Fig. 3 — pattern-based ("\n§") won recall/faithfulness/similarity/correctness; sentence-level won precision/F1; semantic lost broadly; and Kalra then proceeded with sentence-level 512-token chunks + 200 overlap. All measured on **15 pages** | L3 unchanged, but its main support now rests on Pipitone (RCTS-500, no overlap, beat naive at 714-doc scale, §4.3) and Reuter's baseline; Kalra contributes only "semantic loses," with a scale caveat on every HyPA number |
| R6 | SVM chosen over DistilBERT at 0.86 vs 0.90 (T6) | Kalra Table 8: on the **binary** task, TF-IDF+SVM ties fine-tuned DistilBERT at 0.92 | T6 strengthened; our router is effectively binary |
| R7 | Embedding fine-tuning staged, entry criteria | Wahidur §VI-A, Fig. 2: the +13% HR / +15% MRR are relative gains, trained and evaluated on **synthetic GPT-3.5 query-context pairs from a legal book corpus** — in-distribution eval, different document genre from contracts | Staged non-goal reinforced; the distribution caveat is now citable rather than inferred |
| R8 | Min-of-triad gate, abstain-first | Wahidur Table 12 (closed-domain): naive RAG scored AR 0.88 / CR 0.31 / G 0.26; LQ-RAG's regenerate-on-fail loop lifted G only to 0.35 while doubling latency (Fig. 9: 7.2s→14.6s) | Gate design confirmed; **regenerate loops added to non-goals** — abstention is cheaper and more honest than iterated regeneration that still fails the triad |
| R9 | DRM >95% "in pools of hundreds" | Verified: pool of 362 documents, ContractNLI (§3.2, Fig. 2a). Sharper: LegalBench-RAG queries **contain a document description in the query text** (Pipitone §3.1.3, format "Consider the X agreement between A and B; …") and DRM was still >95% | P5 strengthened: identity stated in the query does not bind identity in embedding space. Symbolic metadata binding is not an optimization; it is the only reliable binder |
| R10 | Domain-pretrained embedders excluded | Doubly verified: Reuter App. C Fig. 4 (legal-bert near-flat on both precision and recall) and Wiratunga §6.1 Fig. 4 (LegalBERT similarities densely clustered → poor discrimination), §7 | Unchanged; L4 stands |

### 0.2 The hardware inversion

Verified against Apple's current lineup: the Mac mini tops out at **64GB unified memory** (M4 Pro, 14-core CPU / 20-core GPU, 273 GB/s). No 128GB mini exists; **128GB unified means Mac Studio-class silicon** (M4 Max, 410–546 GB/s) or a future higher-spec mini. This document therefore budgets two columns: **column A** = Mac mini M4 Pro 64GB (the binding, concrete SKU), **column B** = 128GB Studio-class (the comfortable case). The architecture is identical; only throughput and headroom differ.

The headline finding, and it inverts v1: **on this envelope, memory is not the constraint — time is.** On the 4090, the design fought over 24GB of VRAM partitioning while wall-clock was cheap. On Apple silicon, unified memory swallows the entire steady-state system (~22GB at 12k docs, ~32GB at 50k — comfortable even at 64GB), but ingestion throughput and per-query decode are 3–6× slower than the 4090. Cold start stretches from "one working day" to **~1.5–2.5 days (A)** / **~0.5–1.25 days (B)**; per-query p50 from 4–6s to **~10–16s (A)** / **~6–9s (B)**; and a full re-summarization at 50k documents on column A is **on the order of a week**. Consequences ripple through the whole document: the decide-once discipline (P7) hardens from hygiene into schedule-critical; the degradation ladder (§11) reorders around time instead of bytes; and early abstention (Gate 1) graduates from efficiency nicety to the single biggest latency lever.

### 0.3 Assumptions v2

A1–A7 carry over unchanged from v1 (20% scanned, ~4 chars/token, ~2MB/PDF, zero-overlap chunking, ~200 chunks/doc, ~170-token embed inputs, ~3.5k-in/300-out summarizer calls). A8 is replaced:

| # | Assumption | Column A (mini M4 Pro 64GB) | Column B (128GB Studio-class) |
|---|---|---|---|
| A8′-1 | 8B Q4 prefill (batched) | 0.5–1.2k tok/s | 1–2.5k tok/s |
| A8′-2 | 8B Q4 decode, single stream | 30–45 tok/s | 55–80 tok/s |
| A8′-3 | 8B Q4 decode, batched aggregate | 150–400 tok/s | 300–800 tok/s |
| A8′-4 | 335M encoder fp16 (MLX/Metal) | 6–12k tok/s | 12–25k tok/s |
| A8′-5 | NLI claim–span pair | 20–60ms | 10–30ms |
| A8′-6 | OCR (Tesseract, CPU) | ~1.2 s/page/worker × 12 workers | similar |

Uncertainty on A8′ is **±2–3×** — wider than the CUDA estimates in v1, because Metal/MLX serving maturity varies by model and library. Every wall-clock figure below inherits this, and step B0 (§8) re-measures all six numbers on the actual machine before any schedule is promised. Serving stack: llama.cpp or MLX (mlx-lm) for the 8B; MLX/Core ML for the encoder and NLI; hnswlib/FAISS on CPU; everything offline.

---

## 1. Decision ledger v2 (▲ = changed from v1)

The system remains a **flat, summary-augmented dense chunk index** with symbolic metadata binding, one 8B model on triple duty, int8 vectors with fp16 exact rescoring, an NLI-first groundedness cascade, and abstention as a first-class outcome. What changed: the default summary is now Reuter's actual winner rather than my extension of it, the lexical channel is conditional rather than default, and every budget is re-derived for unified memory.

| # | Decision | Choice | Traces to (verified) | Confidence | Gate |
|---|---|---|---|---|---|
| L1 | Retrieval architecture | Flat SAC chunk index; no doc-level stage on critical path | Reuter §3.2 (DRM >95% @362 docs = doc identity is the hard part) | High | Probe |
| L2 ▲ | Summary type | **Generic-150 default**; contrastive-delta as probe arm only | Reuter §5.1–5.3 (generic beat expert; expert = span-level boilerplate failure, Ex. 1) | High (for default) | Probe arm C vs. B under span guard |
| L3 | Chunking | Delimiter-first recursive → ~500 chars, no overlap | Pipitone §4.3 (RCTS > naive); Reuter §4, App. A; Kalra §6 (semantic loses; 15-page caveat) | High | Spot-check |
| L4 | Embedder | gte-large-class, 335M, 1024d, fp16 | Reuter App. C Fig. 4; Wiratunga §6.1 Fig. 4, §7 | High | gte-base ≤1pt → ladder rung |
| L5 | Vector storage | int8 SQ in HNSW + fp16 mmap exact rescore of fused top-200 | §5 budget; recall math | High | int8 within 0.5pt of fp16 on probe |
| L6 ▲ | Lexical channel | **Conditional route only** — router-flagged exact-term queries (quoted phrases, §-numbers, defined terms); dense-only otherwise | Reuter App. B Table 2 (BM25: DRM ↓ slightly, precision/recall ↓ clearly) | Medium-high | Probe exact-term slice; removable rung |
| L7 | Reranker | None | Pipitone §5.1, Tables 4/6; conflict with Kalra Table 3 resolved by setting-match (§12) | High | Non-goal |
| L8 | Metadata binding | Hard filter on high-confidence extractions; soft boost otherwise | Pipitone §3.1.3 + Reuter §3.2 (identity in query text still yields >95% DRM) | High | Extraction precision audit ≥97% |
| L9 | Generator/summarizer | One 8B-class instruct, 4-bit; 3B is a real ladder rung on this hardware | Cost math §7 (3B halves the dominant cold-start pass) | Medium-high | 3B probe arm |
| L10 | Groundedness | NLI cross-encoder cascade; 8B judges borderline ~15% | Wahidur Table 12, Fig. 8 (min-of-triad; AR alone dangerous) | Medium | κ ≥0.9 vs. 8B judge on probe |
| L11 | Query rewriting | Off by default; one rewrite retry on weak-retrieval path pre-abstention | Kalra Table 3 (k,K,S,Q below k,K,S on correctness; Q=3–7 on a 15-page corpus doesn't transfer); latency on A | Medium | A/B on weak slice |
| L12 | Complexity router | TF-IDF + SVM | Kalra Tables 1 & 8 (binary tie at 0.92) | High | Routing-error slice |
| L13 | Near-duplicates | Cluster (MinHash), never dedup; deltas feed the contrastive probe arm and the audit strata | The 95%-DRM condition; legal-instrument integrity | High | Probe cluster cells |
| L14 | Excluded | Reranker, KG, semantic chunking, expert summaries, regenerate loops, RAPTOR/late chunking; fine-tuning staged | §12, all citations verified | High | — |

---

## 2. Architecture (Deliverable 1)

### 2.1 Component diagram (v2)

```
 OFFLINE — INGESTION                              ONLINE — QUERY
 ═══════════════════                              ══════════════
 PDF drop (watch folder)                          user query
      │                                                │
 [I1] Extract / OCR ───► canonical text          [Q1] normalize: entity alias
      │   (PyMuPDF; Tesseract fallback,                (Aho-Corasick), date parse,
      │    CPU workers)                                exact-term detector
      │   + char↔page/bbox offset map                  (quoted phrase / §-number)
      ▼                                                │
 [I2] near-dup clustering (MinHash/LSH)          [Q2] route: TF-IDF+SVM tier
      │   cluster ID + delta-vs-representative         → k, mode, lexical flag
      ▼                                                │
 [I3] chunker: section delimiters →              [Q3] metadata pre-filter
      │   recursive ≤500 chars, offsets kept           (SQLite; hard filter only on
      ▼                                                high-confidence bindings)
 [I4] 8B LLM, one call/doc:                            │
      │   metadata JSON + GENERIC-150            [Q4] dense HNSW int8 top-200
      │   summary (contrastive only in probe)          │
      │       ├──► metadata DB ◄───────────      [Q4′] conditional lexical route:
      │       ▼                                        only if exact-term flag —
 [I5] embedder (gte-large):                            BM25 top-50 → RRF w/ dense
      │   embed(summary ⊕ chunk)                       │
      │   → fp16 shards (SSD, mmap)              [Q5] fp16 exact rescore top-200
      │   → int8 quantize → HNSW                       group by doc, top-k
      ▼                                                │
 [I6] small BM25 index (conditional route)       [Q6] GATE 1: score floor
      │                                                │ fail → ABSTAIN (+nearest
 [I7] manifest commit                                  │  docs; 0 LLM calls; optional
      (doc_sha × stage × config_hash)                  │  1 rewrite retry first)
                                                       ▼
 shared stores (unified memory + SSD):           [Q7] 8B generate, span-tagged
   object store (PDFs, text, maps)                     context, citations required
   chunk DB (doc_id, offsets, section)                 │
   metadata DB + alias table                     [Q8] NLI groundedness per claim
   fp16 shards / int8 HNSW / small BM25                │ borderline band → 8B judge
   manifest + tombstones + eval DB               [Q9] GATE 2: min-of-triad
                                                       │ strip unsupported claims;
                                                       │ core lost → ABSTAIN
                                                       ▼
                                                 cited answer: every claim →
                                                 (doc_id, char_start, char_end)
                                                 → page/bbox for highlighting
```

### 2.2 Offline path

Unchanged in structure from v1; the load-bearing choices restated with verified support. Canonical text is frozen at extraction with a char→page/bbox map, so citations are stable spans for the document's lifetime (§6.2). MinHash/LSH clustering (I2) assigns cluster IDs and computes per-document diffs against a frozen representative — nothing is deduplicated, because the near-duplicate that differs in one quietly modified clause is the highest-value document in its cluster and precisely what Reuter's lexical-redundancy analysis (§2.2) says retrievers confuse. Chunking (I3) splits on contract structure first (numbered sections, ALL-CAPS captions, recital markers, signature blocks), then recursively to ≤500 characters at sentence boundaries, no overlap — Pipitone §4.3 showed structure-aware recursive splitting beating naive fixed-size at 714-document scale, and Reuter's whole result stack sits on RCTS-500-no-overlap.

I4 is the corpus's one LLM call per document and now emits the **generic-150 summary** (Reuter's winning configuration, §5.1, App. A) plus the metadata JSON — parties, effective date, type, governing law, term — that powers symbolic binding. The contrastive-delta summary variant is generated *only for probe documents* until and unless the probe promotes it (§9). I5 embeds `summary ⊕ chunk` to fp16 shards, derives int8, builds HNSW. I6 builds a small chunk-level BM25 index that only the conditional route ever queries. I7 commits the (doc_sha, stage, config_hash) manifest that makes everything resumable and every migration blue/green.

### 2.3 Online path

Q1 adds one new element: an **exact-term detector** — quoted phrases, §-number patterns, defined-term capitalization — whose flag is the only thing that can activate the lexical route (Q4′). This is the architectural translation of Reuter App. B: BM25 helps bind documents (their DRM improved) but hurts span selection (their precision/recall fell), so it runs only where exact-term matching is the point of the query, and its candidates are RRF-fused into the dense list and then densely rescored, never trusted for final ranking alone.

Everything else is as v1: hard metadata filtering only on high-confidence bindings (an alias-table hit, a parsed date), dense HNSW over int8, fp16 exact rescore of the top-200, Gate 1's score floor with zero-LLM-call abstention (now worth 10+ seconds per avoided generation on column A), span-tagged generation with mandatory chunk-ID citations, per-claim NLI entailment, and the min-of-triad Gate 2 that strips unsupported claims and abstains when the core is lost. The query taxonomy (lookup / clause QA / document discovery / filtered ≤50-doc map-reduce aggregation / unanswerable) is unchanged; note that map-reduce aggregation costs ~10–15s *per document* on column A, so the ≤50-doc cap is now also a latency guardrail, not just a cost one.

---

## 3. The seven scale problems, v2

### P1. Does SAC survive a 30× larger pool? *(updated)*

**Verdict unchanged in shape, sharpened in mechanism, and humbler about the fix.** Vanilla generic SAC will degrade gracefully across template clusters and collapse within them — Reuter's own residual DRM after SAC is still ~35–68% on ContractNLI across k (Fig. 2b) at pool 362, and intra-cluster confusability only grows with cluster size. What changed is the assessment of the summary-side fix. v1 bet on metadata-contrastive summaries; the actual paper contains that bet's nearest neighbor — the expert-guided, identifier-rich summary — and it *found the right document while retrieving useless boilerplate spans* (Ex. 1, cases C/D). Reuter's two explanations both cut against v1's proposal: identifier-dense cues can overfit narrow features, and dense structured text strains a 335M embedder that must compress summary and chunk into one vector (§5.3).

There remains a real difference to test: the contrastive-delta summary carries the **negotiated deviations** ("deviates from template T3: 3-yr term, residuals carve-out"), which are exactly what clause-level queries about near-duplicates target — content Reuter's expert prompt never included, because their corpus had no cluster structure to diff against. That mechanism story keeps the arm alive. But the burden of proof has flipped: contrastive-delta must now beat generic on document-hit **without losing span precision** (the new guard in §9's adoption rule), and the default if it fails is generic-150, which is the published winner.

The second half of the answer gains strength from a verified detail: **LegalBench-RAG queries state the document's identity in plain text** ("Consider the X agreement between A and B; …", Pipitone §3.1.3) — and retrieval still mismatched documents >95% of the time. The embedding channel cannot be trusted to bind identity even when identity is handed to it. So the primary near-duplicate defense in v2 is unambiguous: **symbolic metadata binding** (P5) for every query that names parties, dates, or types — which, in a contracts practice, is most of them — with SAC carrying only the residual unbound queries. The Pool-Scaling Probe (§9) still answers the extrapolation question before full ingestion, using cluster-complete sampling so intra-cluster confusability at pool 3k approximates pool 12k.

### P2. Summary length vs. pool size *(prior updated)*

Reuter App. A Table 1 resolves more than the brief conveyed: at chunk 500, the 150-char summary beat 300 on precision (11.03 vs 8.45%), recall (41.8 vs 37.8%) **and DRM (19.29 vs 20.61%)**. Longer summaries did not buy document-level discrimination even at their scale — the dilution mechanism (the summary consuming encoder capacity that the chunk needs) is supported, and the pool-dependent reading (more documents need more bits) has no support yet. The factorial {150, 300} × {generic, contrastive} × pool {300, 1k, 3k} stays in the probe because pool-dependence could still emerge at 12k, but the expectation is now firmly 150, and a 300-char arm that loses at pool 300 is dropped before pool 1k rather than carried.

### P3. Near-duplicate templates *(unchanged)*

Detect and diff; never dedup. MinHash/LSH clustering is minutes of CPU at 12k documents and an LSH lookup per incremental arrival. The clusters now serve three purposes: stratifying the probe and audits, feeding the contrastive-delta *probe arm*, and — regardless of the probe's outcome — feeding the answer UI ("this NDA follows template T3 with 2 deviations"), which is valuable even if the summary experiment fails. Cost over doing nothing: ~15 minutes of cold start plus slightly longer probe-arm prompts.

### P4. Flat vs. hierarchical retrieval *(decision unchanged, easier)*

Flat wins, and the unified-memory envelope removes the only argument two-stage ever had. int8 vectors + HNSW graph cost ~3GB at 12k and ~12.5GB at 50k — trivially inside 64GB alongside everything else (§5). Two-stage's savings are therefore worthless here, while its cost is unchanged: a recall ceiling P(correct doc in top-D) sitting exactly on the failure mode Reuter measured at >95%. The 12k-vector document index (embeddings of the summaries, ~50MB) is retained as an auxiliary for explicit document-discovery queries only, never as a gate.

### P5. Metadata pre-filtering *(strengthened)*

Still nearly free — extraction rides the I4 call — and now the load-bearing near-duplicate defense per P1's re-weighting, with R9 as its sharpest justification: if >95% DRM survives queries that *contain the document description*, then identity must be bound symbolically, not semantically. Policy unchanged: hard filter only on high-confidence extractions (alias-table exact hit, cleanly parsed date), soft boost otherwise, so a mis-OCR'd party name can cost an optimization but never exclude the right document. Extraction precision is audited on the probe sample with the ≥97%-or-soft-boost-only rule; entity normalization uses fuzzy matching plus a curated head-entity alias table inside the audit budget.

### P6. Cold start and incremental ingestion *(recomputed)*

Mechanics unchanged (§8); the clock is new. Headline: **~1.5–2.5 days on column A, ~0.5–1.25 days on column B**, phase-serial with document pipelining, OCR overlapped on CPU. Monthly increments are a ~40–75-minute GPU batch on A (nightly slices of ~10 docs run in ~2–3 minutes). Full stage table in §7.2.

### P7. Re-index cost as a design constraint *(now schedule-critical)*

The artifact separation from v1 (text/offsets/chunks/summaries/metadata durable; embeddings derived) carries over intact and matters more, because the clock got slower:

| Re-index class | @12k, col. A | @12k, col. B | @50k, col. A |
|---|---|---|---|
| Embedder swap (re-embed + index) | **~11–21h (≈1 day)** | ~5–10h | ~2–3.5 days |
| Summarizer / summary-prompt change | ~1–2 days | ~0.5–1 day | **~4–8 days** |
| Chunker change (+ downstream) | ~1.5–2.5 days | ~0.7–1.25 days | ~6–10 days |
| Full from-PDF (extractor change) | ~1.5–2.5 days | ~0.75–1.25 days | ~6–10 days |

At 50k documents on the 64GB mini, a re-summarization event is **a week of the machine**. That is why the probe (§9) exists before ingestion, why its decision rules are pre-registered, and why the summarizer prompt is the most change-controlled artifact in the system. All migrations remain blue/green under config hashes; the eval suite is the cutover gate.

---

## 4. The six lite-vs-performant tensions, v2

**T1. Summarizer size.** *Choice: 8B-class 4-bit, same model as the generator — but the 3B stake is now real money.* On the 4090, 3B saved ~an hour; on column A the summarization pass is the dominant cold-start cost (~12–24h), so 3B saves roughly **half a day once and halves the worst re-index class** (P7). The choice stays 8B because summary quality is load-bearing for DRM and 8B-class models are markedly more reliable at the structured-JSON extraction riding the same call — but the probe's 3B arm is no longer a formality, and its pre-registered rule (admitted iff within 1pt of 8B on DocHit@8 and DRM@8) now guards a meaningful prize. One free optimization regardless: capping the summarizer input at ~2.5k tokens (first two pages + section headings + cluster delta) instead of 3.5k cuts the prefill bill ~30% with no plausible quality cost, since Reuter generated competitive summaries from far less curated input with gpt-4o-mini (§4).

**T2. Embedder size.** *Choice: keep gte-large (335M) — and v1's pushback on the brief gets partially retracted.* On this hardware the brief's claim that smaller embedders save "hours of ingestion" is simply true: the embed pass is ~9–19h on column A at 335M and would be ~3–6h at 110M. What keeps the decision is quality logic that the papers now state outright: gte-large was the best open-source embedder in Reuter's ablation (App. C, Fig. 4, clearly above bge-base and far above legal-bert), retrieval is the measured bottleneck everywhere (Pipitone throughout; Wahidur Fig. 8), and Reuter's own hypothesis for the expert-summary failure is **embedder capacity** (§5.3) — an argument for more capacity, not less, and specifically for testing any information-dense summary variant only in combination with the larger encoder. *Cost of the choice: ~10h of one-off ingestion and ~2× on embedder-class re-index events, priced in P7.* *Measurement:* probe head-to-head vs. gte-base with the ≤1pt rule; gte-base is ladder rung 3.

**T3. Quantization.** *Choice unchanged: int8 in HNSW + fp16 mmap exact rescore; binary demoted further.* Memory ceased to be the binding resource (§0.2), so binary's ~2GB saving buys nothing on this envelope, while its recall risk concentrates exactly on near-duplicate margins. Binary survives only as the floor-config rung (32GB machines) with its price measured in advance by the probe's quantization cells.

**T4. Verification cost.** *Choice unchanged: NLI cascade (DeBERTa-v3-large-MNLI class on Metal), 8B judge on a calibrated ~15% borderline band; triad gate per Wahidur.* The verified numbers behind the design: naive RAG's closed-domain profile of AR 0.88 / CR 0.31 / G 0.26 (Table 12) is the fluent-failure signature the gate exists to catch, and LQ-RAG's alternative — regenerate on failure — doubled latency (Fig. 9) while leaving closed-domain groundedness at 0.35, which is why v2 abstains instead of retrying generation. New on this hardware: a judge escalation costs +9–14s on column A, so the κ calibration (delegation stands iff κ ≥0.9 against the 8B judge on probe answers) is now a latency-budget item, not just a quality one; if κ falls short, the band widens and p95 latency visibly pays — an honest price shown on the dashboard rather than a silent gate weakening.

**T5. Query rewriting.** *Choice unchanged and reinforced: escalation-only.* The verified record is less favorable to rewriting than the brief implied: HyPA's rewrite gains came bundled with adaptive k on a 15-page corpus using 3–7 rewrites per query, and in their own ablation adding Q to the KG configuration *lowered* correctness (Table 3, k,K,S 0.8030 vs k,K,S,Q 0.7918). On column A a default rewrite adds ~3–5s to every query for a benefit their evidence doesn't establish at scale. It remains as the one retry before abstention on the weak-retrieval path, A/B-measured on the paraphrase slice.

**T6. Classifier.** *Choice unchanged and strengthened: TF-IDF+SVM.* Kalra Table 8 shows the binary SVM tying the fine-tuned DistilBERT at 0.92 F1, and this router's live decisions are binary-ish (simple vs. complex; exact-term flag is rule-based, not learned). ~50MB, microseconds, trained on the audit's ~300 labeled queries. DistilBERT is reconsidered only if the routing-error slice becomes a top-3 failure mode in ongoing eval.

---

## 5. Model inventory and unified-memory budget (Deliverable 2)

### 5.1 Inventory

| Model | Role | Params | Quant | Resident | Runtime | Loaded |
|---|---|---|---|---|---|---|
| 8B-class instruct (Qwen2.5-7B / Llama-3.1-8B) | Generator + summarizer + borderline judge | ~8B | 4-bit (GGUF Q4_K_M / MLX 4-bit) | ~7GB incl. KV + Metal buffers | llama.cpp / mlx-lm, Metal | always |
| gte-large-en-v1.5 | Embedder | 335M | fp16 | ~1.0GB | MLX / Core ML | always |
| DeBERTa-v3-large-MNLI | Groundedness NLI | 435M | fp16 | ~1.2GB | MPS / Core ML | always |
| TF-IDF + SVM | Router | ~0 | — | ~50MB | CPU | always |
| Alias matcher + date rules + exact-term detector | Query binder | — | — | ~50MB | CPU | always |
| Tesseract | OCR fallback | — | — | CPU workers | CPU | ingest |
| MinHash/LSH | Clustering | — | — | ~0.5GB transient | CPU | ingest |

### 5.2 Unified memory budget

| Component | Col. A steady @12k | @50k | Col. B options |
|---|---|---|---|
| Generator (8B Q4 + KV + buffers) | ~7GB | ~7GB | 14B Q4 ≈ +9GB, or 32B Q4 ≈ +19GB |
| Embedder + NLI | ~2.2GB | ~2.2GB | same |
| HNSW int8 + graph | ~3.0GB | ~12.5GB | same |
| Metadata/chunk DBs, router, manifests | ~1.5GB | ~2.5GB | same |
| macOS + services | ~8GB | ~8GB | same |
| **Resident total** | **~22GB** | **~32GB** | ~31–41GB with 14B |
| Page-cache headroom (fp16 shards, BM25, working set) | ~42GB — 5GB of shards fully hot | ~32GB — 20GB of shards mostly hot | vast |

**Verdict: 64GB clears both scales with real headroom; nothing in the memory column forces a cut even at 50k documents.** Peak ingestion is phase-serial and stays at or below steady-state residency. Column B's extra memory buys model-size options, not survival: a 14B Q4 generator fits column A's memory too, but its ~17–26 tok/s decode there means 14–22s answers — so 14B is recommended only on column B (~35–55 tok/s → 7–10s), gated on the answer-quality audit; 32B is possible on B at 12–18s if that audit ever demands it.

---

## 6. Storage design (Deliverable 3)

Sizes are corpus-determined and unchanged from v1: **~45GB at 12k documents, ~180GB at 50k** (originals ~25/100GB, canonical text 1.2/5GB, offset+bbox maps ~3/12GB, chunk table ~2/8GB, summaries+metadata ~0.1/0.4GB, fp16 shards 5/20GB, int8+HNSW ~3/13GB, small conditional BM25 ~2/8GB, manifests/eval ~1/3GB). One hardware note replaces v1's: Mac mini base storage is small — specify **≥2TB internal**, or an external Thunderbolt 5 NVMe enclosure (~3GB/s, ample for mmap'd shards), and leave room for blue/green duplication of all derived artifacts during migrations.

**Character offsets end-to-end** are unchanged and restated as the invariant: canonical text is immutable from extraction; chunks store canonical spans; summaries live beside chunks and enter only embedder input, never canonical text; generation cites chunk IDs which resolve to (doc_id, char_start, char_end), merging adjacent chunks into multi-span citations; the NLI gate verifies each claim against exactly its cited span, so a drifted citation fails groundedness rather than shipping; the UI maps spans to page+bbox via the I1 map. Any extraction change is a new config hash and a full-class re-index, never an in-place edit.

---

## 7. Compute budget (Deliverable 4)

### 7.1 Per-query cost

Typical answerable clause-QA query, k=8, A8′ assumptions:

| Stage | Work | Col. A p50 | Col. B p50 | LLM calls |
|---|---|---|---|---|
| Normalize + route + exact-term detect | CPU | <5ms | <5ms | 0 |
| Metadata filter | SQLite | 1–5ms | 1–5ms | 0 |
| Dense HNSW int8 (+ conditional BM25 on flagged queries) | top-200 (+50) | 3–20ms | 3–20ms | 0 |
| RRF (if fired) + fp16 rescore | 200 exact dots, mmap | ~3ms | ~3ms | 0 |
| Gate 1 | threshold | ~0 — **abstain path total <100ms** | ~0 | 0 |
| Generation | ~2.0k in / ~350 out, 8B Q4 | 1.7–4s prefill + 8–12s decode | 0.8–2s + 4.5–6.5s | 1 |
| NLI groundedness | 6–12 claim–span pairs | 0.1–0.7s | 0.05–0.4s | 0 |
| Gate 2 / borderline judge | ~15% of answers | +9–14s when fired | +5–7s | 0.15 |
| **Totals** | | **p50 ≈ 10–16s** | **p50 ≈ 6–9s** | **E[calls] ≈ 1.05–1.35** |

The pre-generation pipeline still costs ~30ms; on this hardware that asymmetry is the design's biggest latency lever — every correctly-abstained or metadata-resolved query saves 10+ seconds, and the weak-path rewrite retry (+~3–5s) stays confined to the ~10–20% of queries already heading toward abstention.

### 7.2 Ingestion cost

| Stage | Per doc | Cold 12k, col. A | Cold 12k, col. B | Monthly 300, col. A |
|---|---|---|---|---|
| Parse, native (80%) | ~1–2s CPU | ~25 min (12 workers) | same | seconds |
| OCR (20%, A1) | ~40–90s parallel | ~2.5–3.5h (overlaps GPU) | same | minutes |
| Cluster + chunk | ~0.3s | ~30 min | same | seconds |
| Summarize + metadata (A7, tightened ~2.5–3.5k in / 300 out) | ~4–8s amortized | 42M in + 3.6M out ⇒ **~12–24h** | ~5–10h | ~40–60 min |
| Embed (~200 chunks × 170 tok) | ~0.6–1.2s amortized | 408M tok ⇒ **~9–19h** | ~4–8h | ~15–30 min |
| Quantize + HNSW + small BM25 | — | ~1.5–2h CPU | same | delta-insert |
| **Total wall-clock** | | **≈1.5–2.5 days** (GPU-serial 21–43h, CPU overlapped) | **≈0.5–1.25 days** | **≈1h GPU/month; nightly ~2–3 min** |

All figures inherit A8′'s ±2–3×; B0 (§8) replaces them with measurements before any schedule is committed. The probe (§9) adds ≈2–3 GPU-days once on column A (≈1–1.5 on B), before the cold start.

### 7.3 Re-index classes

Given as the P7 table (§3) — on this envelope that table *is* the argument for pre-registered, decide-once experiments.

---

## 8. Ingestion plan (Deliverable 5)

**B0 — microbenchmark (mandatory first step).** Measure all six A8′ numbers plus OCR pages/sec on 50 real documents spanning the quality range, on the actual machine and serving stack (llama.cpp vs MLX can differ 1.5–2× on Apple silicon — pick the winner per model here). Every §7 figure is then re-derived.

**B0.5 — external anchor (new in v2).** Before touching the private corpus, run **LegalBench-RAG-mini** (public: 776 queries over 72 documents, Pipitone §3.3.2) through the exact local stack — gte-large, RCTS-500, dense-only — and compare precision/recall@k against Pipitone's published Tables 4–5. This validates the retrieval code, the embedder integration, and the metric implementations against numbers that exist independently of this project, for a few hours of column-A compute. A harness that can't reproduce the published baseline within noise has a bug that would otherwise contaminate every probe decision.

**Cold start.** Phase-serial (parse/OCR → cluster/chunk → summarize → embed → index) with document-level pipelining: the GPU starts summarizing clean-PDF documents within minutes while OCR grinds the scanned tail on CPU cores. Batching: continuous batches of 16–32 for summarization (batched decode is where column A's aggregate throughput lives, per A8′-3), 256 for embedding. Phase boundaries are natural checkpoints; only one GPU model is resident per phase.

**Checkpointing, idempotency, failure at 70%.** Unchanged from v1 and restated briefly: identity = SHA-256 of PDF bytes; every artifact keyed (doc_sha, stage, config_hash), write-once; embedding shards sealed in 1,024-document units with checksums committed atomically to the manifest (SQLite WAL). A crash costs at most one shard's embedding work (~10–20 min of column-A GPU); restart is a manifest scan — completed (doc, stage, config) pairs are no-ops by construction, the bad shard rebuilds from checkpointed chunks+summaries, stragglers resume at their recorded stage, and indexes are built only from sealed shards so a crash can never leave a half-consistent index.

**Incremental.** Nightly watch-folder job: parse → LSH cluster-assign → summarize → embed → append to delta shard → insert into HNSW delta. Minutes on column A. Monthly: blue/green rebuild folding delta into base, tombstone compaction, full eval-suite regression as the cutover gate (~1.5–2h at 12k on A, growing toward ~6h at 50k — the §10 trigger for moving to segment merges).

**Corrections and withdrawals.** New bytes ⇒ new SHA ⇒ normal ingestion with a supersedes-link; old version tombstoned; tombstones filter results at query time (O(k)) and purge physically at the next rebuild; supersedes chains retained as audit trail; cluster representatives frozen at creation so member corrections never cascade.

---

## 9. Evaluation harness (Deliverable 6)

Gold labels **by construction**, human labor spent **auditing the construction**, confidence intervals stated — unchanged principles, with three v2 upgrades: an external anchor (B0.5, above), a calibration arm that ties the probe to published results, and a span guard on the headline adoption rule.

### 9.1 The Pool-Scaling DRM Probe, v2

**Sampling:** cluster-complete stratified, ~3,000 documents — at least 5 large template clusters taken whole (intra-cluster confusability at pool 3k ≈ pool 12k, the P1 argument that licenses pre-ingestion measurement), plus strata over type, scan quality, and singletons.

**Queries:** 3–5 per document generated by the 8B model *from specific spans*, span located by string-match back into canonical text so the gold (doc, span) is known mechanically — the same construction LegalBench-RAG used (Pipitone §3.1.2), with a paraphrase pass and reported query↔span lexical-overlap statistics as the validity check. An **unanswerable slice** (~20%) by mutation — nonexistent parties, clause types verifiably absent — because Wahidur Table 12's 0.88-relevant/0.26-grounded profile is the failure this slice calibrates the gates against.

**Arms.** Pool 300, full grid: summaries {none, generic-150, generic-300, **contrastive-delta-150**, **expert-replica-150**} × summarizer {8B, 3B}. Pools 1k and 3k: the ≤3 surviving summary arms × embedder {gte-large, gte-base} × vectors {fp16, int8-rescored, binary}. The **expert-replica arm** runs Reuter's own App. D expert prompt verbatim as a *calibration cell*: the expected signature is document-hit ≈ generic with worse span precision (their Ex. 1). If the harness fails to reproduce that published qualitative result at pool 300, the harness is audited before any other cell is believed.

**Pre-registered decision rules.** Contrastive-delta is adopted **iff** ΔDocHit@8 ≥ +3pts over generic-150 at pool 3k with CI excluding zero **and** span-F1 within 1pt of generic-150 — the span guard is new, and it is Reuter Ex. 1 written into the protocol. 3B summarizer admitted iff within 1pt of 8B on DocHit@8 and DRM@8. gte-base admitted iff within 1pt on all headline metrics. int8 stands iff within 0.5pt of fp16. Defaults on any inconclusive cell: generic-150, 8B, gte-large, int8 — the published-evidence configuration.

**Budget:** ≈2–3 GPU-days on column A (≈1–1.5 on B): ~3k summary calls + ~600k embeds at pool 300; ~9k calls + ~3.6M embeds across arms at pools 1k/3k; ~10k query generations; retrieval eval is CPU-cheap. Human: ~1.5 person-days auditing ~200 queries and ~150 extraction records (binomial CI at n=150 is ±8% at 95%, stated wherever used), plus ~0.5 day for the 8B answer-quality read and ~300 complexity labels for the SVM — ≈2–3 person-days total.

**Confirmation checkpoint:** the winning arm's DRM is re-measured at pool 12k during full ingestion on the same queries; a trend break is caught before users see it, with blast radius = the summaries+embeddings re-index class (P7), not the corpus.

### 9.2 Triad calibration and ongoing measurement

NLI-vs-8B-judge calibration on ~500 claim–span pairs from probe answers; κ reported; borderline band sized to keep NLI-alone decisions in its high-agreement region (κ ≥0.9 to delegate; shortfall widens the band and shows up as p95 latency, per T4). Ongoing: uniform weekly sample (~20 live queries, unbiased triad/abstention estimates) plus an adversarial near-threshold pool (worst-case tracking), never mixed unweighted; dashboards on abstention rate, retrieval-score distribution, groundedness pass rate, judge-band occupancy; the monthly rebuild re-runs the full suite as a regression gate. The harness's chief validity threat remains stated in the open: span-reverse-engineered queries are lexically closer to sources than real questions will be — a caution LQ-RAG's synthetic-training-and-eval setup (Wahidur §IV, §VIII) illustrates from the other direction — and the weekly real-query audit progressively re-anchors every threshold.

---

## 10. Scaling headroom at 50k (Deliverable 7)

Memory does not break: ~32GB resident on column A at 50k, shards mostly page-cached, disk ~180GB. What degrades is **time**, in order: **(1) re-index event cost** — the P7 table's right column; a summarizer change at 50k on the mini is ~a week, which turns change control on the summary prompt from discipline into necessity, and makes column B (or a temporary second machine for a one-off migration) the pragmatic answer if such an event becomes unavoidable. **(2) The monthly rebuild** stretches toward ~6h — fix: segment-based merges with quarterly compaction. **(3) Cold-start-class events** generally: ~6–10 days on A — schedule as planned downtime or borrow bigger silicon. **(4) Evaluation representativeness** — new template families arrive unseen; fix: automatic probe enrollment triggered by cluster-census drift. **(5) Entity-table curation** grows superlinearly; periodic head-entity review stays in the audit budget. First *soft* failure overall remains (4): the measurements go stale before the system does, hence the automatic trigger.

---

## 11. Degradation ladder, v2 (Deliverable 8)

Reordered for this envelope: the pressure is latency and ingestion time, not memory, so time-buying rungs climb and byte-buying rungs sink. Each rung states its price and where that price is measured.

1. **Trim k 8→5 and cap answers at ~220 tokens.** p50 falls to ~7–11s on column A. Cost: recall on multi-clause questions and terser answers — measured on the probe's hard slice and the answer audit.
2. **8B → 3B-class generator+summarizer.** p50 ~5–8s; cold start and the worst re-index class roughly halve. Cost: the probe's 3B-arm DRM delta plus answer-nuance loss — only take this rung if the 3B arm passed its pre-registered gate.
3. **gte-large → gte-base embedder.** Embed pass ~3× faster; embedder-class re-index drops to ~4–7h at 12k. Cost: ≤1pt by rule, priced by the probe; requires one re-embed to enter.
4. **DeBERTa-large NLI → MiniLM-class NLI.** Cost: κ drops, judge band widens — the price is visible latency, never a silently weaker gate.
5. **Drop the conditional lexical route.** Cost: exact-term slice recall, priced by the probe ablation.
6. **int8 → binary vectors.** Relevant only on a 32GB floor config or ≥50k on small machines; price known in advance from the probe's binary cells.
7. **Flat → two-stage document-scoped retrieval.** Last resort — the P4 recall ceiling; re-measure DocHit@D on this corpus before ever taking it.
8. **Floor config: base M4 / 32GB** — rungs 2+3+4+6 together. It works, abstention discipline intact; every delta was measured on the way down.

---

## 12. Explicit non-goals, v2 (Deliverable 9)

| Excluded | Why | Verified evidence |
|---|---|---|
| Cross-encoder reranker | The papers **conflict**, and the setting decides it: Cohere rerank collapsed ContractNLI precision (16.45→6.63 P@1) and the aggregate while helping only PrivacyQA; bge-reranker-large's faithfulness gain came on a 15-page single-document corpus where document mismatch cannot occur. Our corpus is the ContractNLI-shaped one. RRF here is rank fusion, not a learned reranker | Pipitone §5.1, Tables 4/6/7; Kalra §8.4 Table 3 |
| Always-on lexical/hybrid retrieval | Improved DRM marginally, cost text-level precision and recall; retained only as the conditional exact-term route (L6) | Reuter App. B, Table 2 |
| Knowledge graph | −0.13 absolute correctness at constant threshold-correctness — on 15 pages, where construction is even feasible; at 12k documents construction cost alone is disqualifying | Kalra §8.3–8.4, Table 3 |
| Expert / identifier-dense summaries as default | Correct document, boilerplate span; narrow-cue overfitting and embedder-capacity strain | Reuter §5.1–5.3, Ex. 1, App. D |
| Embedding fine-tuning | **Staged, not rejected** — the +13%/+15% gain is real but relative, trained and evaluated on synthetic in-distribution pairs from legal *books*, and it re-arms the 2.4M-chunk lock-in (P7). Entry criteria: probe recall misses target after L2–L6 settle, **and** ≥1k real-query relevance labels exist. Then: hours of training + one embedder-class re-index (~1 day, col. A) | Wahidur §IV, §VI-A Fig. 2, §VIII |
| Regenerate-on-fail loops | Doubled latency (7.2s→14.6s) while closed-domain groundedness reached only 0.35; abstention with sources is cheaper and more honest | Wahidur Fig. 9, Table 12 |
| Semantic chunking | Underperformed simpler methods unless heavily tuned | Kalra §6, Fig. 3 |
| Domain-pretrained embedders | Near-flat retrieval curves; densely clustered similarities → poor discrimination | Reuter App. C Fig. 4; Wiratunga §6.1 Fig. 4, §7 |
| RAPTOR / late chunking / hierarchical summaries | Named by Reuter as heavier alternatives and future work, unbenchmarked on legal near-duplicates; violates measurable-over-clever for v1 of this system | Reuter §2.3, §5.4 |
| Doc-level-first retrieval on the critical path | Self-imposed ceiling on the measured dominant failure | Reuter §3.2 |
| Corpus-wide analytics beyond filtered ≤50-doc map-reduce | Unbounded LLM cost; now also ~10–15s/doc on col. A | — |
| Cloud anything; query-time network; UI/auth/multi-tenant | Hard constraint / out of scope; §6.2's span contract is the UI interface | brief |

---

## 13. Open uncertainties register, v2

1. **SAC extrapolation from probe scale to 12k** — still #1. The cluster-completeness argument is an argument, not a measurement. *Resolver:* §9.1 trend fit + mid-ingestion confirmation; blast radius if wrong = summaries+embeddings re-index (~1–2 days, col. A).
2. **Contrastive-delta vs. generic** — now carries adverse nearest-neighbor evidence (Reuter's expert-SAC span failure). The delta-content mechanism keeps it testable; the span guard keeps it honest; the default if inconclusive is the published winner, generic-150. *Resolver:* probe arms C/E with the calibration cell.
3. **NLI groundedness on legal prose** — definitional cross-references may defeat sentence-level entailment. *Resolver:* κ calibration; failure mode is a wider judge band (visible latency), never a weaker gate; κ <0.75 even widened ⇒ revert to full LLM judging and re-issue §7.1.
4. **Synthetic-query distribution shift** — all pre-launch numbers ride span-reverse-engineered queries. *Resolver:* overlap statistics now, weekly real-query audit re-anchoring thresholds later; LQ-RAG's synthetic-eval circularity is the cautionary citation.
5. **Metadata extraction precision on scans** — the one path by which this design could exclude a correct document. *Resolver:* precision audit with the ≥97%-or-soft-boost rule; the failure mode is a lost optimization, not a lost document.
6. **Apple-silicon throughput (A8′) at ±2–3×** — wider than v1's CUDA band; every schedule in §7–§8 is provisional until B0, and B0.5 additionally anchors retrieval *quality* to published numbers before the corpus is touched.

---

*End of design v2. Next steps, in order: B0 (throughput microbenchmark on the actual machine), B0.5 (reproduce LegalBench-RAG-mini locally against Pipitone's published tables), then the §9.1 probe — which now settles L2 with a span guard, prices rungs 2/3/6 of the ladder in advance, and retires uncertainties 1, 2, 3, and 5 before a single production embedding is written.*