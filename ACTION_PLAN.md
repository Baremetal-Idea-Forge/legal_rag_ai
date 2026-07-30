# Legal RAG System — Action Plan

A phased build plan for the two-plane legal RAG architecture. Each phase is scoped to fit
one or two Claude Code sessions and ends with a measurable acceptance gate.

**Read this first:** the source papers disagree with each other on several points (rerankers
help / rerankers hurt; domain-pretrained embeddings help / hurt; KG helps correctness /
hurts answer quality). Every one of those disagreements is corpus-dependent. That is why
Phase 0 builds the measurement harness before anything else exists. Do not skip it.

---

## Decisions already locked by the research

Put these in `CLAUDE.md` so they don't get re-litigated mid-session.

| Decision | Value | Source |
|---|---|---|
| Chunk size | 500 chars, recursive character split, no overlap | Reuter et al. §A; Pipitone & Alami §4.2 |
| Document summary length | ~150 chars, **generic** prompt | Reuter et al. §5.1 — expert prompts retrieved the right doc but returned boilerplate |
| Embedding model | `text-embedding-3-large` (hosted) or `thenlper/gte-large` (local) | Reuter et al. Appendix C |
| Embedding models to avoid | `nlpaueb/legal-bert-base-uncased` | Near-zero precision in the same ablation |
| Reranker | **Off by default**, feature-flagged | Cohere rerank degraded every dataset in Pipitone & Alami §5.1 |
| Sparse retrieval | Dense-only in stage 2; BM25 optional in stage 1 | BM25 improved DRM but cut span precision (Reuter et al. Appendix B) |
| Knowledge graph | Off by default; opt-in for statute corpora only | KG cut absolute correctness by 0.13 in Kalra et al. §8.4 |
| Verification gate | Minimum of the triad, not the mean | Wahidur et al. Table 12 — 88% answer relevance with 26% groundedness |
| Failure mode | Explicit abstention, never a low-groundedness answer | Hindi et al. §VII-B |

### Anti-goals

- Do **not** add a reranker, a KG, or query rewriting until the benchmark shows the simpler
  configuration is the bottleneck. Every one of these has a documented negative result.
- Do **not** tune on the full LegalBench-RAG set. Use `-mini` (776 queries) for iteration and
  hold the full set out for release gates.
- Do **not** store `summary + chunk` as display text. The summary is a retrieval artifact only;
  leaking it into the generation prompt as if it were source text breaks citation integrity.
- Do **not** build the UI before Phase 4. The citation contract has to stabilize first.

---

## Repository layout

```
legal-rag/
├── CLAUDE.md                    # conventions + locked decisions above
├── pyproject.toml               # uv-managed
├── docker-compose.yml           # postgres+pgvector, qdrant, opensearch (phase 5)
├── src/legalrag/
│   ├── config.py                # pydantic-settings, all tunables in one place
│   ├── types.py                 # Document, Chunk, Span, QueryPlan, RetrievalResult
│   ├── ingest/
│   │   ├── parse.py             # → normalized text + offset map
│   │   ├── chunk.py             # strategy registry: recursive | delimiter | sentence
│   │   ├── summarize.py         # 1 LLM call per doc, 150 chars, retry on overrun
│   │   └── index.py             # fan-out writer, idempotent on (doc_hash, pipeline_ver)
│   ├── retrieve/
│   │   ├── dense.py
│   │   ├── sparse.py
│   │   ├── fusion.py            # reciprocal rank fusion
│   │   └── two_stage.py         # doc scoping → span selection
│   ├── plan/
│   │   ├── classifier.py        # DistilBERT complexity classifier
│   │   └── policy.py            # class → QueryPlan(k, Q, flags, budget)
│   ├── generate/
│   │   ├── prompt.py
│   │   ├── cite.py              # bind claims → (doc_id, start, end)
│   │   └── verify.py            # context relevance / groundedness / answer relevance
│   └── api/                     # FastAPI, phase 5
├── eval/
│   ├── datasets/                # legalbench-rag-mini loader
│   ├── metrics/
│   │   ├── drm.py               # document-level retrieval mismatch
│   │   ├── span.py              # char-level precision / recall @ k
│   │   └── triad.py             # RAGAS wrappers
│   ├── run_benchmark.py         # CLI: --config, --split, --k-values
│   └── baselines.json           # frozen expected ranges, CI gate
├── tests/
└── scripts/
```

---

## Phase 0 — Measurement harness

**Goal:** be able to score any retrieval configuration before writing a retriever.

### Tasks

1. Scaffold the repo with `uv`, `ruff`, `pytest`, `pydantic-settings`. Python 3.12.
2. Write `src/legalrag/types.py` first. `Span` must carry `(document_id, start_char, end_char, text)`.
   Everything downstream depends on this contract — get it right before anything consumes it.
3. Build the LegalBench-RAG-mini loader in `eval/datasets/`. Download from the public repo,
   verify 776 queries across ContractNLI / MAUD / CUAD / PrivacyQA (194 each), and assert the
   corpus character counts match the paper's Table 3.
4. Implement `eval/metrics/drm.py`: DRM = fraction of top-k retrieved chunks whose parent
   document is not the ground-truth document. Report per-dataset and weighted-average.
5. Implement `eval/metrics/span.py`: character-level precision and recall against ground-truth
   spans, at k ∈ {1, 2, 4, 8, 16, 32, 64}.
6. Implement `run_benchmark.py` as a CLI taking a config object and emitting a JSON report.
   Weight each dataset equally regardless of query count, matching the paper's methodology.
7. Write a `NullRetriever` that returns random chunks, and assert the harness produces
   near-zero precision on it. This is the harness's own unit test.

### Acceptance

- `uv run python -m eval.run_benchmark --retriever null` completes and reports DRM ≈ 100%,
  precision ≈ 0.
- Metric functions have unit tests with hand-computed expected values.
- CI runs the harness on every push.

### Session prompt

> Read `CLAUDE.md`. Scaffold the repo per the layout in the plan and implement Phase 0 only:
> types, the LegalBench-RAG-mini loader, the DRM and span-PRF metrics, and the benchmark CLI.
> Verify the harness against a null retriever. Do not implement any real retrieval yet.

---

## Phase 1 — Baseline pipeline

**Goal:** reproduce the paper's naive numbers, and freeze them as a regression floor.

### Tasks

1. `ingest/parse.py` — read the corpus `.txt` files, normalize to UTF-8, and build the offset
   map. For LegalBench-RAG the source is already plain text, so offsets are identity; keep the
   abstraction anyway because real PDFs will not be.
2. `ingest/chunk.py` — recursive character splitter at 500 chars, no overlap. Register it under
   a strategy name so `delimiter` and `sentence` strategies can slot in later. Each chunk
   carries its parent document id and its `(start, end)` offsets.
3. `ingest/index.py` — embed chunks with the configured model, write to the vector store.
   Start with pgvector for simplicity; the interface should not leak the backend.
4. `retrieve/dense.py` — single-stage cosine top-k. Nothing clever.
5. Run the full benchmark and record results in `eval/baselines.json`.

### Acceptance

Reproduce the paper's naive/RCTS ranges within a reasonable margin. Expected shape:

| Metric | Expected range |
|---|---|
| Weighted avg precision @ k=1 | 4–7% |
| Weighted avg recall @ k=64 | 60–78% |
| DRM @ k=1, weighted | 40–50% |
| DRM @ k=64, weighted | 75–85% |
| ContractNLI DRM @ k=8 | > 85% (this is the pathological case) |

If ContractNLI DRM is not catastrophically high, something is wrong with the loader —
that failure is the whole motivation for Phase 2 and it should reproduce.

### Session prompt

> Implement Phase 1: parse, recursive chunking at 500 chars, dense embedding + indexing, and
> single-stage cosine retrieval. Run the benchmark and write results to `eval/baselines.json`.
> Report whether ContractNLI DRM exceeds 85% at k=8 — it should, and that confirms the loader
> is correct.

---

## Phase 2 — SAC and two-stage retrieval

**Goal:** cut DRM roughly in half. This is the highest-value phase in the entire plan.

### Tasks

1. `ingest/summarize.py` — one LLM call per document producing a ≤150-char generic summary.
   Allow a 20-char tolerance; regenerate with a reduced target on overrun. Cache by document
   hash so re-ingestion is free.
2. Extend `Chunk` with two distinct fields: `retrieval_text` (`summary + "\n" + body`) and
   `display_text` (body only). Embed the former, cite the latter. Add a test asserting they
   are never conflated.
3. Re-index the corpus with summary-augmented chunks.
4. `retrieve/two_stage.py`:
   - **Stage 1:** retrieve over SAC chunks, aggregate scores per parent document (max or
     sum-of-top-3, benchmark both), select top-N documents.
   - **Stage 2:** retrieve spans restricted to the scoped document set.
   - Expose `N` as a config value; sweep it.
5. Add an abstention path: if the stage-1 top document score falls below a threshold, return
   `NoAnswer` without proceeding.
6. Sweep chunk size ∈ {200, 500, 800} × summary length ∈ {150, 300} and confirm the paper's
   optimum on your corpus.

### Acceptance

| Metric | Phase 1 baseline | Phase 2 target |
|---|---|---|
| Weighted avg DRM (mean over k) | ~55–60% | **< 25%** |
| Precision @ k=1 | ~0.05 | **> 0.15** |
| Recall @ k=64 | ~0.31 | **> 0.50** |
| ContractNLI DRM @ k=8 | > 85% | **< 45%** |

Also expected: the 500-char × 150-char configuration wins on the precision/recall balance.
If it doesn't on your corpus, record the actual winner in `CLAUDE.md` and move on — this is
exactly the corpus-dependence the harness exists to detect.

### Session prompt

> Implement Phase 2: summary-augmented chunking and two-stage document-scoped retrieval.
> Keep `retrieval_text` and `display_text` strictly separate. Re-index, then run the chunk-size
> × summary-length sweep and report the full grid. Compare against `eval/baselines.json` and
> state whether the DRM target was met.

---

## Phase 3 — Adaptive query planner

**Goal:** spend retrieval budget proportional to query difficulty.

### Tasks

1. Generate a query-complexity training set. Label by number of distinct source documents or
   spans required. Apply the augmentation recipe from Kalra et al. §7.1 — vagueness injection,
   noise words, phrase reordering, verbosity on simple queries, compression on complex ones.
   Roughly two thirds of examples should be perturbed.
2. Train two baselines before touching a transformer: TF-IDF + logistic regression and
   TF-IDF + linear SVM. In the paper these hit F1 0.84 and 0.86 against DistilBERT's 0.90.
   If the SVM is close enough, ship the SVM — it costs nothing to serve.
3. Fine-tune DistilBERT (lr 2e-5, batch 32, 10 epochs, weight decay 0.01). Export to ONNX.
4. `plan/policy.py` — map class → `QueryPlan`. Start with the paper's 3-class mapping
   (`k` = 3/5/7, `Q` = 3/5/7) but treat these as config, not constants.
5. Add a **confidence floor**: when classifier confidence is low, default to the highest-effort
   plan rather than guessing. A misrouted hard query is worse than a slightly expensive easy one.
6. Instrument token cost and latency per class and report them alongside quality metrics.

### Acceptance

- Classifier macro-F1 ≥ 0.85 on a held-out split.
- Adaptive routing matches or beats the best fixed-`k` configuration on span recall while
  reducing mean tokens per query by ≥ 20%.
- Classifier inference p99 < 15ms on CPU.
- If adaptive routing does **not** beat fixed-`k`, ship fixed-`k` and document why. The
  planner is an optimization, not a requirement.

### Session prompt

> Implement Phase 3: complexity classifier and query planner. Train the TF-IDF baselines first
> and only move to DistilBERT if they underperform. Add the low-confidence fallback to the
> highest-effort plan. Report macro-F1, plus tokens and latency per complexity class against
> the best fixed-k configuration.

---

## Phase 4 — Generation, citation, verification

**Goal:** produce grounded, span-cited answers — or abstain.

### Tasks

1. `generate/prompt.py` — assemble query + retrieved `display_text` spans with stable ids.
   Instruct the model to attach a span id to every factual claim.
2. `generate/cite.py` — parse claims and bind each to `(document_id, start_char, end_char)`.
   Any claim that cannot bind is flagged. Emit a `citation_coverage` ratio per answer.
3. `generate/verify.py` — score three metrics independently:
   - **Context relevance:** do the retrieved spans address the query?
   - **Groundedness:** is each claim supported by a retrieved span?
   - **Answer relevance:** does the answer address the question asked?
   Gate on `min(triad)`, not the mean.
4. Bounded feedback loop in LangGraph: on gate failure, rewrite the query and retry. Hard cap
   at 2 iterations. On exhaustion, abstain with the reason and the best-effort spans attached.
5. **Calibrate the judge.** Hand-label 100–150 answers on a 1–5 correctness scale, compute
   Spearman against the judge's scores, and iterate on the judge prompt until correlation
   clears 0.6. Store the labeled set as a regression fixture. An uncalibrated judge is a
   liability, not a metric.
6. Build the closed-domain adversarial set: 20 queries whose answers are provably absent from
   the corpus. The system must abstain on ≥ 90% of them.

### Acceptance

- Citation coverage ≥ 95% of factual claims on the benchmark.
- Abstention rate ≥ 90% on the closed-domain adversarial set.
- Judge–human Spearman ≥ 0.6 on the labeled fixture.
- p95 end-to-end latency budgeted and reported per complexity class.

### Session prompt

> Implement Phase 4: prompt assembly, claim-to-span citation binding, triad verification gated
> on the minimum score, and a 2-iteration bounded retry loop with explicit abstention. Then
> build the closed-domain adversarial set and report the abstention rate. Do not proceed to
> Phase 5 until judge calibration clears Spearman 0.6.

---

## Phase 5 — Productionization

**Goal:** make it operable, observable, and auditable.

### Tasks

1. FastAPI service exposing `POST /query` and `POST /documents`. Streaming responses.
2. Split ingestion into a durable Temporal workflow — parse, summarize, chunk, embed, index —
   with per-activity retries and replay on failure.
3. Move to the production store set if the benchmark shows pgvector is the constraint:
   Qdrant for vectors with payload filtering (which makes stage-1 document scoping native),
   OpenSearch for BM25, Postgres for offsets and audit.
4. Semantic cache in Redis keyed on normalized query + corpus version. Legal queries repeat
   heavily inside a single firm; this is the cheapest latency win available.
5. OpenTelemetry tracing across the retrieval loop; Langfuse or Phoenix for LLM spans.
6. Immutable audit log: query, plan, retrieved spans, answer, triad scores, and every model
   version. This is a compliance artifact, not telemetry — write it before responding, not after.
7. Human review queue for abstained and low-confidence answers, with corrections flowing back
   as labeled data for the classifier and any future reranker.
8. Blue/green index strategy so an embedding model change can be rolled back without downtime.

### Acceptance

- Full LegalBench-RAG (not mini) run as a release gate, results archived.
- Reindex of the full corpus completes and is resumable after a forced mid-run kill.
- Every response in the audit log resolves to source spans via the offset map.

---

## `CLAUDE.md` starter content

```markdown
# Legal RAG

## Commands
- `uv run pytest` — tests
- `uv run ruff check --fix && uv run ruff format` — lint
- `uv run python -m eval.run_benchmark --config configs/current.yaml` — benchmark
- `uv run python -m scripts.reindex --corpus legalbench-mini` — reindex

## Non-negotiables
- Never conflate `Chunk.retrieval_text` (summary + body, embedded) with
  `Chunk.display_text` (body only, cited). Tests enforce this.
- Every user-facing claim resolves to (document_id, start_char, end_char).
- Verification gates on min(context_relevance, groundedness, answer_relevance).
- Abstention is a valid, preferred outcome. Never emit a low-groundedness answer.
- No new retrieval component ships without a benchmark run showing it beats the
  current configuration on this corpus. Negative results go in docs/decisions/.

## Locked defaults (see docs/research-notes.md for sources)
chunk=500 chars, summary=150 chars generic prompt, embeddings=gte-large,
reranker=off, BM25=stage-1 only, KG=off.

## Style
type hints everywhere, pydantic models for all boundaries.
Retrieval backends behind protocols — no backend types in business logic.
```

---

## Working rhythm with Claude Code

- **One phase per session.** Start each with "read CLAUDE.md and docs/decisions/, then
  implement Phase N." Phases 2 and 4 may need two sessions each.
- **Benchmark before and after every change.** Paste the delta into the session so the model
  can see whether its change helped. The harness exists to make Claude Code's work falsifiable.
- **Record negative results.** When the reranker or the KG makes things worse — and the papers
  suggest at least one of them will — write it to `docs/decisions/` so the next session doesn't
  try it again.
- **Keep configs in version control.** Every benchmark JSON should name the config that produced
  it. Untraceable numbers are worse than no numbers.

## Suggested first message

> Read the action plan at `docs/action-plan.md`. Implement Phase 0 only: repo scaffold, core
> types, LegalBench-RAG-mini loader, DRM and span-PRF metrics, and the benchmark CLI. Validate
> the harness against a null retriever. Stop when Phase 0 acceptance criteria pass and summarize
> what you built.