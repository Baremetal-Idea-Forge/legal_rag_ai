"""
Central application configuration.

Single source of truth for every locked decision in ACTION_PLAN.md §0.
All tunables live here (pydantic-settings) and are overridable via `.env`
or environment variables — no hardcoded hosts, keys, or model names elsewhere.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # -- App identity -------------------------------------------------------
    APP_NAME: str = "Legal RAG AI"
    APP_VERSION: str = "0.2.0"
    DEBUG: bool = False

    # -- Deployment ------------------------------------------------------------
    DEPLOY_DOMAIN: str = "localhost"

    # -- Typesense (vector + keyword hybrid search) -------------------------
    TYPESENSE_HOST: str = "localhost"
    TYPESENSE_PORT: int = 8108
    TYPESENSE_PROTOCOL: str = "http"
    TYPESENSE_API_KEY: str = "xyz"
    TYPESENSE_CONNECTION_TIMEOUT: int = 5
    # Startup: retry ensuring the collection (handles Typesense still booting),
    # then abort the app if still unreachable rather than degrading silently.
    TYPESENSE_STARTUP_RETRIES: int = 5
    TYPESENSE_STARTUP_RETRY_DELAY: float = 2.0

    # Collections
    # TYPESENSE_TASKS_COLLECTION kept for backward compatibility
    TYPESENSE_TASKS_COLLECTION: str = "tasks"
    TYPESENSE_PDF_CHUNKS_COLLECTION: str = "pdf_chunks"

    # -- Embedding model (ingestion + queries) ------------------------------
    # NOTE: field names match the env vars read by models/embeddings_model.py
    # so a single .env drives both the config layer and the model wrapper.
    EMBEDDING_MODEL_NAME: str = "yuriyvnv/legal-bge-m3"
    EMBEDDING_DEVICE: str | None = None          # e.g. "cpu", "cuda"
    EMBEDDING_DIMENSION: int = 1024              # must equal schema num_dim
    EMBEDDING_NORMALIZE: bool = True

    # -- LLM provider switch ---------------------------------------------------
    LLM_PROVIDER: str = "gemini"  # "gemini" | "ollama"

    # Sampling temperature for grounded answers. 0.0 = deterministic: the same
    # context yields the same answer every time (fixes intermittent refusals).
    LLM_TEMPERATURE: float = 0.0

    # -- Gemini API ------------------------------------------------------------
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_TIMEOUT_SECONDS: int = 120
    GEMINI_MAX_RETRIES: int = 2

    # -- Ollama (local dev) ----------------------------------------------------
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma3:4b"
    OLLAMA_TIMEOUT_SECONDS: int = 120
    OLLAMA_MAX_RETRIES: int = 2

    # -- OCR (PyMuPDF + Tesseract) ---------------------------------------------
    OCR_ENABLED: bool = True
    OCR_MIN_TEXT_LENGTH: int = 50

    # -- Storage monitoring ----------------------------------------------------
    TYPESENSE_DATA_DIR: str = "typesense-data"
    DISK_QUOTA_GB: int = 80

    # -- MCP server (read-side retrieval tool provider) ---------------------
    MCP_SERVER_URL: str = "http://localhost:9000"
    # Backend orchestrates by default; flip to route retrieval through MCP (Step 3)
    RETRIEVAL_VIA_MCP: bool = False

    # -- RAG loop -----------------------------------------------------------
    RAG_TOP_K: int = 5
    RAG_MAX_CONTEXT_CHARS: int = 12000

    # -- Summary-augmented chunking (SAC) — ACTION_PLAN Phase 2 -------------
    # One LLM call per document produces a short GENERIC summary that is
    # prepended to each chunk *for embedding only* (`retrieval_text`). The
    # summary never enters `content`, which is what gets displayed and cited.
    # Expert/role-play summary prompts retrieved the right document but made the
    # model answer with boilerplate, so the prompt stays deliberately generic.
    SAC_ENABLED: bool = True
    SAC_SUMMARY_MAX_CHARS: int = 150
    # Overrun tolerance before a single reduced-target regeneration is attempted.
    SAC_SUMMARY_TOLERANCE_CHARS: int = 20
    # Summaries are cached by document sha256 so re-ingestion costs no tokens.
    SAC_SUMMARY_CACHE_PATH: str = "storage/summaries.json"

    # -- Two-stage retrieval — ACTION_PLAN Phase 2 --------------------------
    # Stage 1 retrieves over summary-augmented chunks and aggregates scores per
    # parent document; stage 2 re-retrieves spans restricted to the winning
    # documents. This is the lever on document-retrieval mismatch (DRM).
    # Off by default: this changes the read path on the very next query, and the
    # plan's own rule is that no retrieval component ships before a benchmark
    # run on the target corpus shows it wins. Flip it on after running
    # `python tests/eval/run_eval.py --two-stage` against your index.
    TWO_STAGE_ENABLED: bool = False
    TWO_STAGE_CANDIDATE_K: int = 50      # stage-1 chunk candidates to aggregate
    TWO_STAGE_DOC_COUNT: int = 3         # documents to scope stage 2 to
    TWO_STAGE_DOC_SCORE: str = "max"     # "max" | "sum_top3"

    # Stage-1 abstention floor: when the best document score falls below this,
    # return NoAnswer instead of retrieving spans. 0.0 disables the floor —
    # raise it only from a benchmark sweep on your own corpus.
    ABSTAIN_MIN_DOC_SCORE: float = 0.0

    # -- L6 conditional lexical routing — ACTION_PLAN v2 --------------------
    # Dense-only retrieval is the v2 default (Reuter App. B: BM25 binds
    # documents but costs span precision); the lexical (hybrid) channel fires
    # only when the exact-term detector flags the query (quoted phrase,
    # §/Section/Article number, defined term). Off = pre-v2 behavior (always
    # hybrid). Enable after `run_eval.py --routed` wins on your corpus.
    EXACT_TERM_ROUTING_ENABLED: bool = False

    # -- Gate 1: retrieval-score floor — ACTION_PLAN v2 ---------------------
    # When the best dense chunk score (1 - vector_distance) falls below this
    # floor, the service abstains BEFORE any generation call, attaching the
    # nearest documents. Keyword-only hits carry no comparable score and skip
    # the gate. 0.0 disables — raise it only from an eval sweep.
    GATE1_MIN_SCORE: float = 0.0
    # One rewrite-and-re-retrieve attempt before a Gate-1 abstention (L11) —
    # the only place query rewriting exists in v2. Costs one small LLM call
    # on the weak-retrieval path only.
    GATE1_REWRITE_RETRY: bool = True

    # -- Verification triad — ACTION_PLAN Phase 4 / v2 Gate 2 ---------------
    # Context relevance / groundedness / answer relevance, each scored 0..1 by
    # an LLM judge. The gate is min(triad), never the mean: a high answer
    # relevance masks a low groundedness when averaged. On failure the service
    # abstains with best-effort sources — regenerate loops are a v2 non-goal
    # (R8: they doubled latency without fixing groundedness).
    VERIFY_ENABLED: bool = False
    VERIFY_MIN_TRIAD_SCORE: float = 0.5

    # -- Audit log — ACTION_PLAN Phase 5 ------------------------------------
    # Compliance artifact, not telemetry: written BEFORE the response is
    # returned, append-only, one JSON object per line.
    AUDIT_LOG_ENABLED: bool = False
    AUDIT_LOG_PATH: str = "storage/audit/audit.jsonl"

    # -- PDF storage + chunking --------------------------------------------
    # NOTE: PDF_STORAGE_DIR / PDF_PUBLIC_BASE_URL match the env vars read by
    # helpers/pdf_helper.py.
    PDF_STORAGE_DIR: str = "storage/pdfs"
    PDF_PUBLIC_BASE_URL: str = ""
    # Chunks are split on Article/Section boundaries (one provision per chunk).
    # A smaller cap keeps each provision's embedding sharp; oversized provisions
    # sub-split with overlap. Re-ingest after changing these.
    CHUNK_MAX_CHARS: int = 1200
    CHUNK_OVERLAP_CHARS: int = 150

    # -- Uploads ------------------------------------------------------------
    MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024      # 50 MB
    ALLOWED_UPLOAD_CONTENT_TYPE: str = "application/pdf"

    # -- CORS (comma-separated; tighten off "*" per security standard) ------
    CORS_ALLOW_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # -- Auth (optional API key) --------------------------------------------
    # Empty = auth disabled (dev default). Set to require X-API-Key / Bearer
    # on write endpoints.
    API_AUTH_TOKEN: str = ""

    # -- Rate limiting (single-process seam; disabled by default) -----------
    RATE_LIMIT_ENABLED: bool = False
    RATE_LIMIT_PER_MINUTE: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    @property
    def cors_origins(self) -> list[str]:
        """CORS_ALLOW_ORIGINS parsed into a list of origins."""
        return [o.strip() for o in self.CORS_ALLOW_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
