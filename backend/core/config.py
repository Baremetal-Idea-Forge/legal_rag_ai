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
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # -- Typesense (vector + keyword hybrid search) -------------------------
    TYPESENSE_HOST: str = "localhost"
    TYPESENSE_PORT: int = 8108
    TYPESENSE_PROTOCOL: str = "http"
    TYPESENSE_API_KEY: str = "xyz"
    TYPESENSE_CONNECTION_TIMEOUT: int = 5

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

    # -- LLM (Gemma 3 4B via Ollama, OpenAI-compatible HTTP API) ------------
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma3:4b"
    OLLAMA_TIMEOUT_SECONDS: int = 120
    OLLAMA_MAX_RETRIES: int = 2

    # -- MCP server (read-side retrieval tool provider) ---------------------
    MCP_SERVER_URL: str = "http://localhost:9000"
    # Backend orchestrates by default; flip to route retrieval through MCP (Step 3)
    RETRIEVAL_VIA_MCP: bool = False

    # -- RAG loop -----------------------------------------------------------
    RAG_TOP_K: int = 5
    RAG_MAX_CONTEXT_CHARS: int = 12000

    # -- PDF storage + chunking --------------------------------------------
    # NOTE: PDF_STORAGE_DIR / PDF_PUBLIC_BASE_URL match the env vars read by
    # helpers/pdf_helper.py.
    PDF_STORAGE_DIR: str = "storage/pdfs"
    PDF_PUBLIC_BASE_URL: str = ""
    CHUNK_MAX_CHARS: int = 3500
    CHUNK_OVERLAP_CHARS: int = 400

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
