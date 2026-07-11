"""
Dependency-injection providers.

FastAPI routers depend on these via `Depends(...)`. Repositories and services
are cached as process singletons; the embedding model inside EmbeddingService
still loads lazily on first real use.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from core.config import Settings, get_settings
from clients.gemini_client import GeminiClient
from clients.mcp_client import McpClient
from clients.ollama_client import OllamaClient
from helpers.pdf_helper import PDFHelper
from repositories.pdf_storage_repository import PdfStorageRepository
from repositories.typesense_repository import TypesenseRepository
from services.embedding_service import EmbeddingService
from services.ingestion_service import IngestionService
from services.rag_service import RagService
from services.search_service import SearchService


@lru_cache(maxsize=1)
def get_typesense_repository() -> TypesenseRepository:
    return TypesenseRepository()


@lru_cache(maxsize=1)
def get_pdf_storage_repository() -> PdfStorageRepository:
    settings = get_settings()
    helper = PDFHelper(
        storage_dir=settings.PDF_STORAGE_DIR,
        public_base_url=settings.PDF_PUBLIC_BASE_URL or None,
    )
    return PdfStorageRepository(helper)


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    # Model loads lazily on first embed call, not here. Wire the device/model
    # from Settings (.env) — the helper otherwise only reads os.environ, which
    # never sees .env-only values, so EMBEDDING_DEVICE would be ignored.
    settings = get_settings()

    def _factory():
        from models.embeddings_model import get_embedding_helper

        return get_embedding_helper(
            model_name=settings.EMBEDDING_MODEL_NAME,
            device=settings.EMBEDDING_DEVICE,
            normalize_embeddings=settings.EMBEDDING_NORMALIZE,
        )

    return EmbeddingService(helper_factory=_factory)


def get_ingestion_service(
    settings: Settings = Depends(get_settings),
    pdf_repo: PdfStorageRepository = Depends(get_pdf_storage_repository),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    typesense_repo: TypesenseRepository = Depends(get_typesense_repository),
) -> IngestionService:
    return IngestionService(
        pdf_repo=pdf_repo,
        embedding_service=embedding_service,
        typesense_repo=typesense_repo,
        settings=settings,
    )


def get_search_service(
    settings: Settings = Depends(get_settings),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    typesense_repo: TypesenseRepository = Depends(get_typesense_repository),
) -> SearchService:
    return SearchService(
        embedding_service=embedding_service,
        typesense_repo=typesense_repo,
        settings=settings,
    )


@lru_cache(maxsize=1)
def get_ollama_client() -> OllamaClient:
    settings = get_settings()
    return OllamaClient(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
        timeout_seconds=settings.OLLAMA_TIMEOUT_SECONDS,
        max_retries=settings.OLLAMA_MAX_RETRIES,
    )


@lru_cache(maxsize=1)
def get_gemini_client() -> GeminiClient:
    settings = get_settings()
    return GeminiClient(
        api_key=settings.GEMINI_API_KEY,
        model=settings.GEMINI_MODEL,
        timeout_seconds=settings.GEMINI_TIMEOUT_SECONDS,
        max_retries=settings.GEMINI_MAX_RETRIES,
    )


@lru_cache(maxsize=1)
def get_llm_client() -> GeminiClient | OllamaClient:
    settings = get_settings()
    if settings.LLM_PROVIDER == "gemini":
        return get_gemini_client()
    return get_ollama_client()


@lru_cache(maxsize=1)
def get_mcp_client() -> McpClient:
    settings = get_settings()
    return McpClient(base_url=settings.MCP_SERVER_URL)


def get_rag_service(
    settings: Settings = Depends(get_settings),
    search_service: SearchService = Depends(get_search_service),
    llm_client: GeminiClient | OllamaClient = Depends(get_llm_client),
    mcp_client: McpClient = Depends(get_mcp_client),
) -> RagService:
    return RagService(
        search_service=search_service,
        llm_client=llm_client,
        settings=settings,
        mcp_client=mcp_client,
    )
