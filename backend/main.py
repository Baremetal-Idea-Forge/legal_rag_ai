"""
FastAPI application entry point.

Wires logging, request-ID middleware, CORS, exception handlers, routers, and a
lifespan that ensures the Typesense collection exists at startup.
"""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core.config import get_settings
from core.dependencies import get_typesense_repository
from core.exceptions import register_exception_handlers
from core.logging import RequestIDMiddleware, configure_logging
from routers.chat import router as chat_router
from routers.health import router as health_router
from routers.ingest import router as ingest_router
from routers.search import router as search_router

configure_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)

    logger.info("Ensuring Typesense collection '%s' exists…",
                settings.TYPESENSE_PDF_CHUNKS_COLLECTION)
    try:
        get_typesense_repository().ensure_collection()
        logger.info("Typesense ready.")
    except Exception as exc:
        # Degrade gracefully — app still serves /health; readiness reports down.
        logger.warning("Typesense unavailable at startup: %s", exc)

    yield

    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Legal RAG AI — PDF ingestion and hybrid semantic search over legal documents.",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Request correlation + timing (outermost middleware).
    app.add_middleware(RequestIDMiddleware)

    # CORS — explicit origins from settings (no wildcard).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(search_router)
    app.include_router(chat_router)

    # Serve stored PDFs so citation `file_url`s (/pdfs/{pdf_id}/{file}) resolve.
    # check_dir=False → mount never fails if the dir isn't created yet.
    app.mount(
        "/pdfs",
        StaticFiles(directory=settings.PDF_STORAGE_DIR, check_dir=False),
        name="pdfs",
    )

    return app


app = create_app()

# ---------------------------------------------------------------------------
# Dev runner (invoked by the project-root main.py or directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
