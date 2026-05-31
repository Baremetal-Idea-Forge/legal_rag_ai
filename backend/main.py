from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import get_settings
import helpers.typesense_helper as ts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)

    logger.info("Ensuring Typesense collection exists…")
    try:
        ts.ensure_tasks_collection()
        logger.info("Typesense ready.")
    except Exception as exc:
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
        description="Legal RAG AI — PDF ingestion and semantic search over legal documents.",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],      # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------------------------------------------------------------
    # Routers (add here as they are implemented)
    # ---------------------------------------------------------------------------
    # from routers.ingest import router as ingest_router
    # from routers.search import router as search_router
    # app.include_router(ingest_router, prefix="/ingest", tags=["Ingest"])
    # app.include_router(search_router, prefix="/search", tags=["Search"])

    @app.get("/health", tags=["Health"])
    def health():
        return {"status": "ok", "version": settings.APP_VERSION}

    return app


app = create_app()

# ---------------------------------------------------------------------------
# Dev runner (invoked by the project-root main.py or directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
