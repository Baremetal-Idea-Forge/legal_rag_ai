"""
Pydantic request/response DTOs for the public API.

These are the contract between the HTTP layer and clients. Internal dataclasses
(StoredPDF, PDFChunk, …) never leak past the service boundary — services map
them into these schemas.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Search text")
    top_k: int = Field(5, ge=1, le=50, description="Max number of chunks to return")
    mode: str = Field("hybrid", description="keyword | vector | hybrid")
    filter_by: str | None = Field(
        None, description="Typesense filter expression, e.g. 'pdf_id:=<id>'"
    )


class ChunkHit(BaseModel):
    id: str
    pdf_id: str
    pdf_name: str
    page_start: int
    page_end: int
    chunk_index: int
    content: str
    file_url: str | None = None
    score: float | None = None
    text_match: int | None = None
    vector_distance: float | None = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    count: int
    hits: list[ChunkHit]


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    pdf_id: str
    filename: str
    sha256: str
    num_pages: int
    num_chunks: int
    status: str = "indexed"


class IngestFailure(BaseModel):
    filename: str
    error: str


class BulkIngestResponse(BaseModel):
    ingested: list[IngestResponse]
    failed: list[IngestFailure]
    total_ingested: int
    total_failed: int


# ---------------------------------------------------------------------------
# Chat / RAG  (wired in Step 2; schemas defined now per ACTION_PLAN §4 Step 1)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=50)


class Citation(BaseModel):
    pdf_id: str
    pdf_name: str
    page_start: int
    page_end: int
    chunk_index: int
    file_url: str | None = None


class ChatResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    chunks_used: list[ChunkHit]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    version: str
    checks: dict[str, bool]


# ---------------------------------------------------------------------------
# Storage monitoring
# ---------------------------------------------------------------------------

class PdfStorageInfo(BaseModel):
    num_files: int
    total_size_bytes: int
    total_size_human: str


class StorageResponse(BaseModel):
    typesense_data_dir: str
    disk_used_bytes: int
    disk_used_human: str
    disk_total_bytes: int
    disk_total_human: str
    disk_used_pct: float
    pdf_storage: PdfStorageInfo
