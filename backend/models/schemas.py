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
    # Character span in the parent document's normalized text. -1 for chunks
    # indexed before spans existed; re-ingest to populate them.
    start_char: int = -1
    end_char: int = -1


class SearchResponse(BaseModel):
    query: str
    mode: str
    count: int
    hits: list[ChunkHit]


class DocumentScope(BaseModel):
    """A parent document ranked by its chunks' aggregated scores (stage 1)."""

    pdf_id: str
    pdf_name: str
    score: float
    chunk_count: int


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
    # Character span in the parent document, so a claim resolves to exact source
    # text rather than to a page number a reader has to scan.
    start_char: int = -1
    end_char: int = -1
    # Span label the generator cited (e.g. "S2"); None for unlabelled citations.
    span_id: str | None = None


class VerificationScores(BaseModel):
    """
    The RAG triad, scored independently in 0..1.

    Gated on the MINIMUM, never the mean: an answer can be highly relevant to
    the question while being largely unsupported by the retrieved text, and
    averaging hides exactly that failure.
    """

    context_relevance: float
    groundedness: float
    answer_relevance: float

    @property
    def minimum(self) -> float:
        return min(
            self.context_relevance, self.groundedness, self.answer_relevance
        )


class ChatResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    chunks_used: list[ChunkHit]
    # Fraction of the answer's factual claims that bound to a retrieved span.
    citation_coverage: float = 1.0
    # True when the system declined to answer. An abstention is a valid,
    # preferred outcome — never emit a low-groundedness answer instead.
    abstained: bool = False
    abstain_reason: str | None = None
    verification: VerificationScores | None = None
    # Documents stage 1 scoped to (two-stage), or the nearest documents
    # attached to a Gate-1 abstention so a reviewer sees what came closest.
    scoped_documents: list[DocumentScope] = Field(default_factory=list)
    # Search mode the L6 router chose for this query ("vector" | "hybrid");
    # None only for responses built before retrieval ran.
    retrieval_mode: str | None = None


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
