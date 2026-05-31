"""IngestionService — orchestration, idempotency, batch error isolation."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pymupdf
import pytest

from core.config import get_settings
from core.exceptions import IngestionError
from helpers.pdf_helper import PDFChunk, PDFHelper, PDFPageText, StoredPDF
from repositories.pdf_storage_repository import PdfStorageRepository
from services.ingestion_service import IngestionService


# --- Fakes -----------------------------------------------------------------

class FakeEmbeddings:
    def embed_chunk_documents(self, docs, text_key="content"):
        return [{**d, "embedding": [0.1] * 4} for d in docs]


class FakeTypesense:
    def __init__(self):
        self.indexed: list[dict] = []

    def index_chunks(self, docs):
        self.indexed.extend(docs)
        return [{"success": True} for _ in docs]


class FakePdfRepo:
    """Minimal in-memory PDF repo returning canned structures."""

    def __init__(self, *, pages=2, chunks=3):
        self._pages = pages
        self._chunks = chunks
        self.saved_pdf_id = None

    def save(self, *, pdf_bytes, original_filename, pdf_id=None):
        self.saved_pdf_id = pdf_id
        return StoredPDF(
            pdf_id=pdf_id or "auto",
            original_filename=original_filename,
            stored_filename=original_filename,
            local_path=f"/tmp/{pdf_id}/{original_filename}",
            file_url=f"/pdfs/{pdf_id}/{original_filename}",
            sha256=pdf_id or "auto",
            size_bytes=len(pdf_bytes),
        )

    def extract_pages(self, local_path):
        return [PDFPageText(i + 1, f"page {i+1} text") for i in range(self._pages)]

    def chunk(self, pages, *, max_chars, overlap_chars):
        return [PDFChunk(i, 1, 1, f"chunk {i}") for i in range(self._chunks)]

    def build_documents(self, *, pdf, chunks):
        return [
            {"id": f"{pdf.pdf_id}_chunk_{c.chunk_index}", "content": c.content,
             "pdf_id": pdf.pdf_id}
            for c in chunks
        ]


def _make_service(pdf_repo=None, ts=None):
    return IngestionService(
        pdf_repo=pdf_repo or FakePdfRepo(),
        embedding_service=FakeEmbeddings(),
        typesense_repo=ts or FakeTypesense(),
        settings=get_settings(),
    )


# --- ingest_bytes ----------------------------------------------------------

def test_ingest_bytes_success():
    ts = FakeTypesense()
    svc = _make_service(FakePdfRepo(pages=2, chunks=3), ts)
    result = svc.ingest_bytes(pdf_bytes=b"hello pdf", filename="x.pdf")
    assert result.num_pages == 2
    assert result.num_chunks == 3
    assert result.status == "indexed"
    assert len(ts.indexed) == 3
    assert all("embedding" in d for d in ts.indexed)


def test_idempotent_pdf_id_is_sha256():
    pdf_repo = FakePdfRepo()
    svc = _make_service(pdf_repo)
    data = b"deterministic bytes"
    result = svc.ingest_bytes(pdf_bytes=data, filename="x.pdf")
    expected_sha = hashlib.sha256(data).hexdigest()
    assert result.pdf_id == expected_sha
    assert pdf_repo.saved_pdf_id == expected_sha


def test_empty_bytes_raises():
    with pytest.raises(IngestionError):
        _make_service().ingest_bytes(pdf_bytes=b"", filename="x.pdf")


def test_no_chunks_raises():
    svc = _make_service(FakePdfRepo(pages=1, chunks=0))
    with pytest.raises(IngestionError):
        svc.ingest_bytes(pdf_bytes=b"data", filename="empty.pdf")


# --- ingest_directory (real PDF repo, corrupted-file isolation) ------------

def _valid_pdf_bytes() -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Valid legal content about torts.")
    return doc.tobytes()


def test_ingest_directory_isolates_failures(tmp_path):
    data_dir = tmp_path / "corpus"
    data_dir.mkdir()
    (data_dir / "good.pdf").write_bytes(_valid_pdf_bytes())
    (data_dir / "bad.pdf").write_bytes(b"%PDF-1.4 garbage not a real pdf")

    pdf_repo = PdfStorageRepository(PDFHelper(storage_dir=tmp_path / "store"))
    svc = _make_service(pdf_repo)

    ingested, failed = svc.ingest_directory(data_dir)

    assert [r.filename for r in ingested] == ["good.pdf"]
    assert [f.filename for f in failed] == ["bad.pdf"]


def test_ingest_directory_not_a_dir_raises(tmp_path):
    with pytest.raises(IngestionError):
        _make_service().ingest_directory(tmp_path / "missing")


def test_ingest_directory_reports_unreadable_entry_without_crashing(tmp_path):
    """B1 regression: a glob match that can't be read (here a directory named
    *.pdf) must be reported as a failure, not abort the whole batch."""
    data_dir = tmp_path / "corpus"
    data_dir.mkdir()
    (data_dir / "good.pdf").write_bytes(_valid_pdf_bytes())
    (data_dir / "is_a_dir.pdf").mkdir()  # matches *.pdf but read_bytes → OSError

    pdf_repo = PdfStorageRepository(PDFHelper(storage_dir=tmp_path / "store"))
    svc = _make_service(pdf_repo)

    ingested, failed = svc.ingest_directory(data_dir)

    assert [r.filename for r in ingested] == ["good.pdf"]
    assert [f.filename for f in failed] == ["is_a_dir.pdf"]
    assert failed[0].error  # carries the OS error message
