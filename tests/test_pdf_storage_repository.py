"""PdfStorageRepository — real PDFHelper over a temp dir."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pymupdf
import pytest

from core.exceptions import IngestionError
from helpers.pdf_helper import PDFHelper
from repositories.pdf_storage_repository import PdfStorageRepository


def _pdf_bytes(text: str = "Section 1. A contract requires offer and acceptance.") -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    return doc.tobytes()


@pytest.fixture
def repo(tmp_path):
    return PdfStorageRepository(PDFHelper(storage_dir=tmp_path / "pdfs"))


def test_full_pipeline(repo):
    stored = repo.save(pdf_bytes=_pdf_bytes(), original_filename="contract.pdf", pdf_id="id1")
    assert Path(stored.local_path).exists()
    assert stored.pdf_id == "id1"

    pages = repo.extract_pages(stored.local_path)
    assert len(pages) >= 1

    chunks = repo.chunk(pages, max_chars=3500, overlap_chars=400)
    assert len(chunks) >= 1

    docs = repo.build_documents(pdf=stored, chunks=chunks)
    assert docs[0]["pdf_id"] == "id1"
    assert docs[0]["content"].strip()


def test_extract_missing_file_raises_ingestion_error(repo):
    with pytest.raises(IngestionError):
        repo.extract_pages("/nonexistent/file.pdf")


def test_extract_corrupted_bytes_raises_ingestion_error(repo, tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 not really a pdf")
    with pytest.raises(IngestionError):
        repo.extract_pages(str(bad))


def test_save_returns_sha_and_size(repo):
    data = _pdf_bytes()
    stored = repo.save(pdf_bytes=data, original_filename="x.pdf", pdf_id="abc")
    assert stored.size_bytes == len(data)
    assert len(stored.sha256) == 64


def test_save_wraps_oserror_as_ingestion_error(repo):
    # A storage failure (disk full, permissions) must surface as IngestionError.
    def boom(**kwargs):
        raise OSError("disk full")

    repo._helper.save_pdf_bytes = boom
    with pytest.raises(IngestionError):
        repo.save(pdf_bytes=b"data", original_filename="x.pdf", pdf_id="id")
