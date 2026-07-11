from __future__ import annotations

import hashlib
import logging
import os
import re
import uuid
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO

import pymupdf  # PyMuPDF

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _find_tessdata() -> str | None:
    """Locate Tesseract's tessdata dir for PyMuPDF OCR. None → let PyMuPDF try."""
    env = os.getenv("TESSDATA_PREFIX")
    if env and Path(env).is_dir():
        return env
    for candidate in (
        "/usr/share/tesseract-ocr/5/tessdata",
        "/usr/share/tesseract-ocr/4.00/tessdata",
        "/usr/share/tessdata",
        "/usr/local/share/tessdata",
        "/opt/homebrew/share/tessdata",
    ):
        if Path(candidate).is_dir():
            return candidate
    return None


@dataclass
class StoredPDF:
    pdf_id: str
    original_filename: str
    stored_filename: str
    local_path: str
    file_url: str
    sha256: str
    size_bytes: int


@dataclass
class PDFPageText:
    page_number: int
    text: str


@dataclass
class PDFChunk:
    chunk_index: int
    page_start: int
    page_end: int
    content: str


class PDFHelper:
    """
    Handles:
      - storing uploaded PDFs locally
      - extracting page text
      - chunking pages into searchable text chunks
      - building chunk documents for Typesense

    This class is intentionally separate from embeddings.
    """

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        public_base_url: str | None = None,
    ) -> None:
        self.storage_dir = Path(storage_dir or os.getenv("PDF_STORAGE_DIR", "storage/pdfs"))
        self.public_base_url = (
            public_base_url
            if public_base_url is not None
            else os.getenv("PDF_PUBLIC_BASE_URL", "").strip()
        )
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def process_upload_file(
        self,
        upload_file: Any,
        *,
        max_chars: int = 3500,
        overlap_chars: int = 400,
    ) -> dict[str, Any]:
        """
        FastAPI-friendly async wrapper for UploadFile.
        """
        filename = getattr(upload_file, "filename", None) or "document.pdf"
        pdf_bytes = await upload_file.read()
        return self.process_pdf_bytes(
            pdf_bytes=pdf_bytes,
            original_filename=filename,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

    def process_pdf_bytes(
        self,
        *,
        pdf_bytes: bytes,
        original_filename: str,
        max_chars: int = 3500,
        overlap_chars: int = 400,
        pdf_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Save a PDF locally, extract its text, chunk it, and return everything
        in a single structure.
        """
        stored = self.save_pdf_bytes(
            pdf_bytes=pdf_bytes,
            original_filename=original_filename,
            pdf_id=pdf_id,
        )

        pages = self.extract_pages(stored.local_path)
        chunks = self.chunk_pages(
            pages,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

        return {
            "pdf": asdict(stored),
            "pages": [asdict(page) for page in pages],
            "chunks": [asdict(chunk) for chunk in chunks],
        }

    def save_pdf_bytes(
        self,
        *,
        pdf_bytes: bytes,
        original_filename: str,
        pdf_id: str | None = None,
    ) -> StoredPDF:
        """
        Store the uploaded PDF under:
            {storage_dir}/{pdf_id}/{safe_filename}
        """
        pdf_id = pdf_id or str(uuid.uuid4())
        safe_name = self._sanitize_filename(original_filename)
        pdf_dir = self.storage_dir / pdf_id
        pdf_dir.mkdir(parents=True, exist_ok=True)

        local_path = pdf_dir / safe_name
        local_path.write_bytes(pdf_bytes)

        sha256 = hashlib.sha256(pdf_bytes).hexdigest()
        file_url = self._build_file_url(pdf_id, safe_name)

        return StoredPDF(
            pdf_id=pdf_id,
            original_filename=original_filename,
            stored_filename=safe_name,
            local_path=str(local_path),
            file_url=file_url,
            sha256=sha256,
            size_bytes=len(pdf_bytes),
        )

    def save_pdf_fileobj(
        self,
        file_obj: BinaryIO,
        *,
        original_filename: str,
        pdf_id: str | None = None,
    ) -> StoredPDF:
        """
        Sync helper for file-like objects.
        """
        pdf_bytes = file_obj.read()
        if not isinstance(pdf_bytes, (bytes, bytearray)):
            raise TypeError("file_obj.read() must return bytes")
        return self.save_pdf_bytes(
            pdf_bytes=bytes(pdf_bytes),
            original_filename=original_filename,
            pdf_id=pdf_id,
        )

    def extract_pages(
        self,
        pdf_path: str | Path,
        *,
        ocr_enabled: bool = True,
        ocr_min_text_length: int = 50,
    ) -> list[PDFPageText]:
        """
        Extract text from each page of the PDF.
        Falls back to OCR (Tesseract via PyMuPDF) when a page yields too little text.
        Page numbers are 1-based.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(str(pdf_path))

        pages: list[PDFPageText] = []
        try:
            with pymupdf.open(pdf_path) as doc:
                for idx in range(doc.page_count):
                    page = doc.load_page(idx)
                    text = page.get_text("text") or ""
                    text = self._normalize_text(text)

                    if ocr_enabled and len(text) < ocr_min_text_length:
                        ocr_text = self._ocr_page(page, pdf_path.name, idx + 1)
                        if len(ocr_text) > len(text):
                            text = ocr_text

                    pages.append(PDFPageText(page_number=idx + 1, text=text))
        except RuntimeError as exc:
            raise ValueError(
                f"Cannot read PDF (corrupted or unsupported format): {pdf_path}"
            ) from exc

        return pages

    @staticmethod
    def _ocr_page(page: Any, pdf_name: str, page_no: int) -> str:
        """OCR a single page via PyMuPDF + Tesseract. Returns "" on failure."""
        try:
            tp = page.get_textpage_ocr(
                language="eng",
                dpi=300,
                full=True,
                tessdata=_find_tessdata(),
            )
            ocr_text = page.get_text("text", textpage=tp) or ""
            ocr_text = PDFHelper._normalize_text(ocr_text)
            if ocr_text:
                logger.info(
                    "OCR page %d of %s → %d chars", page_no, pdf_name, len(ocr_text)
                )
            return ocr_text
        except Exception as exc:
            logger.warning(
                "OCR failed for page %d of %s: %s", page_no, pdf_name, exc
            )
            return ""

    def chunk_pages(
        self,
        pages: list[PDFPageText],
        *,
        max_chars: int = 3500,
        overlap_chars: int = 400,
    ) -> list[PDFChunk]:
        """
        Group consecutive pages into chunks up to max_chars.
        If a single page is too large, split it into smaller chunks.

        Consecutive accumulated chunks share overlap_chars characters from the
        end of the previous chunk, improving RAG recall at chunk boundaries.
        Overlap is skipped when the next page is too large to fit alongside it.
        """
        chunks: list[PDFChunk] = []

        current_parts: list[str] = []
        current_len = 0
        current_start_page: int | None = None
        current_end_page: int | None = None
        chunk_index = 0

        # FIX B1: carry overlap text from each flushed chunk into the next
        _overlap_text: str = ""
        _overlap_page: int | None = None

        def _seed_overlap(next_page_len: int) -> None:
            """Prepend overlap carry into the current buffer if it fits."""
            nonlocal current_parts, current_len, current_start_page, current_end_page
            nonlocal _overlap_text, _overlap_page
            if not _overlap_text:
                return
            fits = len(_overlap_text) + 2 + next_page_len <= max_chars
            if fits:
                current_parts.append(_overlap_text)
                current_len = len(_overlap_text)
                current_start_page = _overlap_page
                current_end_page = _overlap_page
            _overlap_text = ""
            _overlap_page = None

        def flush_current() -> None:
            nonlocal current_parts, current_len, current_start_page, current_end_page
            nonlocal chunk_index, _overlap_text, _overlap_page
            if not current_parts:
                return
            content = "\n\n".join(current_parts).strip()
            if content:
                chunks.append(
                    PDFChunk(
                        chunk_index=chunk_index,
                        page_start=current_start_page or 1,
                        page_end=current_end_page or (current_start_page or 1),
                        content=content,
                    )
                )
                chunk_index += 1
                if overlap_chars > 0 and len(content) > overlap_chars:
                    _overlap_text = content[-overlap_chars:].strip()
                    _overlap_page = current_end_page
                else:
                    _overlap_text = ""
                    _overlap_page = None
            current_parts.clear()
            current_len = 0
            current_start_page = None
            current_end_page = None

        for page in pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            if len(page_text) > max_chars:
                flush_current()
                # Large-page split manages its own overlap; discard carry
                _overlap_text = ""
                _overlap_page = None
                page_chunks = self._split_long_text(
                    page_text, max_chars=max_chars, overlap_chars=overlap_chars
                )
                for piece in page_chunks:
                    piece = piece.strip()
                    if not piece:
                        continue
                    chunks.append(
                        PDFChunk(
                            chunk_index=chunk_index,
                            page_start=page.page_number,
                            page_end=page.page_number,
                            content=piece,
                        )
                    )
                    chunk_index += 1
                continue

            # Seed a fresh buffer with overlap from the previous chunk
            if not current_parts:
                _seed_overlap(len(page_text))

            candidate_len = current_len + len(page_text) + (2 if current_parts else 0)
            if current_parts and candidate_len > max_chars:
                flush_current()
                _seed_overlap(len(page_text))

            if not current_parts:
                current_start_page = page.page_number

            current_parts.append(page_text)
            current_len += len(page_text) + (2 if current_len else 0)
            current_end_page = page.page_number

        flush_current()
        return chunks

    def build_typesense_chunk_documents(
        self,
        *,
        pdf: StoredPDF | dict[str, Any],
        chunks: list[PDFChunk | dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Convert chunks into Typesense-ready documents with shared PDF metadata.

        FIX B4: emit both page_start/page_end for range tracking and a single
        `page` field (= page_start) to satisfy the Typesense schema's int32 sort field.
        """
        pdf_dict = asdict(pdf) if isinstance(pdf, StoredPDF) else dict(pdf)

        docs: list[dict[str, Any]] = []
        for chunk in chunks:
            chunk_dict = asdict(chunk) if isinstance(chunk, PDFChunk) else dict(chunk)
            docs.append(
                {
                    "id": f"{pdf_dict['pdf_id']}_chunk_{chunk_dict['chunk_index']}",
                    "pdf_id": pdf_dict["pdf_id"],
                    "pdf_name": pdf_dict["original_filename"],
                    "file_url": pdf_dict["file_url"],
                    "local_path": pdf_dict["local_path"],
                    "page_start": chunk_dict["page_start"],
                    "page_end": chunk_dict["page_end"],
                    # Schema sort field: use page_start as the representative page
                    "page": chunk_dict["page_start"],
                    "chunk_index": chunk_dict["chunk_index"],
                    "content": chunk_dict["content"],
                }
            )

        return docs

    def _build_file_url(self, pdf_id: str, stored_filename: str) -> str:
        relative = f"/pdfs/{pdf_id}/{stored_filename}"
        if self.public_base_url:
            return f"{self.public_base_url.rstrip('/')}{relative}"
        return relative

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        filename = Path(filename).name
        filename = re.sub(r"[^\w.\- ]+", "_", filename).strip()
        return filename or "document.pdf"

    @staticmethod
    def _normalize_text(text: str) -> str:
        # FIX B3: normalise CRLF and bare CR before any other processing
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _split_long_text(text: str, *, max_chars: int, overlap_chars: int) -> list[str]:
        """
        Split a large block of text into overlapping pieces.
        This is a fallback for very large pages or extracted legal clauses.

        FIX B2: carry overlap_chars from each flushed paragraph group into the
        next group so that consecutive pieces share context.
        """
        if max_chars <= 0:
            raise ValueError("max_chars must be > 0")
        overlap_chars = max(0, min(overlap_chars, max_chars - 1))

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        pieces: list[str] = []
        current = ""

        for para in paragraphs:
            if not current:
                if len(para) <= max_chars:
                    current = para
                else:
                    pieces.extend(
                        PDFHelper._hard_split_text(
                            para,
                            max_chars=max_chars,
                            overlap_chars=overlap_chars,
                        )
                    )
                    current = ""
                continue

            candidate = f"{current}\n\n{para}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                pieces.append(current)
                # FIX B2: seed next paragraph group with overlap from current
                overlap_seed = (
                    current[-overlap_chars:].strip()
                    if overlap_chars > 0 and len(current) > overlap_chars
                    else ""
                )
                if len(para) <= max_chars:
                    if overlap_seed and len(overlap_seed) + 2 + len(para) <= max_chars:
                        current = overlap_seed + "\n\n" + para
                    else:
                        current = para
                else:
                    pieces.extend(
                        PDFHelper._hard_split_text(
                            para,
                            max_chars=max_chars,
                            overlap_chars=overlap_chars,
                        )
                    )
                    current = ""

        if current.strip():
            pieces.append(current.strip())

        return pieces

    @staticmethod
    def _hard_split_text(text: str, *, max_chars: int, overlap_chars: int) -> list[str]:
        """
        Last-resort fixed-size split with overlap.
        """
        cleaned = text.strip()
        if not cleaned:
            return []
        if len(cleaned) <= max_chars:
            return [cleaned]

        chunks: list[str] = []
        start = 0
        n = len(cleaned)

        while start < n:
            end = min(start + max_chars, n)
            chunk = cleaned[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= n:
                break
            start = max(end - overlap_chars, start + 1)

        return chunks


def get_pdf_helper() -> PDFHelper:
    return PDFHelper()
