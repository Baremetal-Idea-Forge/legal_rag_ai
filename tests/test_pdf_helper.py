"""
Comprehensive test suite for backend/helpers/pdf_helper.py.

Covers:
  - _normalize_text
  - _sanitize_filename
  - _hard_split_text
  - _split_long_text
  - chunk_pages
  - extract_pages
  - save_pdf_bytes / save_pdf_fileobj
  - process_pdf_bytes / process_upload_file
  - build_typesense_chunk_documents
  - Integration: all readable PDFs in /data (corrupted PDFs tested separately)
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pymupdf
import pytest

from helpers.pdf_helper import (
    PDFChunk,
    PDFHelper,
    PDFPageText,
    StoredPDF,
    get_pdf_helper,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pages(*texts: str) -> list[PDFPageText]:
    return [PDFPageText(i + 1, t) for i, t in enumerate(texts)]


def all_chunks_within(chunks: list[PDFChunk], max_chars: int) -> bool:
    return all(len(c.content) <= max_chars for c in chunks)


def chunk_indices_sequential(chunks: list[PDFChunk]) -> bool:
    return [c.chunk_index for c in chunks] == list(range(len(chunks)))


# ===========================================================================
# 1. _normalize_text
# ===========================================================================

class TestNormalizeText:
    def test_removes_null_bytes(self):
        assert "\x00" not in PDFHelper._normalize_text("hello\x00world")

    def test_collapses_horizontal_whitespace(self):
        result = PDFHelper._normalize_text("a   b\t\tc")
        assert "  " not in result
        assert "\t" not in result

    def test_collapses_triple_newlines_to_double(self):
        result = PDFHelper._normalize_text("a\n\n\n\n\nb")
        assert "\n\n\n" not in result
        assert "a\n\nb" == result

    def test_strips_leading_trailing_whitespace(self):
        assert PDFHelper._normalize_text("  hello  ") == "hello"

    def test_empty_string(self):
        assert PDFHelper._normalize_text("") == ""

    # FIX B3 regression tests
    def test_crlf_line_endings_replaced(self):
        result = PDFHelper._normalize_text("Line1\r\nLine2\r\nLine3")
        assert "\r" not in result
        assert result == "Line1\nLine2\nLine3"

    def test_bare_cr_replaced(self):
        result = PDFHelper._normalize_text("Line1\rLine2")
        assert "\r" not in result
        assert result == "Line1\nLine2"

    def test_crlf_paragraph_boundary_preserved(self):
        result = PDFHelper._normalize_text("Para1\r\n\r\nPara2")
        assert result == "Para1\n\nPara2"

    def test_mixed_crlf_and_lf(self):
        result = PDFHelper._normalize_text("a\r\nb\nc\r\nd")
        assert "\r" not in result

    def test_null_byte_with_crlf(self):
        result = PDFHelper._normalize_text("a\x00b\r\nc")
        assert "\x00" not in result
        assert "\r" not in result


# ===========================================================================
# 2. _sanitize_filename
# ===========================================================================

class TestSanitizeFilename:
    def test_normal_filename_unchanged(self):
        assert PDFHelper._sanitize_filename("document.pdf") == "document.pdf"

    def test_strips_directory_path(self):
        result = PDFHelper._sanitize_filename("/etc/passwd")
        assert "/" not in result

    def test_replaces_special_chars(self):
        result = PDFHelper._sanitize_filename("my<file>.pdf")
        assert "<" not in result
        assert ">" not in result

    def test_preserves_spaces(self):
        # filenames like "ANCIENT_LAW _F.pdf" have internal spaces — keep them
        result = PDFHelper._sanitize_filename("ANCIENT_LAW _F.pdf")
        assert result == "ANCIENT_LAW _F.pdf"

    def test_preserves_hyphens_and_underscores(self):
        result = PDFHelper._sanitize_filename("my-file_v2.pdf")
        assert result == "my-file_v2.pdf"

    def test_special_chars_replaced_with_underscore(self):
        # Special chars become underscores; the result is still a valid filename
        result = PDFHelper._sanitize_filename("<<>>")
        assert len(result) > 0
        assert result == "_"

    def test_empty_string_falls_back_to_default(self):
        # Only a truly empty string triggers the fallback
        result = PDFHelper._sanitize_filename("")
        assert result == "document.pdf"

    def test_filename_with_only_extension(self):
        result = PDFHelper._sanitize_filename(".pdf")
        assert result == ".pdf"

    def test_real_data_filenames_are_safe(self):
        """All filenames from /data should survive sanitization intact."""
        data_dir = Path(__file__).parent.parent / "data"
        for pdf in data_dir.glob("*.pdf"):
            sanitized = PDFHelper._sanitize_filename(pdf.name)
            assert len(sanitized) > 0


# ===========================================================================
# 3. _hard_split_text
# ===========================================================================

class TestHardSplitText:
    def test_short_text_returns_single_piece(self):
        result = PDFHelper._hard_split_text("Hello", max_chars=100, overlap_chars=10)
        assert result == ["Hello"]

    def test_exact_max_chars_returns_single_piece(self):
        text = "A" * 100
        result = PDFHelper._hard_split_text(text, max_chars=100, overlap_chars=0)
        assert len(result) == 1

    def test_all_pieces_within_max_chars(self):
        text = "X" * 1000
        for overlap in (0, 50, 99):
            result = PDFHelper._hard_split_text(text, max_chars=100, overlap_chars=overlap)
            assert all(len(p) <= 100 for p in result), f"overlap={overlap}"

    def test_no_pieces_empty(self):
        text = "A" * 500
        result = PDFHelper._hard_split_text(text, max_chars=100, overlap_chars=20)
        assert all(p.strip() for p in result)

    def test_overlap_present_between_consecutive_pieces(self):
        text = "ABCDEFGHIJ" * 20  # 200 chars, predictable content
        result = PDFHelper._hard_split_text(text, max_chars=30, overlap_chars=10)
        for i in range(len(result) - 1):
            tail = result[i][-10:]
            head = result[i + 1][:10]
            assert tail == head, f"No overlap between piece {i} and {i+1}"

    def test_zero_overlap_no_repeated_content(self):
        text = "A" * 100
        result = PDFHelper._hard_split_text(text, max_chars=10, overlap_chars=0)
        # With zero overlap each char appears exactly once
        total = sum(len(p) for p in result)
        assert total == 100

    def test_full_content_covered(self):
        # The first and last characters should appear in the output
        text = "START" + "M" * 200 + "END"
        result = PDFHelper._hard_split_text(text, max_chars=50, overlap_chars=10)
        assert result[0].startswith("START")
        assert result[-1].endswith("END")

    def test_empty_text_returns_empty_list(self):
        result = PDFHelper._hard_split_text("", max_chars=100, overlap_chars=10)
        assert result == []

    def test_whitespace_only_returns_empty_list(self):
        result = PDFHelper._hard_split_text("   ", max_chars=100, overlap_chars=10)
        assert result == []

    def test_invalid_max_chars_raises(self):
        with pytest.raises(ValueError):
            PDFHelper._split_long_text("text", max_chars=0, overlap_chars=0)

    def test_overlap_capped_at_max_chars_minus_one(self):
        # Providing overlap >= max_chars should not cause infinite loop
        text = "A" * 100
        result = PDFHelper._hard_split_text(text, max_chars=10, overlap_chars=10)
        assert len(result) > 0

    def test_single_char_max(self):
        text = "ABCDE"
        result = PDFHelper._hard_split_text(text, max_chars=1, overlap_chars=0)
        assert result == ["A", "B", "C", "D", "E"]


# ===========================================================================
# 4. _split_long_text
# ===========================================================================

class TestSplitLongText:
    def test_short_text_single_piece(self):
        result = PDFHelper._split_long_text("Short text.", max_chars=1000, overlap_chars=100)
        assert len(result) == 1
        assert result[0] == "Short text."

    def test_all_pieces_within_max_chars(self):
        text = ("Legal clause number one. " * 40 + "\n\n") * 5
        result = PDFHelper._split_long_text(text, max_chars=500, overlap_chars=50)
        assert all(len(p) <= 500 for p in result), f"max={max(len(p) for p in result)}"

    def test_no_empty_pieces(self):
        text = ("Para text. " * 30 + "\n\n") * 4
        result = PDFHelper._split_long_text(text, max_chars=400, overlap_chars=50)
        assert all(p.strip() for p in result)

    def test_zero_overlap(self):
        text = ("X" * 200 + "\n\n") * 5
        result = PDFHelper._split_long_text(text, max_chars=300, overlap_chars=0)
        assert len(result) >= 2
        assert all(len(p) <= 300 for p in result)

    def test_invalid_max_chars_raises(self):
        with pytest.raises(ValueError):
            PDFHelper._split_long_text("text", max_chars=0, overlap_chars=0)

    def test_paragraph_boundary_respected(self):
        # Two paragraphs each 200 chars, max_chars=300 — must split between them
        p1 = "A" * 200
        p2 = "B" * 200
        text = f"{p1}\n\n{p2}"
        result = PDFHelper._split_long_text(text, max_chars=300, overlap_chars=0)
        assert len(result) == 2
        assert result[0] == p1
        assert result[1] == p2

    # FIX B2 regression test
    def test_overlap_carried_between_paragraph_groups(self):
        """Each piece after the first should START with content from the end of the previous."""
        paragraphs = ["Legal clause " + str(i) + ". " * 30 for i in range(6)]
        text = "\n\n".join(paragraphs)
        result = PDFHelper._split_long_text(text, max_chars=300, overlap_chars=50)
        # At least two pieces needed to test overlap
        if len(result) < 2:
            pytest.skip("text too short to produce multiple pieces")
        for i in range(1, len(result)):
            prev_tail = result[i - 1][-50:]
            curr_head = result[i][:50]
            assert curr_head in result[i - 1], (
                f"Piece {i} head={repr(curr_head)} not in end of piece {i-1}: {repr(result[i-1][-80:])}"
            )

    def test_text_without_paragraph_breaks(self):
        # Falls back to hard split; should not raise
        text = "A" * 1000
        result = PDFHelper._split_long_text(text, max_chars=200, overlap_chars=30)
        assert len(result) > 1
        assert all(len(p) <= 200 for p in result)

    def test_empty_paragraphs_ignored(self):
        text = "Para1\n\n\n\nPara2"
        result = PDFHelper._split_long_text(text, max_chars=1000, overlap_chars=50)
        # Should produce a single piece (both paragraphs fit)
        assert len(result) == 1


# ===========================================================================
# 5. chunk_pages
# ===========================================================================

class TestChunkPages:
    def test_empty_pages_returns_empty(self):
        assert PDFHelper().chunk_pages([]) == []

    def test_all_empty_pages_returns_empty(self):
        pages = make_pages("", "   ", "\n")
        assert PDFHelper().chunk_pages(pages) == []

    def test_single_short_page(self):
        pages = make_pages("Hello world.")
        chunks = PDFHelper().chunk_pages(pages)
        assert len(chunks) == 1
        assert chunks[0].content == "Hello world."
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 1
        assert chunks[0].chunk_index == 0

    def test_short_pages_accumulate_into_one_chunk(self):
        pages = make_pages("Page one.", "Page two.", "Page three.")
        chunks = PDFHelper().chunk_pages(pages, max_chars=200, overlap_chars=0)
        assert len(chunks) == 1
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 3

    def test_chunk_indices_are_sequential_from_zero(self):
        pages = make_pages(*["P" * 2000 for _ in range(6)])
        chunks = PDFHelper().chunk_pages(pages, max_chars=3500, overlap_chars=0)
        assert chunk_indices_sequential(chunks)

    def test_all_chunks_within_max_chars(self):
        pages = make_pages(*["Legal text. " * 200 for _ in range(10)])
        chunks = PDFHelper().chunk_pages(pages, max_chars=3500, overlap_chars=400)
        assert all_chunks_within(chunks, 3500), \
            f"max chunk size={max(len(c.content) for c in chunks)}"

    def test_oversized_page_is_split(self):
        # A page with 8000 chars must produce multiple chunks
        pages = make_pages("A" * 8000)
        chunks = PDFHelper().chunk_pages(pages, max_chars=3500, overlap_chars=400)
        assert len(chunks) >= 2
        assert all_chunks_within(chunks, 3500)

    def test_page_start_end_correct_multi_page_chunk(self):
        pages = make_pages("Short A.", "Short B.", "Short C.")
        chunks = PDFHelper().chunk_pages(pages, max_chars=200, overlap_chars=0)
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 3

    def test_page_start_end_correct_single_page_chunk(self):
        pages = make_pages("A" * 3400, "B" * 3400)
        chunks = PDFHelper().chunk_pages(pages, max_chars=3500, overlap_chars=0)
        assert all(c.page_start == c.page_end for c in chunks)

    def test_empty_pages_skipped_in_numbering(self):
        pages = [PDFPageText(1, ""), PDFPageText(2, "Content"), PDFPageText(3, "")]
        chunks = PDFHelper().chunk_pages(pages)
        assert len(chunks) == 1
        assert chunks[0].page_start == 2

    def test_no_empty_chunk_content(self):
        pages = make_pages(*["Word " * 100 for _ in range(20)])
        chunks = PDFHelper().chunk_pages(pages, max_chars=3500, overlap_chars=400)
        assert all(c.content.strip() for c in chunks)

    def test_page_after_oversized_page_gets_own_chunk(self):
        pages = make_pages("A" * 5000, "B" * 100)
        chunks = PDFHelper().chunk_pages(pages, max_chars=3500, overlap_chars=400)
        last = chunks[-1]
        # The "B" page should appear as its own chunk (not merged into large-page chunks)
        assert "B" in last.content
        assert last.page_start == 2

    def test_chunk_pages_skips_empty_pieces_from_split_long_text(self):
        """
        Line 277 guard: if _split_long_text returns whitespace-only strings
        they must be silently dropped (never stored as chunks).
        This is a defensive guard; it exercises the `if not piece: continue` branch.
        """
        from unittest.mock import patch
        pages = make_pages("A" * 5000)  # triggers the large-page split path
        with patch.object(PDFHelper, "_split_long_text", return_value=["real content", "   ", ""]):
            chunks = PDFHelper().chunk_pages(pages, max_chars=3500, overlap_chars=400)
        assert len(chunks) == 1
        assert chunks[0].content == "real content"

    # FIX B1 regression tests
    def test_overlap_between_accumulated_chunks(self):
        """After flush, the next chunk should start with content from end of previous."""
        # p1 fills chunk to ~3400 chars; p2 (200) forces a flush + new chunk
        p1_text = "A" * 3400
        p2_text = "B" * 200
        pages = [PDFPageText(1, p1_text), PDFPageText(2, p2_text)]
        chunks = PDFHelper().chunk_pages(pages, max_chars=3500, overlap_chars=400)
        assert len(chunks) >= 2
        # chunk1 must start with overlap from chunk0 (tail of A's)
        assert chunks[1].content.startswith("A"), \
            f"Expected chunk1 to start with overlap 'A...', got: {repr(chunks[1].content[:50])}"

    def test_overlap_does_not_exceed_max_chars(self):
        pages = make_pages(*["Legal text. " * 150 for _ in range(15)])
        chunks = PDFHelper().chunk_pages(pages, max_chars=3500, overlap_chars=400)
        assert all_chunks_within(chunks, 3500)

    def test_overlap_skipped_when_page_too_large(self):
        """When a page is so large that overlap + page > max_chars, overlap is skipped."""
        # overlap=400, p2=3200: 400+2+3200=3602 > 3500 → no overlap seeded
        pages = [PDFPageText(1, "A" * 3400), PDFPageText(2, "B" * 3200)]
        chunks = PDFHelper().chunk_pages(pages, max_chars=3500, overlap_chars=400)
        # chunk1 should contain only B's (no overlap from A's)
        assert chunks[1].content.startswith("B")

    def test_zero_overlap_no_shared_content(self):
        """With overlap=0, consecutive chunks must not share content."""
        p1_text = "A" * 3400
        p2_text = "B" * 200
        pages = [PDFPageText(1, p1_text), PDFPageText(2, p2_text)]
        chunks = PDFHelper().chunk_pages(pages, max_chars=3500, overlap_chars=0)
        assert len(chunks) >= 2
        # chunk1 must NOT start with A (no overlap)
        assert not chunks[1].content.startswith("A"), \
            "Unexpected overlap when overlap_chars=0"

    def test_many_tiny_pages(self):
        # 100 pages × 10 chars = fits in one chunk
        pages = [PDFPageText(i + 1, f"pg{i:03d} ") for i in range(100)]
        chunks = PDFHelper().chunk_pages(pages, max_chars=3500, overlap_chars=100)
        assert len(chunks) >= 1
        assert chunk_indices_sequential(chunks)
        assert all_chunks_within(chunks, 3500)

    def test_single_very_long_page_fully_covered(self):
        """All content from a long page should appear in the output chunks."""
        text = "LEGAL " * 2000  # 12000 chars
        pages = [PDFPageText(1, text)]
        chunks = PDFHelper().chunk_pages(pages, max_chars=3500, overlap_chars=400)
        combined = " ".join(c.content for c in chunks)
        # First and last tokens must be present
        assert "LEGAL" in combined


# ===========================================================================
# 6. extract_pages
# ===========================================================================

class TestExtractPages:
    # Use READABLE_PDFS from conftest (excludes corrupted files)
    _safe_pdfs = None

    @classmethod
    def _get_safe_pdfs(cls, n: int = 5) -> list[Path]:
        from conftest import READABLE_PDFS
        return READABLE_PDFS[:n]

    def test_missing_file_raises_file_not_found(self, tmp_path):
        h = PDFHelper(storage_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            h.extract_pages(tmp_path / "nonexistent.pdf")

    def test_corrupted_pdf_raises_value_error(self, corrupted_pdf_path):
        with pytest.raises(ValueError, match="corrupted"):
            PDFHelper().extract_pages(corrupted_pdf_path)

    def test_page_numbers_are_1_based(self):
        pdf_path = self._get_safe_pdfs(1)[0]
        pages = PDFHelper().extract_pages(pdf_path)
        assert pages[0].page_number == 1

    def test_page_count_positive(self):
        for pdf in self._get_safe_pdfs():
            pages = PDFHelper().extract_pages(pdf)
            assert len(pages) > 0

    def test_no_null_bytes_in_extracted_text(self):
        for pdf in self._get_safe_pdfs():
            for page in PDFHelper().extract_pages(pdf):
                assert "\x00" not in page.text

    def test_no_crlf_in_extracted_text(self):
        for pdf in self._get_safe_pdfs():
            for page in PDFHelper().extract_pages(pdf):
                assert "\r" not in page.text, \
                    f"CR found in {pdf.name} page {page.page_number}"

    def test_no_excessive_newlines(self):
        for pdf in self._get_safe_pdfs():
            for page in PDFHelper().extract_pages(pdf):
                assert "\n\n\n" not in page.text


# ===========================================================================
# 7. save_pdf_bytes / save_pdf_fileobj
# ===========================================================================

class TestSavePDF:
    def _minimal_pdf_bytes(self) -> bytes:
        """Create a minimal valid PDF in memory using PyMuPDF."""
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test PDF content for legal RAG.")
        return doc.tobytes()

    def test_save_pdf_bytes_creates_file(self, tmp_storage):
        pdf_bytes = self._minimal_pdf_bytes()
        stored = tmp_storage.save_pdf_bytes(
            pdf_bytes=pdf_bytes, original_filename="test.pdf"
        )
        assert Path(stored.local_path).exists()

    def test_save_pdf_bytes_sha256_correct(self, tmp_storage):
        pdf_bytes = self._minimal_pdf_bytes()
        stored = tmp_storage.save_pdf_bytes(
            pdf_bytes=pdf_bytes, original_filename="test.pdf"
        )
        expected = hashlib.sha256(pdf_bytes).hexdigest()
        assert stored.sha256 == expected

    def test_save_pdf_bytes_size_correct(self, tmp_storage):
        pdf_bytes = self._minimal_pdf_bytes()
        stored = tmp_storage.save_pdf_bytes(
            pdf_bytes=pdf_bytes, original_filename="test.pdf"
        )
        assert stored.size_bytes == len(pdf_bytes)

    def test_save_pdf_bytes_fixed_pdf_id(self, tmp_storage):
        pdf_bytes = self._minimal_pdf_bytes()
        stored = tmp_storage.save_pdf_bytes(
            pdf_bytes=pdf_bytes, original_filename="test.pdf", pdf_id="my-fixed-id"
        )
        assert stored.pdf_id == "my-fixed-id"
        assert "my-fixed-id" in stored.local_path

    def test_save_pdf_bytes_auto_pdf_id(self, tmp_storage):
        pdf_bytes = self._minimal_pdf_bytes()
        stored = tmp_storage.save_pdf_bytes(
            pdf_bytes=pdf_bytes, original_filename="test.pdf"
        )
        assert len(stored.pdf_id) > 0

    def test_save_pdf_bytes_file_url_relative(self, tmp_storage):
        pdf_bytes = self._minimal_pdf_bytes()
        stored = tmp_storage.save_pdf_bytes(
            pdf_bytes=pdf_bytes, original_filename="test.pdf"
        )
        assert stored.file_url.startswith("/pdfs/")

    def test_save_pdf_bytes_file_url_with_base_url(self, tmp_path):
        h = PDFHelper(storage_dir=tmp_path, public_base_url="https://example.com")
        pdf_bytes = self._minimal_pdf_bytes()
        stored = h.save_pdf_bytes(pdf_bytes=pdf_bytes, original_filename="test.pdf")
        assert stored.file_url.startswith("https://example.com/pdfs/")

    def test_save_pdf_fileobj(self, tmp_storage):
        pdf_bytes = self._minimal_pdf_bytes()
        stored = tmp_storage.save_pdf_fileobj(
            io.BytesIO(pdf_bytes), original_filename="fileobj.pdf"
        )
        assert Path(stored.local_path).exists()
        assert stored.sha256 == hashlib.sha256(pdf_bytes).hexdigest()

    def test_save_pdf_fileobj_non_bytes_raises(self, tmp_storage):
        class BadFile:
            def read(self):
                return "not bytes"

        with pytest.raises(TypeError):
            tmp_storage.save_pdf_fileobj(BadFile(), original_filename="bad.pdf")

    def test_duplicate_pdf_id_overwrites_file(self, tmp_storage):
        pdf_bytes = self._minimal_pdf_bytes()
        s1 = tmp_storage.save_pdf_bytes(
            pdf_bytes=pdf_bytes, original_filename="dup.pdf", pdf_id="dup-id"
        )
        different_bytes = pdf_bytes + b"%comment"
        s2 = tmp_storage.save_pdf_bytes(
            pdf_bytes=different_bytes, original_filename="dup.pdf", pdf_id="dup-id"
        )
        # File on disk should reflect the latest write
        assert Path(s2.local_path).read_bytes() == different_bytes


# ===========================================================================
# 8. build_typesense_chunk_documents
# ===========================================================================

class TestBuildTypesenseDocs:
    def _make_stored(self) -> StoredPDF:
        return StoredPDF(
            pdf_id="abc123",
            original_filename="contract.pdf",
            stored_filename="contract.pdf",
            local_path="/storage/abc123/contract.pdf",
            file_url="/pdfs/abc123/contract.pdf",
            sha256="deadbeef",
            size_bytes=1024,
        )

    def test_returns_correct_count(self):
        stored = self._make_stored()
        chunks = [PDFChunk(i, i + 1, i + 1, f"Content {i}") for i in range(5)]
        docs = PDFHelper().build_typesense_chunk_documents(pdf=stored, chunks=chunks)
        assert len(docs) == 5

    def test_id_format(self):
        stored = self._make_stored()
        chunks = [PDFChunk(0, 1, 1, "text")]
        docs = PDFHelper().build_typesense_chunk_documents(pdf=stored, chunks=chunks)
        assert docs[0]["id"] == "abc123_chunk_0"

    def test_required_fields_present(self):
        stored = self._make_stored()
        chunks = [PDFChunk(0, 1, 2, "Legal text here.")]
        docs = PDFHelper().build_typesense_chunk_documents(pdf=stored, chunks=chunks)
        required = {"id", "pdf_id", "pdf_name", "file_url", "local_path",
                    "page_start", "page_end", "page", "chunk_index", "content"}
        assert required.issubset(docs[0].keys())

    # FIX B4 regression test
    def test_page_field_equals_page_start(self):
        """Schema sort field 'page' must equal page_start."""
        stored = self._make_stored()
        chunks = [PDFChunk(0, 3, 5, "text")]
        docs = PDFHelper().build_typesense_chunk_documents(pdf=stored, chunks=chunks)
        assert docs[0]["page"] == 3
        assert docs[0]["page_start"] == 3
        assert docs[0]["page_end"] == 5

    def test_accepts_dict_pdf(self):
        stored = self._make_stored()
        pdf_dict = {
            "pdf_id": "abc123",
            "original_filename": "contract.pdf",
            "file_url": "/pdfs/abc123/contract.pdf",
            "local_path": "/storage/abc123/contract.pdf",
        }
        chunks = [PDFChunk(0, 1, 1, "text")]
        docs = PDFHelper().build_typesense_chunk_documents(pdf=pdf_dict, chunks=chunks)
        assert docs[0]["pdf_id"] == "abc123"

    def test_accepts_dict_chunks(self):
        stored = self._make_stored()
        chunk_dict = {"chunk_index": 0, "page_start": 1, "page_end": 1, "content": "text"}
        docs = PDFHelper().build_typesense_chunk_documents(pdf=stored, chunks=[chunk_dict])
        assert docs[0]["content"] == "text"

    def test_empty_chunks_returns_empty_list(self):
        stored = self._make_stored()
        docs = PDFHelper().build_typesense_chunk_documents(pdf=stored, chunks=[])
        assert docs == []

    def test_content_matches_chunk(self):
        stored = self._make_stored()
        chunks = [PDFChunk(0, 1, 1, "Specific legal clause content.")]
        docs = PDFHelper().build_typesense_chunk_documents(pdf=stored, chunks=chunks)
        assert docs[0]["content"] == "Specific legal clause content."


# ===========================================================================
# 9. process_pdf_bytes  (lines 98-111)
# ===========================================================================

class TestProcessPdfBytes:
    def _minimal_pdf_bytes(self) -> bytes:
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Section 1. All contracts must be in writing.")
        return doc.tobytes()

    def test_returns_pdf_pages_chunks_keys(self, tmp_storage):
        result = tmp_storage.process_pdf_bytes(
            pdf_bytes=self._minimal_pdf_bytes(),
            original_filename="contract.pdf",
        )
        assert set(result.keys()) == {"pdf", "pages", "chunks"}

    def test_pdf_section_has_required_keys(self, tmp_storage):
        result = tmp_storage.process_pdf_bytes(
            pdf_bytes=self._minimal_pdf_bytes(),
            original_filename="contract.pdf",
        )
        for key in ("pdf_id", "sha256", "size_bytes", "local_path"):
            assert key in result["pdf"]

    def test_pages_is_list(self, tmp_storage):
        result = tmp_storage.process_pdf_bytes(
            pdf_bytes=self._minimal_pdf_bytes(),
            original_filename="contract.pdf",
        )
        assert isinstance(result["pages"], list)
        assert len(result["pages"]) > 0

    def test_chunks_is_list(self, tmp_storage):
        result = tmp_storage.process_pdf_bytes(
            pdf_bytes=self._minimal_pdf_bytes(),
            original_filename="contract.pdf",
        )
        assert isinstance(result["chunks"], list)
        assert len(result["chunks"]) > 0

    def test_custom_pdf_id_propagates(self, tmp_storage):
        result = tmp_storage.process_pdf_bytes(
            pdf_bytes=self._minimal_pdf_bytes(),
            original_filename="contract.pdf",
            pdf_id="custom-id-123",
        )
        assert result["pdf"]["pdf_id"] == "custom-id-123"


# ===========================================================================
# 9b. process_upload_file  (lines 76-78 — async FastAPI wrapper)
# ===========================================================================

class TestProcessUploadFile:
    def _minimal_pdf_bytes(self) -> bytes:
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Uploaded document content.")
        return doc.tobytes()

    def test_process_upload_file_returns_same_structure_as_bytes(self, tmp_storage):
        pdf_bytes = self._minimal_pdf_bytes()

        upload = AsyncMock()
        upload.filename = "upload.pdf"
        upload.read = AsyncMock(return_value=pdf_bytes)

        result = asyncio.run(tmp_storage.process_upload_file(upload))
        assert set(result.keys()) == {"pdf", "pages", "chunks"}
        assert result["pdf"]["original_filename"] == "upload.pdf"

    def test_process_upload_file_fallback_filename(self, tmp_storage):
        pdf_bytes = self._minimal_pdf_bytes()

        upload = AsyncMock()
        upload.filename = None
        upload.read = AsyncMock(return_value=pdf_bytes)

        result = asyncio.run(tmp_storage.process_upload_file(upload))
        assert result["pdf"]["original_filename"] == "document.pdf"


# ===========================================================================
# 9c. Edge-case coverage for _split_long_text and chunk_pages
# ===========================================================================

def test_split_long_text_all_whitespace_paragraphs():
    # Text that splits into only empty strings → uses [text.strip()] fallback (line 379)
    text = "   \n\n   \n\n   "
    result = PDFHelper._split_long_text(text, max_chars=100, overlap_chars=10)
    # all-whitespace → no non-empty pieces
    assert result == []


def test_chunk_pages_hard_split_never_emits_empty_piece():
    # Contrived: a page whose hard-split produces pieces that strip to empty.
    # _hard_split_text already guards this with the `not cleaned` check.
    # Verify chunk_pages itself also drops them (line 277 `continue`).
    pages = [PDFPageText(1, "  \n  ")]   # will be stripped to "" → page skipped
    chunks = PDFHelper().chunk_pages(pages, max_chars=10, overlap_chars=2)
    assert chunks == []


# ===========================================================================
# 10. get_pdf_helper factory
# ===========================================================================

def test_get_pdf_helper_returns_instance():
    assert isinstance(get_pdf_helper(), PDFHelper)


# ===========================================================================
# 10. Integration — all 41 real PDFs in /data
# ===========================================================================

class TestRealPDFs:
    """
    Parametrized over every PDF in /data (see conftest.py: real_pdf_path).
    Each test runs 41 times — once per document.
    """

    MAX_CHARS = 3500
    OVERLAP = 400

    def _get_chunks(self, real_pdf_path: str) -> list[PDFChunk]:
        h = PDFHelper()
        pages = h.extract_pages(real_pdf_path)
        return h.chunk_pages(pages, max_chars=self.MAX_CHARS, overlap_chars=self.OVERLAP)

    def test_pdf_yields_at_least_one_page(self, real_pdf_path):
        pages = PDFHelper().extract_pages(real_pdf_path)
        assert len(pages) > 0

    def test_page_numbers_start_at_one(self, real_pdf_path):
        pages = PDFHelper().extract_pages(real_pdf_path)
        assert pages[0].page_number == 1

    def test_page_numbers_sequential(self, real_pdf_path):
        pages = PDFHelper().extract_pages(real_pdf_path)
        nums = [p.page_number for p in pages]
        assert nums == list(range(1, len(pages) + 1))

    def test_no_null_bytes_in_pages(self, real_pdf_path):
        pages = PDFHelper().extract_pages(real_pdf_path)
        for p in pages:
            assert "\x00" not in p.text, f"Null byte on page {p.page_number}"

    def test_no_carriage_returns_in_pages(self, real_pdf_path):
        pages = PDFHelper().extract_pages(real_pdf_path)
        for p in pages:
            assert "\r" not in p.text, f"CR on page {p.page_number}"

    def test_no_excessive_newlines_in_pages(self, real_pdf_path):
        pages = PDFHelper().extract_pages(real_pdf_path)
        for p in pages:
            assert "\n\n\n" not in p.text, f"Triple newline on page {p.page_number}"

    def test_pdf_yields_at_least_one_chunk(self, real_pdf_path):
        chunks = self._get_chunks(real_pdf_path)
        assert len(chunks) > 0, "No chunks produced"

    def test_all_chunks_within_max_chars(self, real_pdf_path):
        chunks = self._get_chunks(real_pdf_path)
        over = [c for c in chunks if len(c.content) > self.MAX_CHARS]
        assert not over, (
            f"{len(over)} chunks exceed {self.MAX_CHARS} chars; "
            f"max={max(len(c.content) for c in chunks)}"
        )

    def test_no_empty_chunk_content(self, real_pdf_path):
        chunks = self._get_chunks(real_pdf_path)
        empty = [c for c in chunks if not c.content.strip()]
        assert not empty, f"{len(empty)} empty chunks"

    def test_chunk_indices_sequential(self, real_pdf_path):
        chunks = self._get_chunks(real_pdf_path)
        assert chunk_indices_sequential(chunks)

    def test_page_start_lte_page_end(self, real_pdf_path):
        chunks = self._get_chunks(real_pdf_path)
        bad = [c for c in chunks if c.page_start > c.page_end]
        assert not bad, f"{len(bad)} chunks have page_start > page_end"

    def test_page_numbers_within_document_range(self, real_pdf_path):
        pages = PDFHelper().extract_pages(real_pdf_path)
        total_pages = len(pages)
        chunks = PDFHelper().chunk_pages(
            pages, max_chars=self.MAX_CHARS, overlap_chars=self.OVERLAP
        )
        bad = [c for c in chunks
               if c.page_start < 1 or c.page_end > total_pages]
        assert not bad, f"{len(bad)} chunks have out-of-range page numbers"

    def test_typesense_docs_have_required_fields(self, real_pdf_path):
        h = PDFHelper()
        pages = h.extract_pages(real_pdf_path)
        chunks = h.chunk_pages(pages, max_chars=self.MAX_CHARS, overlap_chars=self.OVERLAP)
        stored = StoredPDF(
            pdf_id="test-id",
            original_filename=Path(real_pdf_path).name,
            stored_filename=Path(real_pdf_path).name,
            local_path=real_pdf_path,
            file_url="/pdfs/test-id/doc.pdf",
            sha256="test",
            size_bytes=0,
        )
        docs = h.build_typesense_chunk_documents(pdf=stored, chunks=chunks)
        required = {"id", "pdf_id", "pdf_name", "page", "page_start", "page_end",
                    "chunk_index", "content"}
        for doc in docs:
            missing = required - doc.keys()
            assert not missing, f"Doc missing fields: {missing}"

    def test_typesense_doc_page_field_matches_page_start(self, real_pdf_path):
        h = PDFHelper()
        pages = h.extract_pages(real_pdf_path)
        chunks = h.chunk_pages(pages, max_chars=self.MAX_CHARS, overlap_chars=self.OVERLAP)
        stored = StoredPDF("tid", Path(real_pdf_path).name, Path(real_pdf_path).name,
                           real_pdf_path, "/url", "sha", 0)
        docs = h.build_typesense_chunk_documents(pdf=stored, chunks=chunks)
        for doc in docs:
            assert doc["page"] == doc["page_start"], \
                f"page={doc['page']} != page_start={doc['page_start']}"

    def test_no_cr_in_chunk_content(self, real_pdf_path):
        chunks = self._get_chunks(real_pdf_path)
        for c in chunks:
            assert "\r" not in c.content, \
                f"CR in chunk {c.chunk_index} page {c.page_start}"


# ===========================================================================
# 11. Structure-aware (legal) chunking — the retrieval-precision fix
# ===========================================================================

class TestStructuralChunking:
    """Each Article/Section must land in its own chunk instead of a page-blob."""

    FOUR_ARTICLES = (
        "Article 19. Freedom of speech. "
        "All citizens shall have the right to freedom of speech and expression.\n"
        "Article 20. Protection for offences. "
        "No person shall be convicted except for violation of a law in force.\n"
        "Article 21. Protection of life and personal liberty. "
        "No person shall be deprived of his life or personal liberty except "
        "according to procedure established by law.\n"
        "Article 22. Protection against arrest. "
        "No arrested person shall be detained without being informed of the grounds."
    )

    def _art21_chunk(self, chunks: list[PDFChunk]) -> PDFChunk:
        matches = [c for c in chunks if "life or personal liberty" in c.content]
        assert len(matches) == 1, f"Article 21 not isolated: {len(matches)} chunks"
        return matches[0]

    def test_each_article_is_its_own_chunk(self):
        pages = make_pages(self.FOUR_ARTICLES)
        chunks = PDFHelper().chunk_pages(pages, max_chars=1200, overlap_chars=150)
        assert len(chunks) == 4
        assert chunk_indices_sequential(chunks)

    def test_article_21_isolated_from_neighbours(self):
        # The core regression: querying Article 21 must not surface a blob that
        # also contains Articles 19/20/22.
        pages = make_pages(self.FOUR_ARTICLES)
        chunks = PDFHelper().chunk_pages(pages, max_chars=1200, overlap_chars=150)
        c21 = self._art21_chunk(chunks)
        assert "Article 21" in c21.content
        assert "freedom of speech" not in c21.content   # Article 19
        assert "detained without" not in c21.content     # Article 22

    def test_provision_continuation_across_pages_stays_one_chunk(self):
        # Article 21 starts on page 1 and continues onto page 2 before Article 22.
        page1 = "Article 21. Protection of life and personal liberty. No person shall be deprived of his life"
        page2 = (
            "or personal liberty except according to procedure established by law.\n"
            "Article 22. Protection against arrest and detention."
        )
        pages = [PDFPageText(1, page1), PDFPageText(2, page2)]
        chunks = PDFHelper().chunk_pages(pages, max_chars=1200, overlap_chars=150)
        matches = [c for c in chunks if "Protection of life and personal liberty" in c.content]
        assert len(matches) == 1, f"Article 21 not isolated: {len(matches)} chunks"
        c21 = matches[0]
        assert c21.page_start == 1 and c21.page_end == 2
        assert "procedure established by law" in c21.content   # the continuation
        assert "arrest and detention" not in c21.content        # Article 22 stays out

    def test_no_headings_falls_back_to_size_accumulation(self):
        # Regression: heading-less text still merges short pages (old behaviour).
        pages = make_pages("Plain one.", "Plain two.", "Plain three.")
        chunks = PDFHelper().chunk_pages(pages, max_chars=1200, overlap_chars=0)
        assert len(chunks) == 1
        assert chunks[0].page_start == 1 and chunks[0].page_end == 3

    def test_long_article_split_keeps_heading_on_every_piece(self):
        long_article = (
            "Article 42. Just and humane conditions of work. "
            + "The State shall make provision for securing conditions of work. " * 40
        )
        pages = make_pages(long_article)
        chunks = PDFHelper().chunk_pages(pages, max_chars=1000, overlap_chars=100)
        assert len(chunks) >= 2
        assert all(len(c.content) <= 1000 for c in chunks)
        assert all("Article 42" in c.content for c in chunks)

    def test_bare_numbered_provisions_detected(self):
        # Constitution-style "21. Title.—text" (no "Article" keyword).
        text = (
            "20. Protection for offences. Clause text here.\n"
            "21. Protection of life and personal liberty. No person shall be deprived."
        )
        segments = PDFHelper._segment_pages([PDFPageText(1, text)])
        headings = [s for s in segments if s[2]]
        assert len(headings) == 2

    def test_cross_reference_and_plural_not_treated_as_heading(self):
        # Plurals and mid-line references must NOT create provision boundaries.
        text = "Articles 12 to 35 are fundamental rights guaranteed under Section 3 here."
        segments = PDFHelper._segment_pages([PDFPageText(1, text)])
        assert [(s.text, s.page_number, s.is_heading) for s in segments] == [
            (text, 1, False)
        ]


# ===========================================================================
# 13. Character spans (citation anchors) + retrieval/display separation
# ===========================================================================

from helpers.pdf_helper import build_retrieval_text  # noqa: E402


class TestBuildDocumentText:
    def test_joins_stripped_pages_with_blank_line(self):
        pages = make_pages("  First page.  ", "Second page.")
        assert PDFHelper.build_document_text(pages) == "First page.\n\nSecond page."

    def test_drops_empty_pages(self):
        pages = make_pages("First.", "   ", "Third.")
        assert PDFHelper.build_document_text(pages) == "First.\n\nThird."

    def test_empty_input(self):
        assert PDFHelper.build_document_text([]) == ""


class TestChunkSpans:
    """Every chunk's (start_char, end_char) must resolve into
    build_document_text(pages) at exactly the material it was cut from."""

    def test_single_page_single_chunk_spans_whole_page(self):
        pages = make_pages("A short provision that fits in one chunk.")
        doc = PDFHelper.build_document_text(pages)
        [chunk] = PDFHelper().chunk_pages(pages, max_chars=500, overlap_chars=0)
        assert (chunk.start_char, chunk.end_char) == (0, len(doc))
        assert chunk.resolve(doc) == chunk.content

    def test_second_page_span_accounts_for_page_join(self):
        p1, p2 = "First page text." * 5, "Second page text." * 5
        pages = make_pages(p1, p2)
        doc = PDFHelper.build_document_text(pages)
        chunks = PDFHelper().chunk_pages(pages, max_chars=90, overlap_chars=0)
        assert len(chunks) == 2
        assert chunks[1].resolve(doc) == p2

    def test_pages_merged_into_one_chunk_span_covers_both(self):
        pages = make_pages("Short first.", "Short second.")
        doc = PDFHelper.build_document_text(pages)
        [chunk] = PDFHelper().chunk_pages(pages, max_chars=500, overlap_chars=0)
        assert chunk.resolve(doc) == doc == chunk.content

    def test_heading_segments_resolve_exactly(self):
        text = (
            "Preamble text introducing the statute in a sentence.\n"
            "Article 1. Short title. This Act may be cited accordingly.\n"
            "Article 2. Commencement. It shall come into force at once."
        )
        pages = make_pages(text)
        doc = PDFHelper.build_document_text(pages)
        chunks = PDFHelper().chunk_pages(pages, max_chars=70, overlap_chars=0)
        assert len(chunks) == 3  # preamble + one per article
        for chunk in chunks:
            assert chunk.resolve(doc) == chunk.content

    def test_overlap_text_is_excluded_from_span(self):
        p1, p2 = "First page text. " * 6, "Second page text. " * 6
        pages = make_pages(p1, p2)
        doc = PDFHelper.build_document_text(pages)
        chunks = PDFHelper().chunk_pages(pages, max_chars=150, overlap_chars=30)
        assert len(chunks) == 2
        # The second chunk's content is seeded with the first chunk's tail…
        assert chunks[1].content != p2.strip()
        # …but its span cites only its own source region.
        assert chunks[1].resolve(doc) == p2.strip()

    def test_sub_split_pieces_resolve_to_their_own_region(self):
        paragraphs = [f"Paragraph number {i} with some words in it." for i in range(12)]
        pages = make_pages("\n\n".join(paragraphs))
        doc = PDFHelper.build_document_text(pages)
        chunks = PDFHelper().chunk_pages(pages, max_chars=120, overlap_chars=0)
        assert len(chunks) > 2
        for chunk in chunks:
            resolved = chunk.resolve(doc)
            assert resolved  # every piece located
            assert resolved.startswith(chunk.content[:40])

    def test_spans_are_monotonic_and_in_bounds(self):
        text = "\n".join(
            f"Article {i}. Provision {i}. Body of provision number {i}."
            for i in range(1, 8)
        )
        pages = make_pages(text)
        doc = PDFHelper.build_document_text(pages)
        chunks = PDFHelper().chunk_pages(pages, max_chars=100, overlap_chars=0)
        for chunk in chunks:
            assert 0 <= chunk.start_char < chunk.end_char <= len(doc)
        starts = [c.start_char for c in chunks]
        assert starts == sorted(starts)

    def test_hand_built_chunk_defaults_to_sentinel_and_empty_resolve(self):
        chunk = PDFChunk(0, 1, 1, "legacy chunk")
        assert (chunk.start_char, chunk.end_char) == (-1, -1)
        assert chunk.resolve("whatever document text") == ""


class TestRetrievalDisplaySeparation:
    """The plan's non-negotiable: retrieval_text (summary + body, embedded)
    and display_text (body only, cited) must never be conflated."""

    def test_display_text_is_body_only(self):
        chunk = PDFChunk(0, 1, 1, "No person shall be deprived of life.")
        assert chunk.display_text == chunk.content
        assert "summary" not in chunk.display_text.lower()

    def test_retrieval_text_prepends_summary_for_embedding_only(self):
        body = "No person shall be deprived of life."
        summary = "The Constitution of India — fundamental rights."
        retrieval = build_retrieval_text(body, summary)
        assert retrieval == f"{summary}\n{body}"
        # And the displayed/cited text is unchanged by the summary's existence.
        assert PDFChunk(0, 1, 1, body).display_text == body

    def test_retrieval_text_without_summary_is_body(self):
        assert build_retrieval_text("body", None) == "body"
        assert build_retrieval_text("body", "   ") == "body"

    def test_typesense_documents_keep_summary_out_of_content(self):
        stored = StoredPDF(
            pdf_id="abc", original_filename="c.pdf", stored_filename="c.pdf",
            local_path="/s/c.pdf", file_url="/pdfs/abc/c.pdf",
            sha256="beef", size_bytes=10,
        )
        chunks = [PDFChunk(0, 1, 1, "Body text.", start_char=5, end_char=15)]
        [doc] = PDFHelper().build_typesense_chunk_documents(
            pdf=stored, chunks=chunks, summary="A doc summary."
        )
        assert doc["content"] == "Body text."          # cited text: body only
        assert doc["summary"] == "A doc summary."      # stored separately
        assert doc["start_char"] == 5 and doc["end_char"] == 15
        # The embedded form is reconstructable without touching content.
        assert build_retrieval_text(doc["content"], doc["summary"]) == \
            "A doc summary.\nBody text."
