import sys
from pathlib import Path

# Make backend importable without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pytest

from helpers.pdf_helper import PDFHelper, PDFPageText, PDFChunk, StoredPDF


DATA_DIR = Path(__file__).parent.parent / "data"
ALL_PDFS = sorted(DATA_DIR.glob("*.pdf"))


def _pdf_is_readable(path: Path) -> bool:
    """Return True if PyMuPDF can open the PDF and count its pages."""
    try:
        PDFHelper().extract_pages(path)
        return True
    except (ValueError, FileNotFoundError):
        return False


# Separate readable PDFs from corrupted ones so tests can target each group
READABLE_PDFS = [p for p in ALL_PDFS if _pdf_is_readable(p)]
CORRUPTED_PDFS = [p for p in ALL_PDFS if not _pdf_is_readable(p)]


@pytest.fixture
def tmp_storage(tmp_path):
    """PDFHelper with an isolated temp storage dir."""
    return PDFHelper(storage_dir=tmp_path / "pdfs")


@pytest.fixture
def helper():
    """Default PDFHelper (not used for file writes in unit tests)."""
    return PDFHelper.__new__(PDFHelper)


@pytest.fixture(
    params=[str(p) for p in READABLE_PDFS],
    ids=[p.name for p in READABLE_PDFS],
)
def real_pdf_path(request):
    """Parametrized fixture: one readable PDF per test invocation."""
    return request.param


@pytest.fixture(
    params=[str(p) for p in CORRUPTED_PDFS],
    ids=[p.name for p in CORRUPTED_PDFS],
)
def corrupted_pdf_path(request):
    """Parametrized fixture: one known-corrupted PDF per test invocation."""
    return request.param
