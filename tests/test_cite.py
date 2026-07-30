"""services/cite.py — span label extraction, citation binding, coverage."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from models.schemas import ChunkHit
from services import cite


def _hit(i=0, **kw):
    return ChunkHit(
        id=f"c{i}", pdf_id=f"p{i}", pdf_name=f"doc{i}.pdf",
        page_start=1, page_end=1, chunk_index=i, content=f"content {i}",
        start_char=kw.get("start_char", i * 100),
        end_char=kw.get("end_char", i * 100 + 50),
    )


# --- span_id_for -----------------------------------------------------------

def test_span_id_for_is_one_based():
    assert cite.span_id_for(0) == "S1"
    assert cite.span_id_for(4) == "S5"


# --- extract_span_ids ------------------------------------------------------

class TestExtractSpanIds:
    def test_single_label(self):
        assert cite.extract_span_ids("Liability is capped [S1].") == ["S1"]

    def test_comma_list(self):
        assert cite.extract_span_ids("Both apply [S1, S3].") == ["S1", "S3"]

    def test_adjacent_labels(self):
        assert cite.extract_span_ids("Twice over [S2][S4].") == ["S2", "S4"]

    def test_dedupes_preserving_first_appearance_order(self):
        assert cite.extract_span_ids("[S2] then [S1] then [S2]") == ["S2", "S1"]

    def test_case_insensitive(self):
        assert cite.extract_span_ids("lower [s3] case") == ["S3"]

    def test_no_labels(self):
        assert cite.extract_span_ids("No labels at all.") == []

    def test_ignores_non_span_brackets(self):
        assert cite.extract_span_ids("[ipc.pdf p.7] and [Section 5]") == []


# --- bind_citations --------------------------------------------------------

class TestBindCitations:
    def test_binds_labels_to_hits_in_cited_order(self):
        hits = [_hit(0), _hit(1), _hit(2)]
        answer = "The cap applies [S2]. It excludes fraud [S1]."
        citations, coverage = cite.bind_citations(answer, hits)
        assert [c.span_id for c in citations] == ["S2", "S1"]
        assert citations[0].pdf_id == "p1"
        assert citations[1].pdf_id == "p0"
        assert coverage == 1.0

    def test_carries_char_span_onto_citation(self):
        hits = [_hit(0, start_char=10, end_char=60)]
        citations, _ = cite.bind_citations("A claim with enough length [S1].", hits)
        assert citations[0].start_char == 10
        assert citations[0].end_char == 60

    def test_invented_label_is_dropped_not_guessed(self):
        hits = [_hit(0)]
        answer = "Supported claim here [S1]. Fabricated support there [S9]."
        citations, _ = cite.bind_citations(answer, hits)
        assert [c.span_id for c in citations] == ["S1"]

    def test_unlabelled_answer_returns_all_hits_with_zero_coverage(self):
        hits = [_hit(0), _hit(1)]
        answer = "A fluent uncited answer that makes several factual claims."
        citations, coverage = cite.bind_citations(answer, hits)
        assert len(citations) == 2
        assert all(c.span_id is None for c in citations)
        assert coverage == 0.0

    def test_only_invented_labels_falls_back_to_unlabelled(self):
        hits = [_hit(0)]
        citations, coverage = cite.bind_citations("Uncheckable claim [S7].", hits)
        assert all(c.span_id is None for c in citations)
        assert coverage == 0.0


# --- citation_coverage -----------------------------------------------------

class TestCitationCoverage:
    def test_all_sentences_cited(self):
        answer = "The act commenced in 1956 [S1]. It applies nationwide [S2]."
        assert cite.citation_coverage(answer) == 1.0

    def test_half_cited(self):
        answer = (
            "The act commenced in 1956 [S1]. "
            "It repealed the earlier ordinance without any citation."
        )
        assert cite.citation_coverage(answer) == 0.5

    def test_empty_answer_is_not_penalised(self):
        assert cite.citation_coverage("") == 1.0

    def test_short_fragments_are_not_claims(self):
        # Headings/list stubs under the length floor need no citation.
        assert cite.citation_coverage("Short. Tiny. Also small.") == 1.0

    def test_single_uncited_claim_is_zero(self):
        answer = "This is a substantive factual claim carrying no label at all."
        assert cite.citation_coverage(answer) == 0.0
