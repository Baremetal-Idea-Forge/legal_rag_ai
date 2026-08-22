"""RagService — retrieval+generation orchestration (fakes for search + Ollama)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import json

from core.config import Settings, get_settings
from models.schemas import ChunkHit, DocumentScope, SearchResponse, VerificationScores
from services import prompts
from services.audit import AuditLog
from services.rag_service import RagService


# --- Fakes -----------------------------------------------------------------

class FakeSearch:
    def __init__(self, hits):
        self._hits = hits
        self.calls = 0
        self.last_mode = None

    def search(self, query, *, top_k=5, mode="hybrid", filter_by=None):
        self.calls += 1
        self.last_mode = mode
        return SearchResponse(query=query, mode=mode, count=len(self._hits), hits=self._hits)


class FakeOllama:
    def __init__(self, answer="Grounded answer.", tokens=None):
        self._answer = answer
        self._tokens = tokens or ["Grounded ", "answer."]
        self.chat_calls = 0

    async def chat(self, messages, *, temperature=0.2):
        self.chat_calls += 1
        self._last_messages = messages
        self.last_temperature = temperature
        return self._answer

    async def stream_chat(self, messages, *, temperature=0.2):
        self.last_temperature = temperature
        for t in self._tokens:
            yield t


def _hit(i=0, content="legal content", pages=(1, 1), dense_score=None):
    """dense_score also sets vector_distance so Gate 1 sees a dense plane."""
    return ChunkHit(
        id=f"c{i}", pdf_id=f"p{i}", pdf_name=f"doc{i}.pdf",
        page_start=pages[0], page_end=pages[1], chunk_index=i, content=content,
        file_url=f"/pdfs/p{i}/doc{i}.pdf",
        score=dense_score,
        vector_distance=None if dense_score is None else round(1.0 - dense_score, 6),
    )


def _service(hits, ollama=None):
    search = FakeSearch(hits)
    ollama = ollama or FakeOllama()
    svc = RagService(
        search_service=search, llm_client=ollama, settings=get_settings()
    )
    return svc, search, ollama


# --- answer ----------------------------------------------------------------

def test_answer_with_hits_returns_cited_response():
    hits = [_hit(0), _hit(1)]
    svc, search, ollama = _service(hits)
    resp = asyncio.run(svc.answer("What is negligence?"))

    assert resp.answer == "Grounded answer."
    assert len(resp.citations) == 2
    assert len(resp.chunks_used) == 2
    assert resp.citations[0].pdf_name == "doc0.pdf"
    assert ollama.chat_calls == 1
    assert search.calls == 1


def test_answer_no_hits_declines_without_calling_llm():
    svc, search, ollama = _service([])
    resp = asyncio.run(svc.answer("Unknown topic?"))

    assert resp.answer == prompts.NO_CONTEXT_ANSWER
    assert resp.citations == []
    assert resp.chunks_used == []
    assert ollama.chat_calls == 0  # no hallucination path


def test_answer_forwards_configured_temperature():
    # Deterministic answers: RagService must pass settings.LLM_TEMPERATURE
    # (not the client's own 0.2 default) to the LLM.
    settings = Settings()
    settings.LLM_TEMPERATURE = 0.0
    ollama = FakeOllama()
    svc = RagService(
        search_service=FakeSearch([_hit(0)]), llm_client=ollama, settings=settings
    )
    asyncio.run(svc.answer("q"))
    assert ollama.last_temperature == 0.0


def test_stream_forwards_configured_temperature():
    settings = Settings()
    settings.LLM_TEMPERATURE = 0.0
    ollama = FakeOllama()
    svc = RagService(
        search_service=FakeSearch([_hit(0)]), llm_client=ollama, settings=settings
    )
    _collect(svc.stream_answer("q"))
    assert ollama.last_temperature == 0.0


def test_answer_passes_context_to_llm():
    hits = [_hit(0, content="duty of care principle")]
    svc, _, ollama = _service(hits)
    asyncio.run(svc.answer("q"))
    system, user = ollama._last_messages
    assert "duty of care principle" in user["content"]


def test_answer_context_truncated_to_budget():
    # Many oversized chunks; only those fitting RAG_MAX_CONTEXT_CHARS are cited.
    big = "x" * 5000
    hits = [_hit(i, content=big) for i in range(10)]
    svc, _, _ = _service(hits)
    resp = asyncio.run(svc.answer("q"))
    assert len(resp.chunks_used) < 10
    assert len(resp.citations) == len(resp.chunks_used)


# --- stream_answer ---------------------------------------------------------

def _collect(aiter):
    async def run():
        return [e async for e in aiter]
    return asyncio.run(run())


def test_stream_yields_tokens_then_citations():
    hits = [_hit(0)]
    svc, _, _ = _service(hits, ollama=FakeOllama(tokens=["Hello", " there"]))
    events = _collect(svc.stream_answer("q"))

    token_events = [e for e in events if e["type"] == "token"]
    citation_events = [e for e in events if e["type"] == "citations"]
    assert [e["data"] for e in token_events] == ["Hello", " there"]
    assert len(citation_events) == 1
    assert citation_events[-1]["data"][0]["pdf_name"] == "doc0.pdf"


def test_stream_no_hits_emits_decline_and_empty_citations():
    svc, _, ollama = _service([])
    events = _collect(svc.stream_answer("q"))
    tokens = [e["data"] for e in events if e["type"] == "token"]
    citations = [e for e in events if e["type"] == "citations"][0]
    assert tokens == [prompts.NO_CONTEXT_ANSWER]
    assert citations["data"] == []


# --- RETRIEVAL_VIA_MCP flag ------------------------------------------------

class FakeMcp:
    def __init__(self, hits):
        self._hits = hits
        self.calls = 0

    async def search(self, query, *, top_k=5, mode="hybrid", filter_by=None):
        self.calls += 1
        return self._hits


def _settings_with_mcp(enabled: bool) -> Settings:
    s = Settings()
    s.RETRIEVAL_VIA_MCP = enabled
    return s


def test_retrieval_via_mcp_uses_mcp_client_not_search():
    hits = [_hit(0)]
    search = FakeSearch([])          # would yield nothing if (wrongly) used
    mcp = FakeMcp(hits)
    svc = RagService(
        search_service=search, llm_client=FakeOllama(),
        settings=_settings_with_mcp(True), mcp_client=mcp,
    )
    resp = asyncio.run(svc.answer("q"))
    assert mcp.calls == 1
    assert search.calls == 0
    assert len(resp.chunks_used) == 1


def test_retrieval_direct_when_flag_disabled():
    hits = [_hit(0)]
    search = FakeSearch(hits)
    mcp = FakeMcp([])
    svc = RagService(
        search_service=search, llm_client=FakeOllama(),
        settings=_settings_with_mcp(False), mcp_client=mcp,
    )
    resp = asyncio.run(svc.answer("q"))
    assert search.calls == 1
    assert mcp.calls == 0
    assert len(resp.chunks_used) == 1


def test_flag_enabled_but_no_mcp_client_falls_back_to_search():
    hits = [_hit(0)]
    search = FakeSearch(hits)
    svc = RagService(
        search_service=search, llm_client=FakeOllama(),
        settings=_settings_with_mcp(True), mcp_client=None,
    )
    resp = asyncio.run(svc.answer("q"))
    assert search.calls == 1
    assert len(resp.chunks_used) == 1


# ===========================================================================
# Two-stage retrieval + abstention (TWO_STAGE_ENABLED)
# ===========================================================================

class FakeTwoStageSearch:
    def __init__(self, hits, scopes):
        self._hits = hits
        self._scopes = scopes
        self.two_stage_calls = 0
        self.search_calls = 0

    def search_two_stage(self, query, *, top_k=5, mode="hybrid", filter_by=None):
        self.two_stage_calls += 1
        resp = SearchResponse(
            query=query, mode=mode, count=len(self._hits), hits=self._hits
        )
        return resp, self._scopes

    def search(self, query, *, top_k=5, mode="hybrid", filter_by=None):
        self.search_calls += 1
        return SearchResponse(query=query, mode=mode, count=0, hits=[])


def _scope(pdf_id="p0", score=0.9):
    return DocumentScope(pdf_id=pdf_id, pdf_name=f"{pdf_id}.pdf", score=score,
                         chunk_count=1)


def _two_stage_settings(**overrides) -> Settings:
    s = Settings()
    s.TWO_STAGE_ENABLED = True
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def test_two_stage_flag_routes_retrieval_and_reports_scopes():
    scopes = [_scope("p0", 0.9), _scope("p1", 0.7)]
    search = FakeTwoStageSearch([_hit(0)], scopes)
    svc = RagService(
        search_service=search, llm_client=FakeOllama(),
        settings=_two_stage_settings(),
    )
    resp = asyncio.run(svc.answer("q"))
    assert search.two_stage_calls == 1
    assert search.search_calls == 0
    assert resp.scoped_documents == scopes
    assert resp.abstained is False


def test_two_stage_no_candidates_abstains_without_llm():
    search = FakeTwoStageSearch([], [])
    ollama = FakeOllama()
    svc = RagService(
        search_service=search, llm_client=ollama, settings=_two_stage_settings()
    )
    resp = asyncio.run(svc.answer("q"))
    assert resp.abstained is True
    assert resp.answer == prompts.NO_CONTEXT_ANSWER
    assert "stage 1" in resp.abstain_reason
    assert ollama.chat_calls == 0


def test_two_stage_score_below_floor_abstains_without_llm():
    scopes = [_scope("p0", 0.3)]
    search = FakeTwoStageSearch([_hit(0)], scopes)
    ollama = FakeOllama()
    svc = RagService(
        search_service=search, llm_client=ollama,
        settings=_two_stage_settings(ABSTAIN_MIN_DOC_SCORE=0.5),
    )
    resp = asyncio.run(svc.answer("q"))
    assert resp.abstained is True
    assert "abstention floor" in resp.abstain_reason
    assert resp.scoped_documents == scopes  # what stage 1 saw stays auditable
    assert ollama.chat_calls == 0


def test_no_context_response_is_marked_abstained():
    svc, _, _ = _service([])
    resp = asyncio.run(svc.answer("q"))
    assert resp.abstained is True
    assert resp.abstain_reason


# ===========================================================================
# Citation binding (span labels in the answer)
# ===========================================================================

def test_answer_binds_span_labelled_citations():
    hits = [_hit(0), _hit(1)]
    answer = "The liability cap is one lakh rupees [S2]. Fraud is excluded [S1]."
    svc, _, _ = _service(hits, ollama=FakeOllama(answer=answer))
    resp = asyncio.run(svc.answer("q"))
    assert [c.span_id for c in resp.citations] == ["S2", "S1"]
    assert resp.citations[0].pdf_name == "doc1.pdf"
    assert resp.citation_coverage == 1.0


def test_uncited_answer_reports_zero_coverage_with_all_sources():
    hits = [_hit(0), _hit(1)]
    answer = "A fluent answer that carries no span labels anywhere at all."
    svc, _, _ = _service(hits, ollama=FakeOllama(answer=answer))
    resp = asyncio.run(svc.answer("q"))
    assert resp.citation_coverage == 0.0
    assert len(resp.citations) == 2  # sources still attached for review
    assert all(c.span_id is None for c in resp.citations)


def test_stream_binds_citations_from_streamed_text():
    hits = [_hit(0), _hit(1)]
    tokens = ["The cap is one lakh rupees", " [S1]."]
    svc, _, _ = _service(hits, ollama=FakeOllama(tokens=tokens))
    events = _collect(svc.stream_answer("q"))
    citations = [e for e in events if e["type"] == "citations"][0]["data"]
    assert len(citations) == 1
    assert citations[0]["span_id"] == "S1"
    assert citations[0]["pdf_name"] == "doc0.pdf"


# ===========================================================================
# Verification triad gate (VERIFY_ENABLED)
# ===========================================================================

class QueueLLM:
    """Replies in order; records every message list sent."""

    def __init__(self, *replies):
        self._replies = list(replies)
        self.chat_calls = 0
        self.messages_log = []

    async def chat(self, messages, *, temperature=0.2):
        self.chat_calls += 1
        self.messages_log.append(messages)
        return self._replies.pop(0) if self._replies else "fallback"


class FakeVerifier:
    def __init__(self, *scores):
        self._scores = list(scores)
        self.calls = 0

    async def verify(self, *, query, context, answer):
        self.calls += 1
        return self._scores.pop(0) if self._scores else None


def _triad(minimum: float) -> VerificationScores:
    return VerificationScores(
        context_relevance=0.9, groundedness=minimum, answer_relevance=0.9
    )


def _verify_settings(**overrides) -> Settings:
    s = Settings()
    s.VERIFY_ENABLED = True
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def test_verified_answer_passes_gate_first_try():
    verifier = FakeVerifier(_triad(0.8))
    ollama = FakeOllama(answer="Verified claim with support [S1].")
    svc = RagService(
        search_service=FakeSearch([_hit(0)]), llm_client=ollama,
        settings=_verify_settings(), verifier=verifier,
    )
    resp = asyncio.run(svc.answer("q"))
    assert resp.abstained is False
    assert resp.verification is not None
    assert resp.verification.minimum == 0.8
    assert verifier.calls == 1
    assert ollama.chat_calls == 1  # no rewrite happened


def test_gate_failure_abstains_first_try_without_regeneration():
    # v2 non-goal R8: an answer that failed the triad is declined, never
    # regenerated — abstention is cheaper and more honest than a retry loop.
    verifier = FakeVerifier(_triad(0.2))
    llm = QueueLLM("Unverifiable answer, drawn from thin air entirely [S1].")
    search = FakeSearch([_hit(0)])
    svc = RagService(
        search_service=search, llm_client=llm,
        settings=_verify_settings(), verifier=verifier,
    )
    resp = asyncio.run(svc.answer("q"))

    assert resp.abstained is True
    assert resp.answer == prompts.UNVERIFIED_ANSWER
    assert "verification" in resp.abstain_reason
    assert resp.verification is not None and resp.verification.minimum == 0.2
    assert len(resp.citations) == 1  # best-effort spans attached
    assert resp.citation_coverage == 0.0
    assert verifier.calls == 1
    assert llm.chat_calls == 1   # one generation; no rewrite, no regeneration
    assert search.calls == 1     # no re-retrieval either


def test_judge_failure_skips_gate_instead_of_blocking():
    verifier = FakeVerifier(None)
    svc = RagService(
        search_service=FakeSearch([_hit(0)]), llm_client=FakeOllama(),
        settings=_verify_settings(), verifier=verifier,
    )
    resp = asyncio.run(svc.answer("q"))
    assert resp.abstained is False
    assert resp.verification is None


def test_verification_off_never_calls_verifier():
    verifier = FakeVerifier(_triad(0.0))
    svc = RagService(
        search_service=FakeSearch([_hit(0)]), llm_client=FakeOllama(),
        settings=Settings(), verifier=verifier,
    )
    resp = asyncio.run(svc.answer("q"))
    assert verifier.calls == 0
    assert resp.verification is None


# ===========================================================================
# L6 exact-term routing (EXACT_TERM_ROUTING_ENABLED)
# ===========================================================================

def _routing_settings() -> Settings:
    s = Settings()
    s.EXACT_TERM_ROUTING_ENABLED = True
    return s


def test_routing_uses_dense_for_semantic_queries():
    search = FakeSearch([_hit(0)])
    svc = RagService(
        search_service=search, llm_client=FakeOllama(), settings=_routing_settings()
    )
    resp = asyncio.run(svc.answer("what are the remedies for breach of contract"))
    assert search.last_mode == "vector"
    assert resp.retrieval_mode == "vector"


def test_routing_fires_lexical_on_exact_term_query():
    search = FakeSearch([_hit(0)])
    svc = RagService(
        search_service=search, llm_client=FakeOllama(), settings=_routing_settings()
    )
    resp = asyncio.run(svc.answer("What does Section 302 of the IPC say?"))
    assert search.last_mode == "hybrid"
    assert resp.retrieval_mode == "hybrid"


def test_routing_off_keeps_hybrid_for_every_query():
    search = FakeSearch([_hit(0)])
    svc = RagService(
        search_service=search, llm_client=FakeOllama(), settings=Settings()
    )
    resp = asyncio.run(svc.answer("what are the remedies for breach of contract"))
    assert search.last_mode == "hybrid"
    assert resp.retrieval_mode == "hybrid"


# ===========================================================================
# Gate 1 — retrieval-score floor (GATE1_MIN_SCORE)
# ===========================================================================

class SequencedSearch:
    """Returns the next hit list on each search() call (Gate-1 retry paths)."""

    def __init__(self, *hit_lists):
        self._lists = list(hit_lists)
        self.calls = 0
        self.queries = []

    def search(self, query, *, top_k=5, mode="hybrid", filter_by=None):
        self.calls += 1
        self.queries.append(query)
        hits = self._lists.pop(0) if self._lists else []
        return SearchResponse(query=query, mode=mode, count=len(hits), hits=hits)


def _gate1_settings(floor: float, *, retry: bool = False) -> Settings:
    s = Settings()
    s.GATE1_MIN_SCORE = floor
    s.GATE1_REWRITE_RETRY = retry
    return s


def test_gate1_below_floor_abstains_without_llm_and_attaches_nearest_docs():
    hits = [_hit(0, dense_score=0.3), _hit(1, dense_score=0.2)]
    ollama = FakeOllama()
    svc = RagService(
        search_service=FakeSearch(hits), llm_client=ollama,
        settings=_gate1_settings(0.5),
    )
    resp = asyncio.run(svc.answer("q"))

    assert resp.abstained is True
    assert resp.answer == prompts.NO_CONTEXT_ANSWER
    assert "Gate-1" in resp.abstain_reason
    assert ollama.chat_calls == 0  # zero-LLM-call abstention
    # Nearest documents attached for review, best first.
    assert [d.pdf_id for d in resp.scoped_documents] == ["p0", "p1"]


def test_gate1_above_floor_answers_normally():
    svc = RagService(
        search_service=FakeSearch([_hit(0, dense_score=0.8)]),
        llm_client=FakeOllama(), settings=_gate1_settings(0.5),
    )
    resp = asyncio.run(svc.answer("q"))
    assert resp.abstained is False


def test_gate1_rewrite_retry_clears_gate():
    search = SequencedSearch(
        [_hit(0, dense_score=0.3)], [_hit(1, dense_score=0.9)]
    )
    llm = QueueLLM("sharper rewritten query", "Grounded answer [S1].")
    svc = RagService(
        search_service=search, llm_client=llm,
        settings=_gate1_settings(0.5, retry=True),
    )
    resp = asyncio.run(svc.answer("vague query"))

    assert resp.abstained is False
    assert resp.answer == "Grounded answer [S1]."
    assert search.calls == 2
    assert search.queries[1] == "sharper rewritten query"
    assert llm.chat_calls == 2  # one rewrite + one generation
    assert resp.chunks_used[0].pdf_id == "p1"  # answered from the retry hits


def test_gate1_rewrite_retry_still_weak_abstains():
    search = SequencedSearch(
        [_hit(0, dense_score=0.3)], [_hit(1, dense_score=0.4)]
    )
    llm = QueueLLM("sharper rewritten query")
    svc = RagService(
        search_service=search, llm_client=llm,
        settings=_gate1_settings(0.5, retry=True),
    )
    resp = asyncio.run(svc.answer("vague query"))

    assert resp.abstained is True
    assert "Gate-1" in resp.abstain_reason
    assert llm.chat_calls == 1  # the rewrite only — never a generation


def test_gate1_skipped_when_hits_carry_no_dense_score():
    # Keyword-only hits have no comparable score plane; the gate must not
    # abstain on them.
    svc = RagService(
        search_service=FakeSearch([_hit(0)]), llm_client=FakeOllama(),
        settings=_gate1_settings(0.5),
    )
    resp = asyncio.run(svc.answer("q"))
    assert resp.abstained is False


def test_gate1_disabled_at_zero_floor():
    svc = RagService(
        search_service=FakeSearch([_hit(0, dense_score=0.01)]),
        llm_client=FakeOllama(), settings=Settings(),
    )
    resp = asyncio.run(svc.answer("q"))
    assert resp.abstained is False


# ===========================================================================
# Audit log (AUDIT_LOG_ENABLED)
# ===========================================================================

def _audit_settings() -> Settings:
    s = Settings()
    s.AUDIT_LOG_ENABLED = True
    return s


def test_answer_writes_audit_record(tmp_path):
    path = tmp_path / "audit.jsonl"
    svc = RagService(
        search_service=FakeSearch([_hit(0)]),
        llm_client=FakeOllama(answer="An answer with a citation [S1]."),
        settings=_audit_settings(), audit_log=AuditLog(path),
    )
    asyncio.run(svc.answer("what is the cap?"))

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["query"] == "what is the cap?"
    assert record["answer"] == "An answer with a citation [S1]."
    assert record["abstained"] is False
    assert record["chunks"][0]["pdf_id"] == "p0"
    assert record["retrieval_mode"] == "hybrid"  # routing off → pre-v2 path
    assert record["plan"]["exact_term_routing"] is False
    assert record["plan"]["gate1_min_score"] == 0.0
    assert record["model"]["provider"]
    assert "ts" in record


def test_abstention_is_audited_too(tmp_path):
    path = tmp_path / "audit.jsonl"
    svc = RagService(
        search_service=FakeSearch([]), llm_client=FakeOllama(),
        settings=_audit_settings(), audit_log=AuditLog(path),
    )
    asyncio.run(svc.answer("q"))
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["abstained"] is True
    assert record["abstain_reason"]


def test_audit_disabled_writes_nothing(tmp_path):
    path = tmp_path / "audit.jsonl"
    svc = RagService(
        search_service=FakeSearch([_hit(0)]), llm_client=FakeOllama(),
        settings=Settings(), audit_log=AuditLog(path),
    )
    asyncio.run(svc.answer("q"))
    assert not path.exists()
