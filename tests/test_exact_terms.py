"""Exact-term detector — the L6 routing signal (rule-based, no model)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from services.exact_terms import detect_exact_terms


# --- quoted phrases ---------------------------------------------------------

def test_double_quoted_phrase_flags():
    flag = detect_exact_terms('what does "consequential damages" mean here')
    assert flag.flagged
    assert "quoted_phrase" in flag.reasons


def test_curly_quoted_phrase_flags():
    flag = detect_exact_terms("define “force majeure” in this contract")
    assert flag.flagged
    assert "quoted_phrase" in flag.reasons


def test_single_stray_quote_does_not_flag():
    assert not detect_exact_terms('what is a "x plea').flagged


# --- section / article numbers ---------------------------------------------

def test_section_number_flags():
    flag = detect_exact_terms("What does Section 302 of the IPC say?")
    assert flag.flagged
    assert "section_number" in flag.reasons


def test_section_symbol_flags():
    assert detect_exact_terms("liability under § 12").flagged


def test_article_abbreviation_flags():
    assert detect_exact_terms("scope of art. 21").flagged


def test_article_with_suffix_flags():
    assert detect_exact_terms("what is article 21A about").flagged


def test_order_roman_numeral_flags():
    flag = detect_exact_terms("execution under Order XXI rule 4")
    assert flag.flagged
    assert "section_number" in flag.reasons


def test_lowercase_roman_after_keyword_does_not_flag():
    # "order i placed" — a lowercase roman numeral is ordinary prose.
    assert not detect_exact_terms("the order i placed last week").flagged


def test_keyword_without_number_does_not_flag():
    assert not detect_exact_terms("which section covers murder").flagged


# --- defined terms ----------------------------------------------------------

def test_capitalized_run_mid_query_flags():
    flag = detect_exact_terms("remedies under the Sale of Goods Act")
    assert flag.flagged
    assert "defined_term" in flag.reasons


def test_capitalized_run_at_start_does_not_flag():
    # Sentence-initial capitalization is not a defined-term signal.
    assert not detect_exact_terms("Criminal Procedure overview please").flagged


def test_single_capitalized_word_does_not_flag():
    assert not detect_exact_terms("what does the Constitution guarantee").flagged


# --- plain queries ----------------------------------------------------------

def test_plain_semantic_query_does_not_flag():
    flag = detect_exact_terms("what are the remedies for breach of contract")
    assert not flag.flagged
    assert flag.reasons == ()


def test_multiple_signals_reported_together():
    flag = detect_exact_terms('does "residuals" survive under the Indian Contract Act, Section 27?')
    assert flag.flagged
    assert set(flag.reasons) == {"quoted_phrase", "section_number", "defined_term"}
