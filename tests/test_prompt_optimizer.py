"""Test prompt optimizer module."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.prompt_optimizer import ( # noqa: E402
    _tokenize,
    _content_terms,
    _clean_output_text,
    _compact_text,
    _approx_tokens,
    _truncate_to_word_boundary,
    _trim_sentence,
    _rewrite_prompt_text,
    _lightweight_short_prompt_rewrite,
    _sentence_rank,
    _summarize_chunk_for_output,
    _target_ratio,
    _fit_citations_into_prompt,
    _rank_and_select_chunks,
    _extract_distilled_points,
    _rewrite_prompt_fallback,
    optimize_prompt,
)
from env.environment import RagContextOptimizerEnv # noqa: E402


def test_tokenize():
    text = "Hello, world! This is a test."
    tokens = _tokenize(text)
    assert set(tokens) == {"hello", "world", "this", "is", "a", "test"}


def test_content_terms():
    text = "The quick brown fox jumps over the lazy dog in the rain."
    terms = _content_terms(text)
    assert "quick" in terms
    assert "fox" in terms


def test_clean_output_text():
    text = "   This  is   a test   \n string.  "
    cleaned = _clean_output_text(text)
    assert cleaned == "This is a test string."


def test_compact_text():
    text = "word " * 50
    compacted = _compact_text(text, max_words=10)
    assert len(compacted.split()) <= 11 # 10 words + maybe ellipsis
    assert compacted.endswith("...")


def test_approx_tokens():
    text = "This is a short test."
    tokens = _approx_tokens(text)
    assert tokens > 0


def test_truncate_to_word_boundary():
    text = "This is a very long string that needs truncation."
    truncated = _truncate_to_word_boundary(text, 15)
    assert len(truncated) <= 15 + 3 # +3 for ...
    assert truncated.endswith("...")
    truncated_no_ellipsis = _truncate_to_word_boundary(text, 15, add_ellipsis=False)
    assert len(truncated_no_ellipsis) <= 15
    assert not truncated_no_ellipsis.endswith("...")


def test_trim_sentence():
    sentence = "This is a sentence with many terms in it."
    trimmed = _trim_sentence(sentence, max_terms=3)
    assert len(trimmed.split()) <= 4 # words + ...


def test_rewrite_prompt_text():
    prompt = "Can you please tell me what the policy is for a refund?"
    rewritten = _rewrite_prompt_text(prompt, target_tokens=5)
    assert _approx_tokens(rewritten) <= 15 # roughly


def test_lightweight_short_prompt_rewrite():
    prompt = "Please explain the policy."
    rewritten = _lightweight_short_prompt_rewrite(prompt)
    assert isinstance(rewritten, str)


def test_sentence_rank():
    query = "refund policy"
    text = "This is irrelevant. The refund policy allows 30 days. Another sentence."
    ranked = _sentence_rank(query, text)
    assert "refund policy" in ranked[0].lower()


def test_summarize_chunk_for_output():
    class DummyChunk:
        def __init__(self, keywords):
            self.keywords = keywords
            self.chunk_id = "dummy"

    chunk = DummyChunk(["refund", "policy"])
    text = "The refund policy allows for 30 days of returns."
    summary = _summarize_chunk_for_output(chunk, text)
    assert "refund" in summary.lower() or "policy" in summary.lower()


def test_target_ratio():
    assert _target_ratio(100, "aggressive") < _target_ratio(100, "balanced")
    assert _target_ratio(100, "balanced") < _target_ratio(100, "grounded")


def test_fit_citations_into_prompt():
    prompt = "This is a test prompt."
    citations = ["doc1", "doc2"]
    result, ready, notes = _fit_citations_into_prompt(prompt, citations, 10, 5, prompt, "balanced")
    assert isinstance(result, str)
    assert isinstance(ready, bool)


@pytest.mark.asyncio
async def test_rank_and_select_chunks():
    env = RagContextOptimizerEnv("refund_triage_easy")
    await env.reset() # mock env state roughly

    class DummyTuning:
        tuned_scores = {}
        suggested_citations = []

    tuning = DummyTuning()
    _rank_and_select_chunks(env, tuning, "test prompt", "balanced")
    assert isinstance(env._selected_chunks, list)


@pytest.mark.asyncio
async def test_extract_distilled_points():
    env = RagContextOptimizerEnv("refund_triage_easy")
    await env.reset()
    points = _extract_distilled_points(env, "balanced", 50, False)
    assert isinstance(points, list)


def test_rewrite_prompt_fallback():
    result, ready, notes = _rewrite_prompt_fallback(
        "test prompt", 10, 5, "balanced", False, [], ["doc1"]
    )
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_optimize_prompt():
    res1 = await optimize_prompt("What is the refund policy?", mode="balanced")
    assert "optimized_prompt" in res1.model_dump() if hasattr(res1, "model_dump") else hasattr(res1, "optimized_prompt")
    res2 = await optimize_prompt("What is the refund policy?", mode="grounded")
    assert "optimized_prompt" in res2.model_dump() if hasattr(res2, "model_dump") else hasattr(res2, "optimized_prompt")
    res3 = await optimize_prompt("What is the refund policy?", mode="aggressive")
    assert "optimized_prompt" in res3.model_dump() if hasattr(res3, "model_dump") else hasattr(res3, "optimized_prompt")


@pytest.mark.asyncio
async def test_extract_distilled_points_with_preserve_short():
    env = RagContextOptimizerEnv("refund_triage_easy")
    await env.reset()
    points = _extract_distilled_points(env, "balanced", 50, preserve_short_prompt=True)
    assert points == []

def test_fit_citations_into_prompt_aggressive():
    prompt = "This is a prompt."
    citations = ["doc1", "doc2"]
    result, ready, notes = _fit_citations_into_prompt(prompt, citations, 10, 5, prompt, "aggressive")
    assert isinstance(result, str)

def test_fit_citations_into_prompt_grounded():
    prompt = "This is a prompt."
    citations = ["doc1", "doc2"]
    result, ready, notes = _fit_citations_into_prompt(prompt, citations, 10, 5, prompt, "grounded")
    assert "doc1" in result or notes is not None

def test_rewrite_prompt_fallback_long():
    long_prompt = "A " * 50
    result, ready, notes = _rewrite_prompt_fallback(
        long_prompt, 50, 10, "balanced", False, [("doc1", "note")], ["doc1"]
    )
    assert "doc1" in result

def test_rewrite_prompt_fallback_preserve_short():
    short_prompt = "Short prompt."
    result, ready, notes = _rewrite_prompt_fallback(
        short_prompt, 2, 2, "balanced", True, [], ["doc1"]
    )
    assert len(result) > 0
