"""
Prompt optimization utilities for the HTTP app.

This module keeps prompt-rewriting and evidence-packaging logic out of `app.py`
so the FastAPI layer stays thinner and easier to review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from env.environment import RagContextOptimizerEnv
from env.llm_runtime import estimate_tokens, llm_configured
from env.llm_services import rewrite_prompt as rewrite_prompt_with_llm


CompressionMode = Literal["balanced", "aggressive", "grounded"]

_PROMPT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could", "do", "does", "did",
    "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "me",
    "my", "of", "on", "or", "our", "should", "so", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "to", "too", "use", "using", "was", "we", "were", "what",
    "when", "where", "which", "while", "with", "without", "would", "you", "your",
}

_INSTRUCTION_PRIORITY_TERMS = {
    "must", "should", "only", "not", "never", "always", "include", "exclude", "cite", "answer",
    "return", "draft", "write", "summarize", "compare", "explain", "verify", "preserve", "focus",
    "keep", "avoid", "report", "escalate", "rollback", "refund", "incident", "customer", "security",
}


@dataclass(frozen=True, slots=True)
class PromptOptimizationResult:
    optimized_prompt: str
    stats: dict[str, int]
    grounding: dict[str, Any]
    context_tuning: dict[str, Any]
    corpus_family: str
    selected_keywords: list[str]
    optimization_mode: CompressionMode


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _content_terms(text: str) -> set[str]:
    return {term for term in _tokenize(text) if len(term) > 2 and term not in _PROMPT_STOPWORDS}


def _clean_output_text(text: str) -> str:
    cleaned = text.replace("```", " ").replace("---", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"[#*_`]+", "", cleaned)
    cleaned = re.sub(r'\b(title|emoji|colorfrom|colorto|sdk|app_file|pinned)\s*:\s*', "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" -:")


def _compact_text(text: str, max_words: int = 28) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,;:") + " ..."


def _approx_tokens(text: str) -> int:
    return max(1, len(text.strip()) // 4) if text.strip() else 0


def _truncate_to_word_boundary(text: str, max_chars: int, add_ellipsis: bool = True) -> str:
    raw = text.strip()
    if not raw or len(raw) <= max_chars:
        return raw

    candidate = raw[:max_chars].rstrip(" ,;:\n\t")
    if max_chars < len(raw) and max_chars > 0 and not raw[max_chars - 1].isspace():
        last_space = candidate.rfind(" ")
        if last_space >= max(4, max_chars // 3):
            candidate = candidate[:last_space].rstrip(" ,;:\n\t")

    if not candidate:
        candidate = raw[:max_chars].rstrip(" ,;:\n\t")

    if add_ellipsis and candidate and not candidate.endswith("..."):
        candidate = candidate + " ..."
    return candidate


def _trim_sentence(sentence: str, max_terms: int) -> str:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_\/]*|[,:;()]", sentence)
    if not words:
        return ""
    kept: list[str] = []

    for index, token in enumerate(words):
        normalized = re.sub(r"[^A-Za-z0-9]+", "", token).lower()
        if token in {",", ":", ";", "(", ")"}:
            if kept and kept[-1] not in {",", ":", ";", "("}:
                kept.append(token)
            continue
        is_priority = normalized in _INSTRUCTION_PRIORITY_TERMS
        is_meaningful = (
            normalized.isdigit()
            or any(ch in token for ch in ("_", "-", "/"))
            or len(normalized) >= 4
            or is_priority
            or index < 3
        )
        if not is_meaningful:
            continue
        if normalized in _PROMPT_STOPWORDS and not is_priority and index >= 3:
            continue
        kept.append(token)
        if len([word for word in kept if word not in {",", ":", ";", "(", ")"}]) >= max_terms:
            break

    text = " ".join(kept)
    text = re.sub(r"\s+([,:;)])", r"\1", text)
    text = re.sub(r"(\()\s+", r"\1", text)
    return text.strip(" ,;:")


def _rewrite_prompt_text(prompt: str, target_tokens: int) -> str:
    raw = " ".join(prompt.strip().split())
    if not raw:
        return ""

    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+|\n+", raw) if segment.strip()]
    if not sentences:
        sentences = [raw]

    rewritten: list[str] = []
    used_terms = 0
    max_terms = max(8, target_tokens)
    for index, sentence in enumerate(sentences):
        remaining = max_terms - used_terms
        if remaining <= 0:
            break
        compact = _trim_sentence(sentence, max(4, remaining if index == 0 else min(remaining, 12)))
        if not compact:
            continue
        rewritten.append(compact)
        used_terms += len(compact.split())
        if used_terms >= max_terms:
            break

    if not rewritten:
        fallback = _trim_sentence(raw, max_terms)
        return fallback or raw[: max(16, target_tokens * 4)].strip()

    output = ". ".join(rewritten).strip()
    if len(rewritten) == 1 and not output.endswith("."):
        output += "."
    return output


def _lightweight_short_prompt_rewrite(prompt: str) -> str:
    raw = " ".join(prompt.strip().split())
    if not raw:
        return ""

    cleaned = raw
    cleaned = re.sub(r"\b[Pp]lease\s+", "", cleaned)
    cleaned = re.sub(r"\bhelp me to\b", "help me", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bhelp me\b", "Help me", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bi want to\b", "I want to", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bcan you help me\b", "Help me", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _sentence_rank(query: str, text: str) -> list[str]:
    query_terms = _tokenize(query)
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
    if not sentences:
        return []

    ranked: list[tuple[float, str]] = []
    for index, sentence in enumerate(sentences):
        sentence_terms = _tokenize(sentence)
        overlap = len(query_terms & sentence_terms)
        score = (overlap * 2.0) + (0.25 if index == 0 else 0.0)
        ranked.append((score, sentence))

    ranked.sort(key=lambda item: (-item[0], len(item[1])))
    return [sentence for _score, sentence in ranked]


def _summarize_chunk_for_output(chunk: Any, effective_text: str) -> str:
    if getattr(chunk, "domain", "").startswith("Project"):
        keywords = ", ".join(chunk.keywords[:5])
        domain = chunk.domain.replace("Project ", "").lower()
        return _compact_text(f"This benchmark's {domain} covers {keywords}.", 24)
    ranked_sentences = _sentence_rank(" ".join(chunk.keywords), _clean_output_text(effective_text))
    if ranked_sentences:
        return _compact_text(_clean_output_text(ranked_sentences[0]))
    return _compact_text(_clean_output_text(effective_text))


def _target_ratio(input_tokens: int, mode: CompressionMode) -> float:
    if mode == "aggressive":
        if input_tokens <= 24:
            return 0.78
        if input_tokens <= 60:
            return 0.66
        if input_tokens <= 120:
            return 0.58
        return 0.52
    if mode == "grounded":
        if input_tokens <= 24:
            return 0.98
        if input_tokens <= 60:
            return 0.90
        if input_tokens <= 120:
            return 0.84
        return 0.78
    if input_tokens <= 24:
        return 0.85
    if input_tokens <= 60:
        return 0.75
    if input_tokens <= 120:
        return 0.68
    return 0.62


def _fit_citations_into_prompt(
    base_prompt: str,
    citation_ids: list[str],
    input_tokens: int,
    target_tokens: int,
    source_prompt: str,
    mode: CompressionMode,
) -> tuple[str, bool, str | None]:
    if not citation_ids:
        return base_prompt, False, "No high-confidence evidence anchors were selected."

    prioritized = citation_ids[: (3 if mode == "grounded" else 2)]
    suffix = " Evidence: " + " ".join(f"[{chunk_id}]" for chunk_id in prioritized)
    with_all = (base_prompt.rstrip(".") + "." + suffix).strip()
    if mode == "grounded" and _approx_tokens(with_all) <= max(input_tokens, target_tokens + 4):
        return with_all, True, None
    if _approx_tokens(with_all) < input_tokens:
        return with_all, True, None

    one_suffix = " Evidence: " + f"[{citation_ids[0]}]"
    with_one = (base_prompt.rstrip(".") + "." + one_suffix).strip()
    if mode == "grounded" and _approx_tokens(with_one) <= max(input_tokens, target_tokens + 2):
        return with_one, True, None
    if _approx_tokens(with_one) < input_tokens:
        return with_one, True, None

    tighter_target = max(8, target_tokens - (2 if mode == "grounded" else 3))
    tighter_prompt = _rewrite_prompt_text(source_prompt, tighter_target)
    tighter_with_one = (tighter_prompt.rstrip(".") + "." + one_suffix).strip()
    if mode == "grounded" and _approx_tokens(tighter_with_one) <= max(input_tokens, target_tokens + 2):
        return tighter_with_one, True, None
    if _approx_tokens(tighter_with_one) < input_tokens:
        return tighter_with_one, True, None

    if mode == "grounded":
        forced = (tighter_prompt.rstrip(".") + "." + one_suffix).strip()
        return forced, True, "Grounded mode preserved at least one inline citation, even at the cost of a slightly longer prompt."

    return base_prompt, False, "Citations were omitted to keep the optimized prompt shorter than the original. Use grounded mode or the evidence notes below if explicit anchors are required."


async def optimize_prompt(
    prompt: str,
    corpus_family: str | None = None,
    mode: CompressionMode = "balanced",
) -> PromptOptimizationResult:
    clean_prompt = prompt.strip()
    env = RagContextOptimizerEnv(
        task_name="single_domain_qa",
        query_override=clean_prompt,
        token_budget_override=800,
        max_steps_override=6,
        corpus_family_override=corpus_family,
    )
    await env.reset()

    tuning = env._last_tuning or env.context_tuner.tune(clean_prompt, env._available_chunks)

    ranked_candidates = []
    for chunk in env._available_chunks:
        tuned = tuning.tuned_scores.get(chunk.chunk_id)
        score = tuned.final_score if tuned is not None else env.retriever.hybrid_score(clean_prompt, chunk)
        if score < 0.16:
            continue
        ranked_candidates.append((chunk, score, tuned))
    ranked_candidates.sort(
        key=lambda item: (
            -(item[2].citation_prior if item[2] is not None else 0.0) if mode == "grounded" else 0.0,
            -(item[1] / max(item[0].tokens, 1)),
            -item[1],
            item[0].chunk_id,
        )
    )

    selected_ids: list[str] = []
    token_cap = 420 if mode == "grounded" else 360
    running_tokens = 0
    for chunk, score, _tuned in ranked_candidates:
        if len(selected_ids) >= (5 if mode == "grounded" else 4):
            break
        if score < (0.18 if mode == "grounded" else 0.22) and selected_ids:
            break
        projected = running_tokens + chunk.tokens
        if projected > token_cap and selected_ids:
            continue
        selected_ids.append(chunk.chunk_id)
        env._selected_chunks.append(chunk.chunk_id)
        running_tokens += chunk.tokens

    if not selected_ids and ranked_candidates:
        best_chunk = ranked_candidates[0][0]
        selected_ids.append(best_chunk.chunk_id)
        env._selected_chunks.append(best_chunk.chunk_id)

    for chunk_id in list(selected_ids):
        chunk = env._chunk_map().get(chunk_id)
        if chunk is None:
            continue
        tuned = tuning.tuned_scores.get(chunk_id)
        score = tuned.final_score if tuned is not None else env.retriever.hybrid_score(clean_prompt, chunk)
        ratio = tuned.compression_ratio if tuned is not None else 0.5
        if mode == "grounded":
            ratio = max(ratio, 0.68 if score >= 0.55 else 0.58)
        elif score >= 0.75:
            ratio = max(ratio, 0.6)
        env._compression_ratios[chunk_id] = ratio

    input_tokens = await estimate_tokens(clean_prompt)
    target_tokens = max(12, int(input_tokens * _target_ratio(input_tokens, mode)))
    target_tokens = min(target_tokens, 120 if mode == "grounded" else 80)
    preserve_short_prompt = mode != "aggressive" and input_tokens <= 12 and len(clean_prompt.split()) <= 8

    distilled_points: list[tuple[str, str]] = []
    if not preserve_short_prompt:
        for chunk_id in env._selected_chunks:
            chunk = env._chunk_map().get(chunk_id)
            if chunk is None:
                continue
            best = _summarize_chunk_for_output(chunk, env._effective_chunk_text(chunk_id))
            if best and all(existing != best for _cid, existing in distilled_points):
                distilled_points.append((chunk_id, best))
            if len(distilled_points) >= (3 if mode == "grounded" else (2 if input_tokens < 80 else 3)):
                break

    citation_ids = tuning.suggested_citations or list(env._selected_chunks)
    if llm_configured():
        llm_result = await rewrite_prompt_with_llm(
            prompt=clean_prompt,
            mode=mode,
            target_tokens=target_tokens,
            evidence_notes=[
                {"chunk_id": chunk_id, "note": note}
                for chunk_id, note in distilled_points
            ],
            citation_ids=citation_ids,
        )
        optimized_prompt = llm_result["optimized_prompt"] or clean_prompt
        citation_ready = llm_result["citation_ready"]
        citation_guidance = llm_result["citation_guidance"]
        optimized_prompt_tokens = llm_result["estimated_tokens"]
    else:
        rewritten = _rewrite_prompt_text(clean_prompt, target_tokens=target_tokens)
        short_prompt_rewrite = _lightweight_short_prompt_rewrite(clean_prompt) if preserve_short_prompt else ""
        lines: list[str] = [
            short_prompt_rewrite if preserve_short_prompt and short_prompt_rewrite else (
                clean_prompt if preserve_short_prompt else (rewritten if rewritten else clean_prompt)
            )
        ]
        if distilled_points and (mode == "grounded" or input_tokens >= 80):
            lines.append("")
            lines.append("Context:")
            lines.extend([f"- [{chunk_id}] {point}" for chunk_id, point in distilled_points])

        optimized_prompt = "\n".join(lines).strip()

        if preserve_short_prompt and not distilled_points:
            optimized_prompt = short_prompt_rewrite if short_prompt_rewrite and short_prompt_rewrite != clean_prompt else clean_prompt
        elif mode != "grounded" and input_tokens > 0 and _approx_tokens(optimized_prompt) >= input_tokens:
            max_chars = max(12, (input_tokens - 1) * 4)
            optimized_prompt = _truncate_to_word_boundary(optimized_prompt, max_chars)
            while input_tokens > 1 and _approx_tokens(optimized_prompt) >= input_tokens and len(optimized_prompt) > 12:
                optimized_prompt = _truncate_to_word_boundary(optimized_prompt, max(8, len(optimized_prompt) - 6))
            if input_tokens > 1 and _approx_tokens(optimized_prompt) >= input_tokens:
                optimized_prompt = _rewrite_prompt_text(clean_prompt, target_tokens=max(5, input_tokens - 1))
                if optimized_prompt and not optimized_prompt.endswith("...") and _approx_tokens(optimized_prompt) >= input_tokens:
                    optimized_prompt = _truncate_to_word_boundary(optimized_prompt, max(8, (input_tokens - 1) * 4))

        optimized_prompt, citation_ready, citation_guidance = _fit_citations_into_prompt(
            optimized_prompt,
            citation_ids,
            input_tokens,
            target_tokens,
            clean_prompt,
            mode,
        )
        optimized_prompt_tokens = await estimate_tokens(optimized_prompt)

    original_prompt_tokens = input_tokens
    source_tokens = sum(env._chunk_map()[chunk_id].tokens for chunk_id in env._selected_chunks if chunk_id in env._chunk_map())
    compressed_tokens = sum(env._effective_chunk_tokens(chunk_id) for chunk_id in env._selected_chunks)
    evidence_terms = _content_terms(" ".join(env._effective_chunk_text(chunk_id) for chunk_id in env._selected_chunks))
    prompt_terms = _content_terms(optimized_prompt)
    inline_citations = set(re.findall(r"\[([a-z0-9_]+)\]", optimized_prompt.lower()))
    grounded_overlap = (len(prompt_terms & evidence_terms) / len(prompt_terms)) if prompt_terms else 0.0

    return PromptOptimizationResult(
        optimized_prompt=optimized_prompt,
        stats={
            "selected_chunks": len(env._selected_chunks),
            "source_tokens": source_tokens,
            "compressed_context_tokens": compressed_tokens,
            "original_prompt_tokens": original_prompt_tokens,
            "optimized_prompt_tokens": optimized_prompt_tokens,
            "compression_gain": max(0, source_tokens - compressed_tokens),
        },
        grounding={
            "citations": tuning.suggested_citations or list(env._selected_chunks),
            "citation_ready": citation_ready and bool(inline_citations),
            "citation_guidance": citation_guidance,
            "grounded_overlap": round(grounded_overlap, 3),
            "evidence_notes": [
                {"chunk_id": chunk_id, "note": note}
                for chunk_id, note in distilled_points
            ],
        },
        context_tuning={
            "mode": tuning.mode,
            "top_demo_cases": tuning.top_demo_cases,
            "suggested_citations": tuning.suggested_citations,
            "token_dropout": tuning.token_dropout,
            "leave_one_out": tuning.leave_one_out,
        },
        corpus_family=env._corpus_family,
        selected_keywords=[
            keyword
            for chunk_id in env._selected_chunks
            for keyword in (env._chunk_map().get(chunk_id).keywords if env._chunk_map().get(chunk_id) else [])
        ][:10],
        optimization_mode=mode,
    )
