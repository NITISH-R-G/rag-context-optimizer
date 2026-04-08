"""
Deterministic graders for rag-context-optimizer tasks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from env.corpus import Chunk
from env.retriever import HybridRetriever
from env.tasks import Task


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "by", "for", "from", "how",
    "if", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "them", "there", "these", "this", "to", "was", "were", "what", "when", "where",
    "which", "while", "with", "within", "without", "you", "your",
}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _content_terms(text: str) -> set[str]:
    return {term for term in _tokenize(text) if len(term) > 2 and term not in _STOPWORDS}


def _extract_citations(text: str) -> list[str]:
    return re.findall(r"\[([a-z0-9_]+)\]", text.lower())


def _normalize_chunk_id(chunk_id: str) -> str:
    chunk_id = chunk_id.strip()
    return chunk_id


def _normalize_domain_filter(domain_filter: str | None) -> str | None:
    if domain_filter is None:
        return None
    mapping = {
        "customer_support_operations": "Customer Support Operations",
        "incident_response_playbooks": "Incident Response Playbooks",
        "platform_reliability_release_engineering": "Platform Reliability & Release Engineering",
    }
    return mapping.get(domain_filter, domain_filter)


def _f1_score(selected: set[str], relevant: set[str]) -> float:
    if not selected and not relevant:
        return 1.0
    if not selected or not relevant:
        return 0.0
    overlap = len(selected & relevant)
    if overlap == 0:
        return 0.0
    precision = overlap / len(selected)
    recall = overlap / len(relevant)
    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True, slots=True)
class GraderResult:
    score: float
    breakdown: dict[str, float]
    passed: bool


class TaskGrader:
    def _filter_relevant_by_domain(self, relevant_ids: set[str], retriever: HybridRetriever, task: Task) -> set[str]:
        normalized_domain = _normalize_domain_filter(task.domain_filter)
        if normalized_domain is None:
            return relevant_ids
        allowed_ids = {chunk.chunk_id for chunk in retriever.corpus if chunk.domain == normalized_domain}
        return relevant_ids & allowed_ids

    def _required_chunks(self, retriever: HybridRetriever, task: Task) -> list[Chunk]:
        normalized_required = {_normalize_chunk_id(chunk_id) for chunk_id in task.required_chunk_ids}
        return [chunk for chunk in retriever.corpus if chunk.chunk_id in normalized_required]

    def _answer_quality(self, answer: str, required_chunks: list[Chunk]) -> float:
        answer_terms = _content_terms(answer)
        required_terms = _content_terms(" ".join(chunk.text for chunk in required_chunks))
        required_terms |= _content_terms(" ".join(" ".join(chunk.keywords) for chunk in required_chunks))
        if not answer_terms or not required_terms:
            return 0.0
        union = answer_terms | required_terms
        if not union:
            return 0.0
        return len(answer_terms & required_terms) / len(union)

    def _citation_accuracy(self, answer: str, selected_chunk_ids: set[str], expected_citation_ids: set[str]) -> float:
        citations = {_normalize_chunk_id(chunk_id) for chunk_id in _extract_citations(answer)}
        if not citations:
            return 0.0
        valid_citations = citations & selected_chunk_ids
        precision = len(valid_citations) / len(citations)
        recall = len(valid_citations & expected_citation_ids) / len(expected_citation_ids) if expected_citation_ids else 1.0
        return (precision + recall) / 2.0

    def _unsupported_claim_rate(self, answer: str, evidence_chunks: list[Chunk]) -> float:
        answer_terms = _content_terms(re.sub(r"\[[a-z0-9_]+\]", " ", answer.lower()))
        evidence_terms = _content_terms(" ".join(chunk.text for chunk in evidence_chunks))
        if not answer_terms:
            return 0.0
        unsupported = answer_terms - evidence_terms
        return len(unsupported) / len(answer_terms)

    def grade(
        self,
        selected_chunk_ids: list[str],
        answer: str,
        token_budget: int,
        total_tokens_used: int,
        retriever: HybridRetriever,
        task: Task,
    ) -> GraderResult:
        normalized_selected = {_normalize_chunk_id(chunk_id) for chunk_id in selected_chunk_ids}
        normalized_required = {_normalize_chunk_id(chunk_id) for chunk_id in task.required_chunk_ids}
        relevant = self._filter_relevant_by_domain(normalized_required, retriever, task)

        retrieval_precision = _f1_score(normalized_selected, relevant)
        token_efficiency = 1.0 - (total_tokens_used / token_budget) if total_tokens_used <= token_budget else 0.0
        token_efficiency = max(0.0, min(1.0, token_efficiency))

        required_chunks = self._required_chunks(retriever, task)
        answer_quality = self._answer_quality(answer, required_chunks)

        normalized_expected_citations = {
            _normalize_chunk_id(chunk_id)
            for chunk_id in (task.expected_citation_ids or task.required_chunk_ids)
        }
        required_chunks_hit = (
            len(normalized_selected & normalized_required) / len(normalized_required)
            if normalized_required
            else 1.0
        )

        selected_chunks = [
            chunk for chunk in retriever.corpus if chunk.chunk_id in normalized_selected
        ]
        evidence_chunks = selected_chunks or required_chunks
        citation_accuracy = self._citation_accuracy(answer, normalized_selected, normalized_expected_citations)
        unsupported_claim_rate = self._unsupported_claim_rate(answer, evidence_chunks)
        hallucination_penalty = min(1.0, unsupported_claim_rate)

        base_score = (
            0.25 * retrieval_precision
            + 0.25 * token_efficiency
            + 0.35 * answer_quality
            + 0.15 * required_chunks_hit
        )
        score = base_score + (0.10 * citation_accuracy) - (0.15 * hallucination_penalty)
        score = max(0.0, min(1.0, score))

        breakdown = {
            "retrieval_precision": retrieval_precision,
            "token_efficiency": token_efficiency,
            "answer_quality": answer_quality,
            "required_chunks_hit": required_chunks_hit,
            "citation_accuracy": citation_accuracy,
            "unsupported_claim_rate": unsupported_claim_rate,
            "hallucination_penalty": hallucination_penalty,
        }
        passed = score >= 0.7
        return GraderResult(score=score, breakdown=breakdown, passed=passed)
