"""
Deterministic graders for the incident operations environment.
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
    return chunk_id.strip()


@dataclass(frozen=True, slots=True)
class GraderResult:
    score: float
    breakdown: dict[str, float | str]
    passed: bool


class TaskGrader:
    def _required_chunks(self, retriever: HybridRetriever, task: Task) -> list[Chunk]:
        normalized_required = {_normalize_chunk_id(chunk_id) for chunk_id in task.required_artifact_ids}
        return [chunk for chunk in retriever.corpus if chunk.chunk_id in normalized_required]

    def _keyword_coverage(self, text: str, required_keywords: list[str]) -> float:
        content = text.lower()
        if not required_keywords:
            return 1.0
        hits = sum(1 for keyword in required_keywords if keyword.lower() in content)
        return hits / len(required_keywords)

    def _artifact_coverage(self, prioritized_artifact_ids: set[str], task: Task) -> float:
        required = {_normalize_chunk_id(chunk_id) for chunk_id in task.required_artifact_ids}
        if not required:
            return 1.0
        return len(prioritized_artifact_ids & required) / len(required)

    def _domain_coverage(self, prioritized_artifact_ids: set[str], retriever: HybridRetriever, task: Task) -> float:
        required = {_normalize_chunk_id(chunk_id) for chunk_id in task.required_artifact_ids}
        required_domains = {
            chunk.domain
            for chunk in retriever.corpus
            if chunk.chunk_id in required
        }
        if not required_domains:
            return 1.0
        prioritized_domains = {
            chunk.domain
            for chunk in retriever.corpus
            if chunk.chunk_id in prioritized_artifact_ids
        }
        return len(prioritized_domains & required_domains) / len(required_domains)

    def _citation_accuracy(self, answer: str, prioritized_artifact_ids: set[str], task: Task) -> float:
        citations = {_normalize_chunk_id(chunk_id) for chunk_id in _extract_citations(answer)}
        expected = {_normalize_chunk_id(chunk_id) for chunk_id in task.expected_citation_ids}
        if not citations:
            return 0.0
        valid = citations & prioritized_artifact_ids
        precision = len(valid) / len(citations)
        recall = len(valid & expected) / len(expected) if expected else 1.0
        return (precision + recall) / 2.0

    def _unsupported_claim_rate(self, answer: str, evidence_chunks: list[Chunk]) -> float:
        answer_terms = _content_terms(re.sub(r"\[[a-z0-9_]+\]", " ", answer.lower()))
        evidence_terms = _content_terms(
            " ".join(chunk.text for chunk in evidence_chunks) + " " +
            " ".join(" ".join(chunk.keywords) for chunk in evidence_chunks)
        )
        if not answer_terms:
            return 0.0
        unsupported = answer_terms - evidence_terms
        return len(unsupported) / len(answer_terms)

    def grade(
        self,
        prioritized_artifact_ids: list[str],
        reviewed_artifact_ids: list[str],
        answer: str,
        plan_draft: str,
        workflow_stage: str,
        token_budget: int,
        total_tokens_used: int,
        retriever: HybridRetriever,
        task: Task,
    ) -> GraderResult:
        prioritized = {_normalize_chunk_id(chunk_id) for chunk_id in prioritized_artifact_ids}
        reviewed = {_normalize_chunk_id(chunk_id) for chunk_id in reviewed_artifact_ids}
        required_chunks = self._required_chunks(retriever, task)
        evidence_chunks = [chunk for chunk in retriever.corpus if chunk.chunk_id in prioritized] or required_chunks

        artifact_coverage = self._artifact_coverage(prioritized, task)
        review_coverage = self._artifact_coverage(reviewed, task)
        domain_coverage = self._domain_coverage(prioritized, retriever, task)
        plan_quality = self._keyword_coverage(plan_draft, task.required_plan_keywords)
        report_quality = self._keyword_coverage(answer, task.required_report_keywords)
        citation_accuracy = self._citation_accuracy(answer, prioritized, task)
        token_efficiency = 1.0 - (total_tokens_used / token_budget) if total_tokens_used <= token_budget else 0.0
        token_efficiency = max(0.0, min(1.0, token_efficiency))
        workflow_readiness = 1.0 if workflow_stage in {"resolution", "submitted"} and plan_draft.strip() else 0.25 if plan_draft.strip() else 0.0
        unsupported_claim_rate = self._unsupported_claim_rate(answer, evidence_chunks)
        hallucination_penalty = min(1.0, unsupported_claim_rate)

        base_score = (
            0.24 * artifact_coverage
            + 0.12 * review_coverage
            + 0.12 * domain_coverage
            + 0.16 * plan_quality
            + 0.18 * report_quality
            + 0.10 * citation_accuracy
            + 0.08 * token_efficiency
            + 0.10 * workflow_readiness
        )
        score = max(0.0, min(1.0, base_score - (0.18 * hallucination_penalty)))

        breakdown: dict[str, float | str] = {
            "artifact_coverage": artifact_coverage,
            "review_coverage": review_coverage,
            "domain_coverage": domain_coverage,
            "plan_quality": plan_quality,
            "report_quality": report_quality,
            "citation_accuracy": citation_accuracy,
            "token_efficiency": token_efficiency,
            "workflow_readiness": workflow_readiness,
            "unsupported_claim_rate": unsupported_claim_rate,
            "hallucination_penalty": hallucination_penalty,
        }
        passed = score >= 0.72
        return GraderResult(score=score, breakdown=breakdown, passed=passed)
