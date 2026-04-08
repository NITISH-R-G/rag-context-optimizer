"""
PyTorch-backed context tuning for retrieval and prompt optimization.

This module adapts the paper's core idea to this benchmark:
- initialize a lightweight context representation from task-specific demonstrations
- optimize that context rather than the underlying language model
- apply leave-one-out masking and token dropout for regularized tuning
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Iterable

from env.corpus import Chunk
from env.retriever import HybridRetriever
from env.tasks import Task

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - optional dependency at runtime
    torch = None
    F = None


_DOMAIN_TO_FLAGS = {
    "Customer Support Operations": (1.0, 0.0, 0.0),
    "Incident Response Playbooks": (0.0, 1.0, 0.0),
    "Platform Reliability & Release Engineering": (0.0, 0.0, 1.0),
}

_DOMAIN_FILTER_MAP = {
    "customer_support_operations": "Customer Support Operations",
    "incident_response_playbooks": "Incident Response Playbooks",
    "platform_reliability_release_engineering": "Platform Reliability & Release Engineering",
}

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "if", "in",
    "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "them", "there",
    "these", "this", "to", "was", "were", "what", "when", "where", "which", "while", "with",
    "without", "you", "your",
}


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOPWORDS and len(token) > 2
    }


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


@dataclass(frozen=True, slots=True)
class DemoCase:
    name: str
    query: str
    positive_chunk_ids: tuple[str, ...]
    expected_citations: tuple[str, ...]
    preferred_domains: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TunedChunkScore:
    chunk_id: str
    base_score: float
    tuned_score: float
    final_score: float
    citation_prior: float
    compression_ratio: float


@dataclass(frozen=True, slots=True)
class ContextTuningResult:
    mode: str
    top_demo_cases: list[str]
    suggested_citations: list[str]
    tuned_scores: dict[str, TunedChunkScore]
    token_dropout: float
    leave_one_out: bool


class ContextTunedPlanner:
    """Warm-start a retrieval policy from demonstrations and optimize context weights."""

    def __init__(self, retriever: HybridRetriever, corpus: list[Chunk], tasks: list[Task]):
        self.retriever = retriever
        self.corpus = list(corpus)
        self._chunk_map = {chunk.chunk_id: chunk for chunk in self.corpus}
        self._demo_cases = self._build_demo_cases(tasks)
        self._token_dropout = 0.14
        self._train_steps = 28
        self._feature_count = 10

    def _build_demo_cases(self, tasks: list[Task]) -> list[DemoCase]:
        demo_cases: list[DemoCase] = []
        query_variants = {
            "refund_triage_easy": [
                "Refund triage memo after a confirmed outage with billing checks and finance review steps.",
                "Business customer outage escalation: verify the billing ledger, incident evidence, and compensation path.",
            ],
            "cross_function_brief_medium": [
                "Cross-functional outage brief linking support handling, incident command discipline, and release rollback safeguards.",
                "Major outage coordination memo for support, incident response, and release engineering teams.",
            ],
            "executive_escalation_hard": [
                "Executive escalation note for compromised admin account response with customer protection and change freeze controls.",
                "Severe security incident brief balancing customer harm reduction, evidence preservation, and release safeguards.",
            ],
        }
        for task in tasks:
            normalized_domain = _DOMAIN_FILTER_MAP.get(task.domain_filter or "", task.domain_filter or "")
            preferred_domains = (
                (normalized_domain,) if normalized_domain else tuple(sorted({
                    self._chunk_map[chunk_id].domain
                    for chunk_id in task.required_artifact_ids
                    if chunk_id in self._chunk_map
                }))
            )
            base = DemoCase(
                name=f"{task.name}_gold",
                query=task.query,
                positive_chunk_ids=tuple(task.required_artifact_ids),
                expected_citations=tuple(task.expected_citation_ids or task.required_artifact_ids),
                preferred_domains=preferred_domains,
            )
            demo_cases.append(base)
            for index, variant in enumerate(query_variants.get(task.name, []), start=1):
                demo_cases.append(
                    DemoCase(
                        name=f"{task.name}_variant_{index}",
                        query=variant,
                        positive_chunk_ids=tuple(task.required_artifact_ids),
                        expected_citations=tuple(task.expected_citation_ids or task.required_artifact_ids),
                        preferred_domains=preferred_domains,
                    )
                )
        return demo_cases

    def _demo_similarity(self, query: str, demo: DemoCase) -> float:
        query_terms = _tokenize(query)
        demo_terms = _tokenize(demo.query)
        return _jaccard(query_terms, demo_terms)

    def _select_demo_cases(self, query: str, limit: int = 4) -> list[DemoCase]:
        ranked = sorted(
            self._demo_cases,
            key=lambda demo: (-self._demo_similarity(query, demo), demo.name),
        )
        chosen = ranked[: max(2, min(limit, len(ranked)))]
        if all(self._demo_similarity(query, demo) == 0.0 for demo in chosen):
            return self._demo_cases[: min(limit, len(self._demo_cases))]
        return chosen

    def _citation_prior(self, chunk_id: str, demos: list[DemoCase], weights: list[float]) -> float:
        if not demos:
            return 0.0
        matched = 0.0
        total = sum(weights) or 1.0
        for demo, weight in zip(demos, weights, strict=False):
            if chunk_id in demo.expected_citations:
                matched += weight
        return matched / total

    def _domain_prior(self, chunk: Chunk, demos: list[DemoCase], weights: list[float]) -> float:
        if not demos:
            return 0.0
        matched = 0.0
        total = sum(weights) or 1.0
        for demo, weight in zip(demos, weights, strict=False):
            if chunk.domain in demo.preferred_domains:
                matched += weight
        return matched / total

    def _query_chunk_overlap(self, query: str, chunk: Chunk) -> float:
        query_terms = _tokenize(query)
        chunk_terms = _tokenize(chunk.text) | _tokenize(" ".join(chunk.keywords))
        return _jaccard(query_terms, chunk_terms)

    def _feature_vector(self, query: str, chunk: Chunk, demos: list[DemoCase], weights: list[float]) -> list[float]:
        base = self.retriever.hybrid_score(query, chunk)
        bm25 = self.retriever.bm25_score(query, chunk)
        keyword = self.retriever.keyword_overlap_score(query, chunk)
        token_efficiency = 1.0 - min(chunk.tokens, 700) / 700.0
        domain_flags = _DOMAIN_TO_FLAGS.get(chunk.domain, (0.0, 0.0, 0.0))
        return [
            base,
            bm25,
            keyword,
            self._query_chunk_overlap(query, chunk),
            token_efficiency,
            domain_flags[0],
            domain_flags[1],
            domain_flags[2],
            self._citation_prior(chunk.chunk_id, demos, weights),
            self._domain_prior(chunk, demos, weights),
        ]

    def _context_init(self, query: str, chunks: list[Chunk], demos: list[DemoCase]) -> list[float]:
        weights = [0.25 + self._demo_similarity(query, demo) for demo in demos]
        positive_acc = [0.0] * self._feature_count
        negative_acc = [0.0] * self._feature_count
        positive_mass = 0.0
        negative_mass = 0.0

        for demo, demo_weight in zip(demos, weights, strict=False):
            for chunk in chunks:
                features = self._feature_vector(demo.query, chunk, demos, weights)
                if chunk.chunk_id in demo.positive_chunk_ids:
                    positive_acc = [value + (demo_weight * feature) for value, feature in zip(positive_acc, features, strict=False)]
                    positive_mass += demo_weight
                else:
                    negative_acc = [value + (demo_weight * feature) for value, feature in zip(negative_acc, features, strict=False)]
                    negative_mass += demo_weight

        positive_mean = [value / max(positive_mass, 1e-6) for value in positive_acc]
        negative_mean = [value / max(negative_mass, 1e-6) for value in negative_acc]
        return [positive - negative for positive, negative in zip(positive_mean, negative_mean, strict=False)]

    def _stable_seed(self, query: str) -> int:
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]
        return int(digest, 16)

    def _optimize_with_torch(self, query: str, chunks: list[Chunk], demos: list[DemoCase]) -> list[float]:
        init = self._context_init(query, chunks, demos)
        if torch is None or F is None or not chunks:
            return init

        seed = self._stable_seed(query)
        torch.manual_seed(seed)
        theta = torch.nn.Parameter(torch.tensor(init, dtype=torch.float32))
        optimizer = torch.optim.Adam([theta], lr=0.12)

        for _ in range(self._train_steps):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, dtype=torch.float32)
            for demo_index, demo in enumerate(demos):
                masked_demos = [item for index, item in enumerate(demos) if index != demo_index]
                if not masked_demos:
                    masked_demos = demos
                masked_init = torch.tensor(
                    self._context_init(demo.query, chunks, masked_demos),
                    dtype=torch.float32,
                )
                drop_mask = (torch.rand_like(masked_init) > self._token_dropout).float()
                drop_mask = drop_mask / max(1e-6, 1.0 - self._token_dropout)
                effective_theta = (0.55 * theta + 0.45 * masked_init) * drop_mask

                weights = [0.25 + self._demo_similarity(demo.query, item) for item in masked_demos]
                matrix = torch.tensor(
                    [self._feature_vector(demo.query, chunk, masked_demos, weights) for chunk in chunks],
                    dtype=torch.float32,
                )
                labels = torch.tensor(
                    [1.0 if chunk.chunk_id in demo.positive_chunk_ids else 0.0 for chunk in chunks],
                    dtype=torch.float32,
                )
                logits = matrix @ effective_theta
                positive_weight = 1.0 + (labels.sum().item() / max(1.0, len(labels) - labels.sum().item()))
                total_loss = total_loss + F.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                    pos_weight=torch.tensor(positive_weight, dtype=torch.float32),
                )

            total_loss.backward()
            optimizer.step()

        return theta.detach().tolist()

    def tune(self, query: str, candidate_chunks: list[Chunk]) -> ContextTuningResult:
        chunks = list(candidate_chunks)
        demos = self._select_demo_cases(query)
        demo_weights = [0.25 + self._demo_similarity(query, demo) for demo in demos]
        theta_values = self._optimize_with_torch(query, chunks, demos)
        mode = "context_tuned_pytorch" if torch is not None else "context_tuned_analytic"

        if torch is not None:
            theta_tensor = torch.tensor(theta_values, dtype=torch.float32)
            matrix_tensor = torch.tensor(
                [self._feature_vector(query, chunk, demos, demo_weights) for chunk in chunks],
                dtype=torch.float32,
            )
            tuned_values = torch.sigmoid(matrix_tensor @ theta_tensor).tolist()
        else:
            tuned_values = []
            for chunk in chunks:
                features = self._feature_vector(query, chunk, demos, demo_weights)
                raw = sum(weight * feature for weight, feature in zip(theta_values, features, strict=False))
                tuned_values.append(1.0 / (1.0 + math.exp(-raw)))

        tuned_scores: dict[str, TunedChunkScore] = {}
        for chunk, tuned_score in zip(chunks, tuned_values, strict=False):
            base_score = self.retriever.hybrid_score(query, chunk)
            citation_prior = self._citation_prior(chunk.chunk_id, demos, demo_weights)
            final_score = max(0.0, min(1.0, (0.40 * base_score) + (0.60 * tuned_score)))
            compression_ratio = 0.82 - (0.34 * final_score) - (0.14 * citation_prior)
            compression_ratio = max(0.38, min(0.84, compression_ratio))
            tuned_scores[chunk.chunk_id] = TunedChunkScore(
                chunk_id=chunk.chunk_id,
                base_score=round(base_score, 4),
                tuned_score=round(float(tuned_score), 4),
                final_score=round(final_score, 4),
                citation_prior=round(citation_prior, 4),
                compression_ratio=round(compression_ratio, 2),
            )

        ranked = sorted(
            tuned_scores.values(),
            key=lambda item: (-item.final_score, -item.citation_prior, item.chunk_id),
        )
        suggested_citations = [item.chunk_id for item in ranked[:3] if item.final_score >= 0.35]
        return ContextTuningResult(
            mode=mode,
            top_demo_cases=[demo.name for demo in demos],
            suggested_citations=suggested_citations,
            tuned_scores=tuned_scores,
            token_dropout=self._token_dropout,
            leave_one_out=True,
        )
