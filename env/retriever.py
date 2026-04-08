"""
Deterministic hybrid retrieval utilities for rag-context-optimizer.

The retriever combines a corpus-aware BM25 score with a keyword-overlap score, using
only Python standard library components so runs are reproducible across environments.

Doctest examples:
>>> from env.corpus import Chunk
>>> corpus = [
...     Chunk(
...         chunk_id="c1",
...         domain="Climate Policy",
...         text="Carbon tax rebates helped households accept climate policy.",
...         tokens=40,
...         keywords=["carbon tax", "rebates", "households", "policy", "climate"],
...         relevance_tags=["carbon_pricing"],
...     ),
...     Chunk(
...         chunk_id="c2",
...         domain="Software Engineering Best Practices",
...         text="Code review and rollback notes improve deployment safety.",
...         tokens=35,
...         keywords=["code review", "rollback", "deployment", "safety", "review"],
...         relevance_tags=["code_review"],
...     ),
... ]
>>> retriever = HybridRetriever(corpus)
>>> round(retriever.keyword_overlap_score("carbon tax rebates", corpus[0]), 3) > 0
True
>>> ranked = retriever.rank_chunks("carbon tax rebates", top_k=1)
>>> ranked[0][0].chunk_id
'c1'
>>> "c1" in retriever.get_ground_truth_relevant("carbon tax rebates", threshold=0.1)
True
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable

from env.corpus import Chunk


class HybridRetriever:
    """Hybrid lexical retriever with deterministic BM25 and keyword overlap scoring."""

    _STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for", "from",
        "how", "if", "in", "into", "is", "it", "its", "of", "on", "or", "should",
        "that", "the", "their", "them", "there", "these", "this", "to", "using",
        "what", "when", "where", "which", "while", "with", "without", "your",
    }

    def __init__(self, corpus: list[Chunk]):
        self.corpus = list(corpus)
        self._k1 = 1.5
        self._b = 0.75
        self._doc_tokens: dict[str, list[str]] = {}
        self._doc_term_freqs: dict[str, Counter[str]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._doc_freqs: dict[str, int] = defaultdict(int)

        total_length = 0
        for chunk in self.corpus:
            tokens = self._tokenize_for_bm25(chunk.text)
            self._doc_tokens[chunk.chunk_id] = tokens
            term_freqs = Counter(tokens)
            self._doc_term_freqs[chunk.chunk_id] = term_freqs
            doc_len = len(tokens)
            self._doc_lengths[chunk.chunk_id] = doc_len
            total_length += doc_len
            for term in term_freqs:
                self._doc_freqs[term] += 1

        self._avg_doc_length = total_length / len(self.corpus) if self.corpus else 0.0
    @staticmethod
    def _tokenize_for_bm25(text: str) -> list[str]:
        return [
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in HybridRetriever._STOPWORDS and len(token) > 1
        ]

    @staticmethod
    def _tokenize_query_terms(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in HybridRetriever._STOPWORDS and len(token) > 1
        }

    @staticmethod
    def _normalize_score(value: float, maximum: float) -> float:
        if maximum <= 0.0:
            return 0.0
        return max(0.0, min(1.0, value / maximum))

    def _idf(self, term: str) -> float:
        """Compute BM25 IDF using corpus statistics."""
        doc_count = len(self.corpus)
        if doc_count == 0:
            return 0.0
        frequency = self._doc_freqs.get(term, 0)
        return math.log(1.0 + ((doc_count - frequency + 0.5) / (frequency + 0.5)))

    def _raw_bm25(self, query_terms: Iterable[str], chunk: Chunk) -> float:
        term_freqs = self._doc_term_freqs.get(chunk.chunk_id, Counter())
        doc_len = self._doc_lengths.get(chunk.chunk_id, 0)
        avg_doc_len = self._avg_doc_length or 1.0
        score = 0.0

        for term in Counter(query_terms):
            freq = term_freqs.get(term, 0)
            if freq == 0:
                continue
            idf = self._idf(term)
            numerator = freq * (self._k1 + 1.0)
            denominator = freq + self._k1 * (1.0 - self._b + self._b * (doc_len / avg_doc_len))
            score += idf * (numerator / denominator)

        return score

    def _max_raw_bm25_for_query(self, query_terms: list[str]) -> float:
        if not self.corpus or not query_terms:
            return 0.0
        return max(self._raw_bm25(query_terms, chunk) for chunk in self.corpus)

    def bm25_score(self, query: str, chunk: Chunk) -> float:
        """
        Implement BM25 scoring using only stdlib and return a normalized score in [0.0, 1.0].

        >>> from env.corpus import Chunk
        >>> corpus = [Chunk(chunk_id="a", domain="Climate Policy", text="carbon tax rebates", tokens=10, keywords=["carbon"], relevance_tags=["x"])]
        >>> retriever = HybridRetriever(corpus)
        >>> 0.0 <= retriever.bm25_score("carbon tax", corpus[0]) <= 1.0
        True
        """
        query_terms = self._tokenize_for_bm25(query)
        raw_score = self._raw_bm25(query_terms, chunk)
        max_score = self._max_raw_bm25_for_query(query_terms)
        return self._normalize_score(raw_score, max_score)

    def keyword_overlap_score(self, query: str, chunk: Chunk) -> float:
        """
        Compute Jaccard similarity between query tokens and chunk keyword tokens.

        >>> from env.corpus import Chunk
        >>> corpus = [Chunk(chunk_id="a", domain="Climate Policy", text="x", tokens=10, keywords=["carbon tax", "rebates"], relevance_tags=["x"])]
        >>> retriever = HybridRetriever(corpus)
        >>> round(retriever.keyword_overlap_score("carbon rebates", corpus[0]), 3)
        0.667
        """
        query_terms = self._tokenize_query_terms(query)
        keyword_terms = self._tokenize_query_terms(" ".join(chunk.keywords))
        if not query_terms or not keyword_terms:
            return 0.0
        union = query_terms | keyword_terms
        if not union:
            return 0.0
        return len(query_terms & keyword_terms) / len(union)

    def hybrid_score(
        self,
        query: str,
        chunk: Chunk,
        bm25_weight: float = 0.6,
        keyword_weight: float = 0.4,
    ) -> float:
        """Return a weighted combination of BM25 and keyword overlap scores."""
        if bm25_weight < 0 or keyword_weight < 0:
            raise ValueError("Weights must be non-negative.")
        weight_sum = bm25_weight + keyword_weight
        if weight_sum == 0:
            raise ValueError("At least one weight must be positive.")
        bm25_component = self.bm25_score(query, chunk)
        keyword_component = self.keyword_overlap_score(query, chunk)
        score = ((bm25_component * bm25_weight) + (keyword_component * keyword_weight)) / weight_sum
        return max(0.0, min(1.0, score))

    def rank_chunks(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        """
        Return chunks sorted by hybrid_score descending.

        >>> from env.corpus import Chunk
        >>> corpus = [
        ...     Chunk(chunk_id="a", domain="Climate Policy", text="carbon tax rebates for households", tokens=10, keywords=["carbon tax"], relevance_tags=["x"]),
        ...     Chunk(chunk_id="b", domain="Software Engineering Best Practices", text="code review safety", tokens=10, keywords=["code review"], relevance_tags=["y"]),
        ... ]
        >>> retriever = HybridRetriever(corpus)
        >>> [chunk.chunk_id for chunk, _ in retriever.rank_chunks("carbon tax", top_k=2)]
        ['a', 'b']
        """
        scored = [(chunk, self.hybrid_score(query, chunk)) for chunk in self.corpus]
        scored.sort(key=lambda item: (-item[1], item[0].chunk_id))
        return scored[: max(0, top_k)]

    def get_ground_truth_relevant(self, query: str, threshold: float = 0.3) -> set[str]:
        """
        Return chunk_ids with hybrid score above or equal to the threshold.

        >>> from env.corpus import Chunk
        >>> corpus = [Chunk(chunk_id="a", domain="Climate Policy", text="carbon tax rebates", tokens=10, keywords=["carbon tax", "rebates"], relevance_tags=["x"])]
        >>> retriever = HybridRetriever(corpus)
        >>> retriever.get_ground_truth_relevant("carbon tax", threshold=0.1) == {'a'}
        True
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0.")
        return {
            chunk.chunk_id
            for chunk, score in self.rank_chunks(query, top_k=len(self.corpus))
            if score >= threshold
        }
