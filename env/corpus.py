"""
Corpus loading utilities for the rag-context-optimizer environment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str = Field(..., description="Unique chunk identifier.")
    domain: str = Field(..., description="High-level corpus domain.")
    text: str = Field(..., description="Document chunk text.")
    tokens: int = Field(..., ge=1, description="Approximate token count for the chunk.")
    keywords: list[str] = Field(..., min_length=1, description="Important keywords for retrieval.")
    relevance_tags: list[str] = Field(
        ...,
        min_length=1,
        description="Tags that describe query relevance and subtopics.",
    )


_CORPUS_CACHE: list[Chunk] = []
_CORPUS_CACHE_PATH: Path | None = None

_CORPUS_FAMILY_FILES = {
    "enterprise_v1": "corpus.jsonl",
    "enterprise_v2": "corpus_pack_enterprise_v2.jsonl",
}


def resolve_corpus_path(path: str | Path | None = None, family: str | None = None) -> Path:
    """Resolve the active corpus path, allowing environment overrides."""
    if path is not None:
        return Path(path)
    family_name = family or os.getenv("RAG_CORPUS_FAMILY")
    if family_name:
        filename = _CORPUS_FAMILY_FILES.get(family_name)
        if filename is None:
            raise ValueError(f"Unknown corpus family: {family_name}")
        return Path(__file__).resolve().parent.parent / "data" / filename
    override = os.getenv("RAG_CORPUS_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "data" / "corpus.jsonl"


def list_corpus_families() -> list[str]:
    return sorted(_CORPUS_FAMILY_FILES)


def load_corpus(path: str | Path) -> list[Chunk]:
    """Load a JSONL corpus file into validated Chunk objects."""
    global _CORPUS_CACHE, _CORPUS_CACHE_PATH
    corpus_path = resolve_corpus_path(path)
    if _CORPUS_CACHE_PATH == corpus_path and _CORPUS_CACHE:
        return list(_CORPUS_CACHE)
    _CORPUS_CACHE = [
        Chunk.model_validate(json.loads(line))
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _CORPUS_CACHE_PATH = corpus_path
    return list(_CORPUS_CACHE)


def get_chunks_by_domain(domain: str) -> list[Chunk]:
    """Return all chunks whose domain matches the provided value."""
    return [chunk for chunk in _CORPUS_CACHE if chunk.domain == domain]


def get_chunk_by_id(chunk_id: str) -> Optional[Chunk]:
    """Return a single chunk by id if it exists in the loaded corpus."""
    for chunk in _CORPUS_CACHE:
        if chunk.chunk_id == chunk_id:
            return chunk
    return None
