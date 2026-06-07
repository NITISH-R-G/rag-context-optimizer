from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.corpus import get_chunks_by_domain, get_chunk_by_id, Chunk  # noqa: E402


def test_get_chunks_by_domain_returns_filtered_chunks():
    # Setup mock corpus
    mock_chunks = [
        Chunk(
            chunk_id="1",
            domain="domain1",
            text="text1",
            tokens=10,
            keywords=["k1"],
            relevance_tags=["t1"]
        ),
        Chunk(
            chunk_id="2",
            domain="domain2",
            text="text2",
            tokens=20,
            keywords=["k2"],
            relevance_tags=["t2"]
        ),
        Chunk(
            chunk_id="3",
            domain="domain1",
            text="text3",
            tokens=30,
            keywords=["k3"],
            relevance_tags=["t3"]
        ),
    ]

    # Save original cache to restore later
    import env.corpus as corpus_module
    original_cache = list(corpus_module._CORPUS_CACHE)

    try:
        # Override cache
        corpus_module._CORPUS_CACHE = mock_chunks

        # Test filtering
        domain1_chunks = get_chunks_by_domain("domain1")
        assert len(domain1_chunks) == 2
        assert {c.chunk_id for c in domain1_chunks} == {"1", "3"}

        domain2_chunks = get_chunks_by_domain("domain2")
        assert len(domain2_chunks) == 1
        assert domain2_chunks[0].chunk_id == "2"

        domain3_chunks = get_chunks_by_domain("domain3")
        assert len(domain3_chunks) == 0
    finally:
        # Restore cache
        corpus_module._CORPUS_CACHE = original_cache

def test_get_chunk_by_id_returns_correct_chunk():
    mock_chunks = [
        Chunk(
            chunk_id="1",
            domain="domain1",
            text="text1",
            tokens=10,
            keywords=["k1"],
            relevance_tags=["t1"]
        )
    ]

    import env.corpus as corpus_module
    original_map = dict(corpus_module._CORPUS_ID_MAP)

    try:
        corpus_module._CORPUS_ID_MAP = {c.chunk_id: c for c in mock_chunks}

        chunk = get_chunk_by_id("1")
        assert chunk is not None
        assert chunk.chunk_id == "1"
        assert chunk.domain == "domain1"

        chunk = get_chunk_by_id("99")
        assert chunk is None
    finally:
        corpus_module._CORPUS_ID_MAP = original_map
