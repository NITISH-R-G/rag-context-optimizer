import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # noqa: E402

import json
import pytest
from pathlib import Path
from typing import Generator
from env.corpus import (
    load_corpus,
    get_chunk_by_id,
    get_chunks_by_domain,
    Chunk,
    _CORPUS_CACHE,
    _CORPUS_ID_MAP,
    _CORPUS_CACHE_PATH,
)

@pytest.fixture(autouse=True)
def reset_corpus_state():
    """Reset the global corpus state before each test."""
    global _CORPUS_CACHE, _CORPUS_ID_MAP, _CORPUS_CACHE_PATH
    import env.corpus as corpus_module
    corpus_module._CORPUS_CACHE = []
    corpus_module._CORPUS_ID_MAP = {}
    corpus_module._CORPUS_CACHE_PATH = None
    yield

@pytest.fixture
def mock_corpus_file(tmp_path: Path) -> Path:
    """Create a temporary mock corpus file."""
    corpus_path = tmp_path / "mock_corpus.jsonl"
    chunks = [
        {
            "chunk_id": "chunk_1",
            "domain": "test_domain",
            "text": "This is test chunk 1.",
            "tokens": 10,
            "keywords": ["test", "chunk", "1"],
            "relevance_tags": ["tag1"]
        },
        {
            "chunk_id": "chunk_2",
            "domain": "test_domain",
            "text": "This is test chunk 2.",
            "tokens": 12,
            "keywords": ["test", "chunk", "2"],
            "relevance_tags": ["tag2"]
        },
        {
            "chunk_id": "chunk_3",
            "domain": "other_domain",
            "text": "This is test chunk 3.",
            "tokens": 15,
            "keywords": ["test", "chunk", "3"],
            "relevance_tags": ["tag3"]
        }
    ]
    with open(corpus_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
    return corpus_path

def test_get_chunk_by_id(mock_corpus_file: Path):
    """Test retrieving a single chunk by its ID."""
    # First, load the corpus so _CORPUS_ID_MAP is populated
    load_corpus(mock_corpus_file)

    # Test retrieving existing chunk
    chunk = get_chunk_by_id("chunk_1")
    assert chunk is not None
    assert isinstance(chunk, Chunk)
    assert chunk.chunk_id == "chunk_1"
    assert chunk.domain == "test_domain"

    chunk2 = get_chunk_by_id("chunk_3")
    assert chunk2 is not None
    assert chunk2.chunk_id == "chunk_3"
    assert chunk2.domain == "other_domain"

    # Test retrieving non-existent chunk
    missing_chunk = get_chunk_by_id("missing_chunk")
    assert missing_chunk is None

def test_get_chunks_by_domain(mock_corpus_file: Path):
    """Test retrieving chunks by domain."""
    load_corpus(mock_corpus_file)

    # Test existing domain
    chunks = get_chunks_by_domain("test_domain")
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "chunk_1"
    assert chunks[1].chunk_id == "chunk_2"

    # Test other domain
    other_chunks = get_chunks_by_domain("other_domain")
    assert len(other_chunks) == 1
    assert other_chunks[0].chunk_id == "chunk_3"

    # Test non-existent domain
    empty_chunks = get_chunks_by_domain("non_existent_domain")
    assert len(empty_chunks) == 0
