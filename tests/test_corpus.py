import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import env.corpus as corpus_module  # noqa: E402
from env.corpus import Chunk, get_chunk_by_id, load_corpus  # noqa: E402


@pytest.fixture(autouse=True)
def clean_corpus_globals():
    """Fixture to ensure the global corpus state is clean before and after each test."""
    # Setup: backup and clear
    old_cache = list(corpus_module._CORPUS_CACHE)
    old_map = dict(corpus_module._CORPUS_ID_MAP)
    old_path = corpus_module._CORPUS_CACHE_PATH

    corpus_module._CORPUS_CACHE.clear()
    corpus_module._CORPUS_ID_MAP.clear()
    corpus_module._CORPUS_CACHE_PATH = None

    yield

    # Teardown: restore
    corpus_module._CORPUS_CACHE = old_cache
    corpus_module._CORPUS_ID_MAP = old_map
    corpus_module._CORPUS_CACHE_PATH = old_path


def test_get_chunk_by_id(tmp_path: Path):
    dummy_file = tmp_path / "corpus.jsonl"
    dummy_data = {
        "chunk_id": "test_chunk_123",
        "domain": "test_domain",
        "text": "This is a test document.",
        "tokens": 15,
        "keywords": ["test", "document"],
        "relevance_tags": ["testing"]
    }

    with open(dummy_file, "w") as f:
        f.write(json.dumps(dummy_data) + "\n")

    # Load corpus using the dummy file
    loaded_chunks = load_corpus(dummy_file)
    assert len(loaded_chunks) == 1

    # Test valid ID retrieval
    chunk = get_chunk_by_id("test_chunk_123")
    assert chunk is not None
    assert isinstance(chunk, Chunk)
    assert chunk.chunk_id == "test_chunk_123"
    assert chunk.domain == "test_domain"

    # Test invalid ID retrieval
    invalid_chunk = get_chunk_by_id("nonexistent_id")
    assert invalid_chunk is None
