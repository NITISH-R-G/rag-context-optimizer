import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.graders import _normalize_chunk_id  # noqa: E402

def test_normalize_chunk_id():
    assert _normalize_chunk_id("  chunk_123  ") == "chunk_123"
    assert _normalize_chunk_id("chunk_456\n") == "chunk_456"
    assert _normalize_chunk_id("\tchunk_789\t") == "chunk_789"
    assert _normalize_chunk_id("chunk_000") == "chunk_000"
    assert _normalize_chunk_id("") == ""
    assert _normalize_chunk_id("   ") == ""
