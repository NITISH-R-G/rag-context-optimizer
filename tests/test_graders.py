from env.graders import _normalize_chunk_id

def test_normalize_chunk_id():
    assert _normalize_chunk_id("  chunk_1  ") == "chunk_1"
    assert _normalize_chunk_id("chunk_2\n") == "chunk_2"
    assert _normalize_chunk_id("\tchunk_3") == "chunk_3"
    assert _normalize_chunk_id("chunk_4") == "chunk_4"
    assert _normalize_chunk_id("") == ""
    assert _normalize_chunk_id("   ") == ""
