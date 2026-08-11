import pytest
from env.environment import RagContextOptimizerEnv


@pytest.mark.asyncio
async def test_load_project_chunks_returns_valid_chunks():
    env = RagContextOptimizerEnv()
    await env.reset()
    # The `_load_project_chunks` method parses README.py, app.py, and other files.
    # We call it explicitly to test its functionality and increase coverage.
    chunks = env._load_project_chunks()

    assert chunks is not None
    assert len(chunks) > 0
    assert all(hasattr(c, "chunk_id") for c in chunks)
    assert all(hasattr(c, "domain") for c in chunks)
    assert all(hasattr(c, "text") for c in chunks)
    assert all(c.tokens > 0 for c in chunks)

    # Specific file we know exists: env/environment.py should be chunked
    environment_chunks = [c for c in chunks if "environment" in c.chunk_id]
    assert len(environment_chunks) > 0

    # Test keyword extraction
    keywords = env._extract_project_keywords(
        "A valid text with some repeating words like words words."
    )
    assert isinstance(keywords, list)

    await env.close()
