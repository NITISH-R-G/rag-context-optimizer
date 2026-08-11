import pytest
from pathlib import Path
from env.corpus import (
    list_corpus_families,
    _CORPUS_FAMILY_FILES,
    resolve_corpus_path,
    load_corpus,
    get_chunks_by_domain,
    get_chunk_by_id,
)
import env.corpus

def test_list_corpus_families():
    families = list_corpus_families()
    assert isinstance(families, list)
    assert families == sorted(_CORPUS_FAMILY_FILES)


@pytest.fixture(autouse=True)
def clean_corpus_cache():
    # Save original state
    orig_cache = env.corpus._CORPUS_CACHE.copy()
    orig_id_map = env.corpus._CORPUS_ID_MAP.copy()
    orig_path = env.corpus._CORPUS_CACHE_PATH

    # Clear state
    env.corpus._CORPUS_CACHE = []
    env.corpus._CORPUS_ID_MAP = {}
    env.corpus._CORPUS_CACHE_PATH = None

    yield

    # Restore original state
    env.corpus._CORPUS_CACHE = orig_cache
    env.corpus._CORPUS_ID_MAP = orig_id_map
    env.corpus._CORPUS_CACHE_PATH = orig_path


def test_resolve_corpus_path_explicit():
    assert resolve_corpus_path("foo/bar.jsonl") == Path("foo/bar.jsonl")


def test_resolve_corpus_path_family_explicit():
    path = resolve_corpus_path(family="enterprise_v1")
    assert path.name == "corpus.jsonl"
    assert path.parent.name == "data"


def test_resolve_corpus_path_family_invalid():
    with pytest.raises(ValueError, match="Unknown corpus family"):
        resolve_corpus_path(family="unknown_family")


def test_resolve_corpus_path_env_family(monkeypatch):
    monkeypatch.setenv("RAG_CORPUS_FAMILY", "enterprise_v2")
    path = resolve_corpus_path()
    assert path.name == "corpus_pack_enterprise_v2.jsonl"


def test_resolve_corpus_path_env_override(monkeypatch, tmp_path):
    custom_path = tmp_path / "custom.jsonl"
    monkeypatch.setenv("RAG_CORPUS_PATH", str(custom_path))
    path = resolve_corpus_path()
    assert path == custom_path


def test_resolve_corpus_path_default():
    path = resolve_corpus_path()
    assert path.name == "corpus.jsonl"


def test_load_corpus_and_getters(tmp_path):
    data = (
        '{"chunk_id": "c1", "domain": "d1", "text": "t1", "tokens": 10, "keywords": ["k1"], "relevance_tags": ["r1"]}\n'
        '{"chunk_id": "c2", "domain": "d1", "text": "t2", "tokens": 20, "keywords": ["k2"], "relevance_tags": ["r2"]}\n'
        '{"chunk_id": "c3", "domain": "d2", "text": "t3", "tokens": 30, "keywords": ["k3"], "relevance_tags": ["r3"]}\n'
    )
    p = tmp_path / "test_corpus.jsonl"
    p.write_text(data)

    chunks = load_corpus(p)
    assert len(chunks) == 3
    assert chunks[0].chunk_id == "c1"

    # Test cache hit
    chunks2 = load_corpus(p)
    assert chunks is chunks2 or chunks == chunks2

    # Test get_chunks_by_domain
    d1_chunks = get_chunks_by_domain("d1")
    assert len(d1_chunks) == 2
    assert {c.chunk_id for c in d1_chunks} == {"c1", "c2"}

    d_unknown = get_chunks_by_domain("unknown")
    assert len(d_unknown) == 0

    # Test get_chunk_by_id
    c3 = get_chunk_by_id("c3")
    assert c3 is not None
    assert c3.chunk_id == "c3"

    assert get_chunk_by_id("unknown") is None
