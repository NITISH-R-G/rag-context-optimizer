import pytest
from env.corpus import list_corpus_families, _CORPUS_FAMILY_FILES

def test_list_corpus_families():
    families = list_corpus_families()
    assert isinstance(families, list)
    assert families == sorted(_CORPUS_FAMILY_FILES)
