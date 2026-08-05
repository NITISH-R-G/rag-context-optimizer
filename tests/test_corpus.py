from env.corpus import _CORPUS_FAMILY_FILES, list_corpus_families


def test_list_corpus_families():
    families = list_corpus_families()
    assert isinstance(families, list)
    assert families == sorted(_CORPUS_FAMILY_FILES)
