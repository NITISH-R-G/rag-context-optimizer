from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.graders import _content_terms  # noqa: E402


def test_content_terms_empty():
    assert _content_terms("") == set()


def test_content_terms_stopwords_filtered():
    assert _content_terms("this is a test and that was the end") == {"test", "end"}


def test_content_terms_short_words_filtered():
    assert _content_terms("go to do in up") == set()


def test_content_terms_punctuation_ignored():
    assert _content_terms("hello, world! How's it going?") == {"hello", "world", "going"}


def test_content_terms_case_insensitivity():
    assert _content_terms("Apple APPLE apple") == {"apple"}


def test_content_terms_alphanumeric():
    assert _content_terms("server1 123 server1") == {"server1", "123"}


def test_content_terms_mixed():
    text = "The quick brown fox jumps over the lazy dog! 123 times..."
    expected = {"quick", "brown", "fox", "jumps", "over", "lazy", "dog", "123", "times"}
    assert _content_terms(text) == expected
