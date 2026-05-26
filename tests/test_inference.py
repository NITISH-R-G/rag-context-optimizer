from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import _tokenize  # noqa: E402


def test_tokenize_empty_string():
    assert _tokenize("") == set()


def test_tokenize_simple_string():
    assert _tokenize("hello world") == {"hello", "world"}


def test_tokenize_mixed_case():
    assert _tokenize("Hello WoRlD") == {"hello", "world"}


def test_tokenize_punctuation():
    assert _tokenize("hello, world! this-is.a-test") == {"hello", "world", "this", "is", "a", "test"}


def test_tokenize_numbers_and_alphanumeric():
    assert _tokenize("hello 123 world456") == {"hello", "123", "world456"}


def test_tokenize_whitespace():
    assert _tokenize(" hello \n world \t ") == {"hello", "world"}


def test_tokenize_all_punctuation():
    assert _tokenize("!@#$%^&*()") == set()
