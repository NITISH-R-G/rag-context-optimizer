import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.graders import _tokenize  # noqa: E402


def test_tokenize():
    # Happy path: simple words
    assert _tokenize("hello world") == {"hello", "world"}

    # Uppercase and mixed case: should lowercase everything
    assert _tokenize("Hello WORLD") == {"hello", "world"}

    # Punctuation: should ignore punctuation
    assert _tokenize("hello, world! this is a test.") == {"hello", "world", "this", "is", "a", "test"}

    # Empty string: should return empty set
    assert _tokenize("") == set()

    # String with numbers
    assert _tokenize("item1 and item2") == {"item1", "and", "item2"}
    assert _tokenize("123 456") == {"123", "456"}

    # Deduplication: should return a set, ignoring duplicates
    assert _tokenize("apple apple banana") == {"apple", "banana"}

    # Whitespace and newlines
    assert _tokenize("  hello \n world \t ") == {"hello", "world"}

    # Special characters
    assert _tokenize("@hello #world") == {"hello", "world"}
