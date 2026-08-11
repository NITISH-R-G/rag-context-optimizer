import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.graders import _tokenize  # noqa: E402


def test_tokenize():
    # Test simple lowercase string
    assert _tokenize("hello world") == {"hello", "world"}

    # Test uppercase letters are lowercased
    assert _tokenize("HELLO World") == {"hello", "world"}

    # Test numbers are kept
    assert _tokenize("hello 123 world") == {"hello", "123", "world"}

    # Test punctuation is ignored
    assert _tokenize("hello, world! This is a test.") == {"hello", "world", "this", "is", "a", "test"}

    # Test uniqueness (set behavior)
    assert _tokenize("hello hello world world") == {"hello", "world"}

    # Test empty string
    assert _tokenize("") == set()

    # Test string with only punctuation
    assert _tokenize("!@#$%^&*()") == set()
