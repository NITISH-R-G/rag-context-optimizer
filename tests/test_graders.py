import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.graders import _extract_citations  # noqa: E402

def test_extract_citations():
    # Simple valid citation
    assert _extract_citations("This is a test [cite1]") == ["cite1"]

    # Multiple citations in one string
    assert _extract_citations("First [cite1] and second [cite2]") == ["cite1", "cite2"]

    # Uppercase citations being lowercased correctly
    assert _extract_citations("Look at [CITE_2]") == ["cite_2"]

    # Strings without citations
    assert _extract_citations("No citations here") == []

    # Strings with empty brackets
    assert _extract_citations("Empty brackets []") == []

    # Citations with invalid characters inside brackets that shouldn't match
    assert _extract_citations("Invalid characters [invalid-cite] [cite!]") == []

    # Adjacent citations
    assert _extract_citations("Adjacent [c1][c2]") == ["c1", "c2"]
