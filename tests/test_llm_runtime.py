"""
Tests for llm_runtime module.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.llm_runtime import _extract_json_object  # noqa: E402


def test_extract_json_object_valid_json():
    """Test extracting valid JSON without wrapping text."""
    valid_json = '{"key": "value", "number": 42}'
    result = _extract_json_object(valid_json)
    assert result == {"key": "value", "number": 42}


def test_extract_json_object_with_markdown():
    """Test extracting JSON when wrapped inside markdown blocks and prose."""
    markdown_json = """
Here is your JSON:
```json
{
    "status": "success",
    "data": [1, 2, 3]
}
```
Hope this helps!
"""
    result = _extract_json_object(markdown_json)
    assert result == {"status": "success", "data": [1, 2, 3]}


def test_extract_json_object_invalid_json():
    """Test exception raised when no JSON-like text is found."""
    invalid_json = "This is just a regular string without any JSON structure."
    with pytest.raises(json.JSONDecodeError):
        _extract_json_object(invalid_json)


def test_extract_json_object_malformed_json_fallback():
    """Test exception raised when fallback text within curly braces is invalid JSON."""
    malformed_json = "Here is something { invalid: json, format: true } that fails."
    with pytest.raises(json.JSONDecodeError):
        _extract_json_object(malformed_json)
