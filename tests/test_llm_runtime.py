from __future__ import annotations

import json
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.llm_runtime import _extract_json_object  # noqa: E402


def test_extract_json_object_valid_json():
    # Test valid JSON
    valid_json = '{"key": "value", "number": 42}'
    result = _extract_json_object(valid_json)
    assert result == {"key": "value", "number": 42}

def test_extract_json_object_with_markdown():
    """Test JSON wrapped in markdown, which triggers the JSONDecodeError fallback."""
    # Test JSON wrapped in markdown, which triggers the JSONDecodeError fallback
    markdown_json = '''
Here is your JSON:
```json
{
    "status": "success",
    "data": [1, 2, 3]
}
```
Hope this helps!
'''
    result = _extract_json_object(markdown_json)
    assert result == {"status": "success", "data": [1, 2, 3]}

def test_extract_json_object_invalid_json():
    # Test invalid JSON which cannot be extracted
    invalid_json = "This is just a regular string without any JSON structure."
    with pytest.raises(json.JSONDecodeError):
        _extract_json_object(invalid_json)

def test_extract_json_object_malformed_json_fallback():
    # Test string with curly braces but invalid JSON inside it
    malformed_json = "Here is something { invalid: json, format: true } that fails."
    with pytest.raises(json.JSONDecodeError):
        _extract_json_object(malformed_json)
