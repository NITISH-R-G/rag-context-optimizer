import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.llm_runtime import _extract_json_object  # noqa: E402


def test_extract_json_object_happy_path():
    payload = '{"key": "value", "number": 42}'
    result = _extract_json_object(payload)
    assert result == {"key": "value", "number": 42}


def test_extract_json_object_fallback():
    # Test valid JSON wrapped in markdown, which fails direct json.loads
    payload = '''```json
{
  "status": "success",
  "data": [1, 2, 3]
}
```'''
    result = _extract_json_object(payload)
    assert result == {"status": "success", "data": [1, 2, 3]}

    # Test JSON with text before and after
    payload_text = 'Here is the response: {"answer": "yes"} Thank you.'
    result_text = _extract_json_object(payload_text)
    assert result_text == {"answer": "yes"}


def test_extract_json_object_failure():
    # Test invalid JSON that also fails regex or is still invalid after regex
    payload = "This is just a regular string without any JSON."
    with pytest.raises(json.JSONDecodeError):
        _extract_json_object(payload)

    # Regex matches { } but the content is not valid JSON
    payload_invalid_inside = '```json { "unquoted_key": value } ```'
    with pytest.raises(json.JSONDecodeError):
        _extract_json_object(payload_invalid_inside)
