import pytest
import json

from env.llm_runtime import _extract_json_object

def test_extract_json_object_valid_json():
    payload = '{"key": "value"}'
    result = _extract_json_object(payload)
    assert result == {"key": "value"}

def test_extract_json_object_markdown_wrapped_json():
    payload = '''```json
{
    "key": "value",
    "nested": {"inner": 123}
}
```'''
    result = _extract_json_object(payload)
    assert result == {"key": "value", "nested": {"inner": 123}}

def test_extract_json_object_text_with_json():
    payload = 'Here is the JSON you requested: {"answer": 42} I hope this helps!'
    result = _extract_json_object(payload)
    assert result == {"answer": 42}

def test_extract_json_object_invalid_json():
    payload = 'This is just some text without any valid JSON.'
    with pytest.raises(json.JSONDecodeError):
        _extract_json_object(payload)

def test_extract_json_object_malformed_json_fallback_fails():
    payload = 'Here is some malformed json: {"key": "value"'
    with pytest.raises(json.JSONDecodeError):
        _extract_json_object(payload)
