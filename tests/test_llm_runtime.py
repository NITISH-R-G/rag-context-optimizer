from __future__ import annotations

import json
import pytest

from env.llm_runtime import _extract_json_object

def test_extract_json_object_valid_json():
    text = '{"key": "value"}'
    result = _extract_json_object(text)
    assert result == {"key": "value"}

def test_extract_json_object_with_markdown_wrapper():
    # This exercises the fallback logic (the except JSONDecodeError path)
    text = '''```json
{
  "key": "value",
  "nested": {"inner": 1}
}
```'''
    result = _extract_json_object(text)
    assert result == {"key": "value", "nested": {"inner": 1}}

def test_extract_json_object_with_text_prefix_and_suffix():
    text = 'Here is the json you requested:\n{"key": "value"}\nHope this helps!'
    result = _extract_json_object(text)
    assert result == {"key": "value"}

def test_extract_json_object_no_json():
    text = 'This is just some plain text without any json.'
    with pytest.raises(json.JSONDecodeError):
        _extract_json_object(text)

def test_extract_json_object_invalid_json_in_braces():
    text = 'Some text { "key": invalid } more text'
    with pytest.raises(json.JSONDecodeError):
        _extract_json_object(text)
