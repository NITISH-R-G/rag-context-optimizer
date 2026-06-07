import pytest
import json
from env.llm_runtime import _extract_json_object

def test_extract_json_object_valid_json():
    text = '{"key": "value"}'
    assert _extract_json_object(text) == {"key": "value"}

def test_extract_json_object_with_markdown_wrapper():
    text = '''```json
{
    "key": "value"
}
```'''
    assert _extract_json_object(text) == {"key": "value"}

def test_extract_json_object_with_prefix_and_suffix():
    text = 'Here is the response: {"key": "value"} And some more text.'
    assert _extract_json_object(text) == {"key": "value"}

def test_extract_json_object_no_json():
    text = 'This is just some text with no JSON.'
    with pytest.raises(json.JSONDecodeError):
        _extract_json_object(text)

def test_extract_json_object_invalid_json_in_braces():
    text = 'Here is {invalid json}'
    with pytest.raises(json.JSONDecodeError):
        _extract_json_object(text)
