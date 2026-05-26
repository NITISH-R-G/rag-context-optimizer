from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import _extract_json_object  # noqa: E402


def test_extract_valid_json():
    text = '{"key": "value"}'
    result = _extract_json_object(text)
    assert result == {"key": "value"}


def test_extract_json_with_whitespace():
    text = '\n  {"key": "value"}\n  '
    result = _extract_json_object(text)
    assert result == {"key": "value"}


def test_extract_json_embedded():
    text = 'Here is the result:\n```json\n{"status": "ok", "count": 5}\n```\nDone.'
    result = _extract_json_object(text)
    assert result == {"status": "ok", "count": 5}


def test_extract_json_missing():
    text = 'This is just some text with no json.'
    with pytest.raises(json.JSONDecodeError):
        _extract_json_object(text)


def test_extract_json_malformed():
    text = '{"missing_quote: value}'
    with pytest.raises(json.JSONDecodeError):
        _extract_json_object(text)


def test_extract_json_malformed_embedded():
    text = 'some text ```json\n{"missing_quote: value}\n```'
    with pytest.raises(json.JSONDecodeError):
        _extract_json_object(text)
