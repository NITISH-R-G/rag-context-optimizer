import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import _clamp_score, _format_bool, _format_reward, _format_error  # noqa: E402


def test_clamp_score():
    assert _clamp_score(-1.0) == 0.0
    assert _clamp_score(0.0) == 0.0
    assert _clamp_score(0.5) == 0.5
    assert _clamp_score(1.0) == 1.0
    assert _clamp_score(2.0) == 1.0


def test_format_bool():
    assert _format_bool(True) == "true"
    assert _format_bool(False) == "false"


def test_format_reward():
    assert _format_reward(None) == "0.00"
    assert _format_reward(0.5) == "0.50"
    assert _format_reward(1.0) == "1.00"


def test_format_error():
    assert _format_error(None) == "null"
    assert _format_error("error\nmessage") == "error message"
