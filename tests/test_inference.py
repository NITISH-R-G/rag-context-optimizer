from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import _clamp_score  # noqa: E402


def test_clamp_score():
    assert _clamp_score(-1.0) == 0.0
    assert _clamp_score(0.0) == 0.0
    assert _clamp_score(0.5) == 0.5
    assert _clamp_score(1.0) == 1.0
    assert _clamp_score(2.0) == 1.0
