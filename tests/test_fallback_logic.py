import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import (  # noqa: E402
    _check_resolution_plan,
    _check_compression,
    _check_early_submit,
    _check_prioritize_candidates,
    _check_inspect_candidates,
    _check_fallback_prioritize,
    _get_final_fallback_action,
    _suggest_action_fallback,
)

class MockChunk:
    def __init__(self, chunk_id, tokens, keywords=None):
        self.chunk_id = chunk_id
        self.tokens = tokens
        self.keywords = keywords or []

class MockScore:
    def __init__(self, final_score, compression_ratio=0.5):
        self.final_score = final_score
        self.compression_ratio = compression_ratio

def test_check_resolution_plan():
    env = MagicMock()
    env.task.required_plan_keywords = ["keyword1", "keyword2", "keyword3"]
    env.task.max_steps = 10
    observation = MagicMock()
    observation.plan_draft = None

    # Not enough reviewed
    assert _check_resolution_plan(env, set(["a"]), observation) is None

    # Already drafted
    observation.plan_draft = "draft"
    assert _check_resolution_plan(env, set(["a", "b"]), observation) is None

    # Should draft
    observation.plan_draft = None
    res = _check_resolution_plan(env, set(["a", "b"]), observation)
    assert res is not None
    assert res["action_type"] == "set_resolution_plan"
    assert "keyword1, keyword2, keyword3" in res["plan"]

def test_check_compression():
    observation = MagicMock()
    observation.total_tokens_used = 700
    observation.token_budget = 1000
    observation.step_number = 2

    chunk = MockChunk("c1", 300)
    score_map = {"c1": MockScore(0.8, 0.4)}

    # Budget exceeded, heavy chunk present
    res = _check_compression(observation, [chunk], score_map)
    assert res is not None
    assert res["action_type"] == "compress_chunk"
    assert res["chunk_id"] == "c1"
    assert res["compression_ratio"] == 0.4

    # No heavy chunk
    chunk_light = MockChunk("c2", 50)
    assert _check_compression(observation, [chunk_light], score_map) is None

def test_check_early_submit():
    env = MagicMock()
    env.task.max_steps = 10
    observation = MagicMock()
    observation.step_number = 3
    selected = set(["c1", "c2"])

    chunk1 = MockChunk("c1", 100, ["key1", "key2"])
    chunk2 = MockChunk("c2", 100, ["key3"])

    # Should early submit due to 2+ selected
    res = _check_early_submit(env, observation, selected, [chunk1, chunk2], ["c1"])
    assert res is not None
    assert res["action_type"] == "submit_answer"
    assert "[c1] key1, key2" in res["answer"]
    assert "[c1]." in res["answer"]

def test_check_prioritize_candidates():
    chunk1 = MockChunk("c1", 100)
    chunk2 = MockChunk("c2", 500)

    # Should prioritize c1 as it fits
    res = _check_prioritize_candidates([chunk1, chunk2], set(["c1", "c2"]), set(), ["c1", "c2"], 200)
    assert res is not None
    assert res["action_type"] == "prioritize_artifact"
    assert res["artifact_id"] == "c1"

def test_check_inspect_candidates():
    chunk1 = MockChunk("c1", 100)
    chunk2 = MockChunk("c2", 50)
    score_map = {"c1": MockScore(0.9), "c2": MockScore(0.8)}

    # Sort logic prioritizes smallest score/tokens ratio. c2 gets prioritized since -0.016 < -0.009.
    res = _check_inspect_candidates([chunk1, chunk2], set(), score_map)
    assert res is not None
    assert res["action_type"] == "inspect_artifact"
    assert res["artifact_id"] == "c2"

def test_check_fallback_prioritize():
    chunk1 = MockChunk("c1", 100)
    chunk2 = MockChunk("c2", 200)
    score_map = {"c1": MockScore(0.5), "c2": MockScore(0.8)}

    # Sort logic prioritizes smallest score/tokens ratio. c1 gets prioritized since -0.005 < -0.004.
    res = _check_fallback_prioritize([chunk1, chunk2], set(["c1", "c2"]), set(), score_map, 300)
    assert res is not None
    assert res["action_type"] == "prioritize_artifact"
    assert res["artifact_id"] == "c1"

def test_get_final_fallback_action():
    # Has selected chunks
    chunk = MockChunk("c1", 100)
    res = _get_final_fallback_action([chunk], set(), [chunk])
    assert res["action_type"] == "submit_report"

    # No selected, available chunks left
    res2 = _get_final_fallback_action([chunk], set(), [])
    assert res2["action_type"] == "submit_answer"
    assert "Increase the budget" in res2["answer"]

    # Nothing available
    res3 = _get_final_fallback_action([], set(), [])
    assert res3["action_type"] == "submit_answer"
    assert "No usable evidence" in res3["answer"]

def test_suggest_action_fallback():
    env = MagicMock()
    observation = MagicMock()
    observation.prioritized_artifacts = ["c1"]
    observation.reviewed_artifacts = ["c1", "c2"]
    observation.token_budget = 1000
    observation.total_tokens_used = 500
    observation.step_number = 1
    observation.plan_draft = None

    chunk1 = MockChunk("c1", 100)
    chunk2 = MockChunk("c2", 200)
    observation.available_artifacts = [chunk1, chunk2]

    env._build_observation.return_value = observation
    tuning = MagicMock()
    tuning.tuned_scores = {"c1": MockScore(0.9), "c2": MockScore(0.8)}
    tuning.suggested_citations = ["c1"]
    env.context_tuner.tune.return_value = tuning
    env.task.max_steps = 10
    env._last_tuning = tuning
    env.task.required_plan_keywords = ["keyword1", "keyword2", "keyword3"]
    env.task.max_steps = 10

    res = _suggest_action_fallback(env)
    assert res is not None
    # With 2 reviewed and no plan draft, it should hit resolution plan first
    assert res["action_type"] == "set_resolution_plan"

def test_check_early_submit_none():
    env = MagicMock()
    env.task.max_steps = 10
    observation = MagicMock()
    # Step number is too low and only 1 selected
    observation.step_number = 1
    selected = set(["c1"])
    chunk1 = MockChunk("c1", 100, ["key1", "key2"])

    res = _check_early_submit(env, observation, selected, [chunk1], ["c1"])
    assert res is None

def test_check_prioritize_candidates_none():
    chunk1 = MockChunk("c1", 100)
    # Not enough budget
    res = _check_prioritize_candidates([chunk1], set(["c1"]), set(), ["c1"], 50)
    assert res is None

def test_check_inspect_candidates_none():
    # Everything is already reviewed
    chunk1 = MockChunk("c1", 100)
    score_map = {"c1": MockScore(0.9)}
    res = _check_inspect_candidates([chunk1], set(["c1"]), score_map)
    assert res is None

def test_check_fallback_prioritize_none():
    chunk1 = MockChunk("c1", 100)
    score_map = {"c1": MockScore(0.5)}
    # Token limit exceeded
    res = _check_fallback_prioritize([chunk1], set(["c1"]), set(), score_map, 50)
    assert res is None

# import pytest

@pytest.mark.anyio
async def test_suggest_action_llm_fallback():
    from app import _suggest_action
    from unittest.mock import patch, AsyncMock

    env = MagicMock()
    env._build_observation.return_value = MagicMock(
        prioritized_artifacts=[],
        reviewed_artifacts=[],
        token_budget=1000,
        total_tokens_used=0,
        step_number=1,
        plan_draft=None,
        available_artifacts=[]
    )
    env.task.max_steps = 10
    tuning = MagicMock()
    tuning.tuned_scores = {}
    tuning.suggested_citations = []
    env.context_tuner.tune.return_value = tuning
    env.task.max_steps = 10
    env._last_tuning = tuning
    env.state = AsyncMock(return_value={})

    # Test that exception in suggest_action_with_llm falls back
    with patch('app.llm_configured', return_value=True), \
         patch('app.suggest_action_with_llm', side_effect=Exception("LLM failed")):
        res = await _suggest_action(env)
        # Should fallback to _get_final_fallback_action which returns submit_answer for empty state
        assert res is not None
        assert res["action_type"] == "submit_answer"
        assert "No usable evidence" in res["answer"]

def test_suggest_action_fallback_full_branch():
    env = MagicMock()
    observation = MagicMock()
    # We want to fall through all conditions until the last return
    # bypass _check_resolution_plan: < 2 reviewed
    observation.reviewed_artifacts = ["c1"]
    # bypass _check_compression: total_tokens_used low
    observation.total_tokens_used = 0
    observation.token_budget = 1000
    observation.step_number = 1
    # bypass _check_early_submit: < 2 selected
    observation.prioritized_artifacts = ["c1"]

    chunk1 = MockChunk("c1", 100)
    # bypass _check_prioritize_candidates, _check_inspect_candidates, _check_fallback_prioritize
    # by having NO other available chunks
    observation.available_artifacts = [chunk1]

    env._build_observation.return_value = observation
    tuning = MagicMock()
    tuning.tuned_scores = {"c1": MockScore(0.9)}
    tuning.suggested_citations = ["c1"]
    env.context_tuner.tune.return_value = tuning
    env.task.max_steps = 10
    env._last_tuning = tuning

    res = _suggest_action_fallback(env)
    assert res is not None
    # Hit the final fallback: submit_report with currently selected evidence
    assert res["action_type"] == "submit_report"
    assert "Optimized answer" in res["answer"]
