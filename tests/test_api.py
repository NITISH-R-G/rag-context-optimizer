from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from app import app, _is_bad_action_event, EpisodeStore, _resolve_env  # noqa: E402


client = TestClient(app)


def test_reset_accepts_empty_body():
    response = client.post("/reset")
    assert response.status_code == 200
    body = response.json()
    assert "episode_id" in body
    assert body["done"] is False
    assert "observation" in body


def test_episode_state_is_isolated():
    first_reset = client.post("/reset", json={"task_name": "refund_triage_easy"})
    second_reset = client.post("/reset", json={"task_name": "cross_function_brief_medium"})
    assert first_reset.status_code == 200
    assert second_reset.status_code == 200

    first_episode = first_reset.json()["episode_id"]
    second_episode = second_reset.json()["episode_id"]
    assert first_episode != second_episode

    first_chunk = first_reset.json()["observation"]["available_chunks"][0]["chunk_id"]
    step = client.post(
        f"/step?episode_id={first_episode}",
        json={"action_type": "inspect_artifact", "artifact_id": first_chunk},
    )
    assert step.status_code == 200
    assert step.json()["episode_id"] == first_episode

    first_state = client.get(f"/state?episode_id={first_episode}")
    second_state = client.get(f"/state?episode_id={second_episode}")
    assert first_state.status_code == 200
    assert second_state.status_code == 200
    assert first_chunk in first_state.json()["reviewed_artifacts"]
    assert second_state.json()["reviewed_artifacts"] == []

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "tasks" in body

def test_tasks_endpoint():
    response = client.get("/tasks")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    if len(body) > 0:
        assert "name" in body[0]
        assert "difficulty" in body[0]

def test_corpus_families_endpoint():
    response = client.get("/corpus-families")
    assert response.status_code == 200
    body = response.json()
    assert "families" in body
    assert isinstance(body["families"], list)

def test_optimize_prompt_empty():
    response = client.post("/optimize-prompt", json={"prompt": "   ", "compression_mode": "balanced"})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_reset_invalid_task():
    response = client.post("/reset", json={"task_name": "invalid_task"})
    assert response.status_code == 400
    assert "Unknown task_name" in response.json()["detail"]

def test_step_invalid_episode_id():
    response = client.post("/step?episode_id=invalid_id", json={"action_type": "submit_report", "answer": "test"})
    assert response.status_code == 404
    assert "Episode not found" in response.json()["detail"]

def test_resolve_env_invalid_episode_id():
    with pytest.raises(HTTPException) as exc_info:
        _resolve_env("invalid_episode_id")
    assert exc_info.value.status_code == 404
    assert "Episode not found. Call /reset first." in exc_info.value.detail

def test_is_bad_action_event():
    assert _is_bad_action_event(None) is False
    assert _is_bad_action_event("") is False
    assert _is_bad_action_event("some_other_event") is False
    assert _is_bad_action_event("artifact_not_found") is False
    assert _is_bad_action_event("bad_action_error") is True
    assert _is_bad_action_event("not_implemented") is True
    assert _is_bad_action_event("some_bad_action_error_suffix") is True
    assert _is_bad_action_event("prefix_not_implemented_suffix") is True

@pytest.mark.anyio
async def test_episode_store_eviction():
    store = EpisodeStore(max_episodes=2)

    # Need real AsyncMock for env.close()
    class MockEnv:
        async def close(self):
            pass
    env1 = MockEnv()
    env2 = MockEnv()
    env3 = MockEnv()

    id1 = await store.create(env1)
    id2 = await store.create(env2)
    assert store.get(id1)[0] == id1
    assert store.get(id2)[0] == id2

    id3 = await store.create(env3)
    # id1 should be evicted
    with pytest.raises(KeyError):
        store.get(id1)
    assert store.get(id2)[0] == id2
    assert store.get(id3)[0] == id3

@pytest.mark.anyio
async def test_episode_store_close_all():
    store = EpisodeStore(max_episodes=2)
    class MockEnv:
        def __init__(self):
            self.closed = False
        async def close(self):
            self.closed = True

    env1 = MockEnv()
    env2 = MockEnv()
    id1 = await store.create(env1)
    id2 = await store.create(env2)

    await store.close_all()
    assert env1.closed is True
    assert env2.closed is True
    with pytest.raises(KeyError):
        store.get(id1)
    with pytest.raises(KeyError):
        store.get(id2)

@pytest.mark.anyio
async def test_log_requests_middleware():
    from app import log_requests
    import os

    class MockRequest:
        def __init__(self):
            self.method = "GET"
            class URL:
                path = "/test"
            self.url = URL()

    async def mock_call_next(request):
        class MockResponse:
            status_code = 200
        return MockResponse()

    # Test with logging disabled (default)
    import builtins
    original_print = builtins.print
    printed_messages = []
    def mock_print(*args, **kwargs):
        printed_messages.append(" ".join(map(str, args)))

    builtins.print = mock_print
    try:
        await log_requests(MockRequest(), mock_call_next)
        assert len(printed_messages) == 0

        # Test with logging enabled
        os.environ["DEBUG_LOG_REQUESTS"] = "1"
        await log_requests(MockRequest(), mock_call_next)
        assert len(printed_messages) == 2
        assert "[request] GET /test" in printed_messages[0]
        assert "[response] GET /test -> 200" in printed_messages[1]
    finally:
        builtins.print = original_print
        os.environ.pop("DEBUG_LOG_REQUESTS", None)

def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert response.headers["Cache-Control"] == "no-store, max-age=0"

def test_serialize_observation():
    from app import _serialize_observation
    from dataclasses import dataclass
    from pydantic import BaseModel

    # Test dictionary fallback
    obs_dict = {"key": "value"}
    assert _serialize_observation(obs_dict) == {"key": "value"}

    # Test tuple fallback (handled by dict())
    obs_tuple = (("key", "value"),)
    assert _serialize_observation(obs_tuple) == {"key": "value"}

    # Test pydantic model
    class MockModel(BaseModel):
        key: str
    obs_pydantic = MockModel(key="value")
    assert _serialize_observation(obs_pydantic) == {"key": "value"}

    # Test dataclass
    @dataclass
    class MockDataclass:
        key: str
    obs_dataclass = MockDataclass(key="value")
    assert _serialize_observation(obs_dataclass) == {"key": "value"}

def test_step_endpoint_bad_action_event():
    # Setup episode
    reset_response = client.post("/reset", json={"task_name": "refund_triage_easy"})
    episode_id = reset_response.json()["episode_id"]

    # Send a good request first to verify the environment handles it
    chunk_id = reset_response.json()["observation"]["available_chunks"][0]["chunk_id"]
    step_response = client.post(f"/step?episode_id={episode_id}", json={"action_type": "inspect_artifact", "artifact_id": chunk_id})
    assert step_response.status_code == 200

    # For a bad action event, we mock `env.step` to return an info dict with event="bad_action_error"
    from unittest.mock import patch
    with patch('env.environment.RagContextOptimizerEnv.step') as mock_step:
        class MockResult:
            def __init__(self):
                self.info = {"event": "bad_action_error"}
                self.observation = {}
                self.reward = 0
                self.done = False

        async def mock_step_coro(*args, **kwargs):
            return MockResult()

        mock_step.side_effect = mock_step_coro

        response = client.post(f"/step?episode_id={episode_id}", json={"action_type": "inspect_artifact", "artifact_id": "test"})
        assert response.status_code == 400
        assert "bad_action_error" in response.json()["detail"]

def test_optimize_step_endpoint():
    # Setup episode
    reset_response = client.post("/reset", json={"task_name": "refund_triage_easy"})
    episode_id = reset_response.json()["episode_id"]

    response = client.post(f"/optimize-step?episode_id={episode_id}")
    assert response.status_code == 200
    assert "action_type" in response.json()

def test_optimize_prompt_endpoint():
    from unittest.mock import patch

    with patch('app._optimize_prompt_backend') as mock_backend:
        async def mock_optimize(*args, **kwargs):
            return {
                "optimized_prompt": "optimized",
                "stats": {},
                "grounding": {},
                "context_tuning": {},
                "corpus_family": "test",
                "selected_keywords": [],
                "optimization_mode": "balanced"
            }
        mock_backend.side_effect = mock_optimize

        response = client.post("/optimize-prompt", json={"prompt": "test prompt", "compression_mode": "balanced"})
        assert response.status_code == 200
        data = response.json()
        assert data["optimized_prompt"] == "optimized"
