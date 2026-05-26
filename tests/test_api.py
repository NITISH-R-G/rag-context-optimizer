from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app  # noqa: E402


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

def test_request_logging_enabled(monkeypatch):
    from app import _request_logging_enabled

    # Test valid 1
    monkeypatch.setenv("LOG_REQUESTS", "1")
    assert _request_logging_enabled() is True

    # Test valid 0
    monkeypatch.setenv("LOG_REQUESTS", "0")
    assert _request_logging_enabled() is False

    # Test invalid string causing ValueError
    monkeypatch.setenv("LOG_REQUESTS", "abc")
    assert _request_logging_enabled() is False

    # Test missing env variable (defaults to 0)
    monkeypatch.delenv("LOG_REQUESTS", raising=False)
    assert _request_logging_enabled() is False
