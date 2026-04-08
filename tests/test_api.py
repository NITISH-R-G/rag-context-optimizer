from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app


client = TestClient(app)


def test_reset_accepts_empty_body():
    response = client.post("/reset")
    assert response.status_code == 200
    body = response.json()
    assert "episode_id" in body
    assert body["done"] is False
    assert "observation" in body


def test_episode_state_is_isolated():
    first_reset = client.post("/reset", json={"task_name": "single_domain_qa"})
    second_reset = client.post("/reset", json={"task_name": "cross_domain_synthesis"})
    assert first_reset.status_code == 200
    assert second_reset.status_code == 200

    first_episode = first_reset.json()["episode_id"]
    second_episode = second_reset.json()["episode_id"]
    assert first_episode != second_episode

    first_chunk = first_reset.json()["observation"]["available_chunks"][0]["chunk_id"]
    step = client.post(f"/step?episode_id={first_episode}", json={"action_type": "select_chunk", "chunk_id": first_chunk})
    assert step.status_code == 200
    assert step.json()["episode_id"] == first_episode

    first_state = client.get(f"/state?episode_id={first_episode}")
    second_state = client.get(f"/state?episode_id={second_episode}")
    assert first_state.status_code == 200
    assert second_state.status_code == 200
    assert first_chunk in first_state.json()["selected_chunks"]
    assert second_state.json()["selected_chunks"] == []
