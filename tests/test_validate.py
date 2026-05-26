import socket
from validate import print_check, find_free_port

def test_print_check(capsys):
    print_check("Test 1", True, "Detail 1")
    captured = capsys.readouterr()
    assert captured.out.strip() == "PASS: Test 1 - Detail 1"

    print_check("Test 2", False)
    captured = capsys.readouterr()
    assert captured.out.strip() == "FAIL: Test 2"


def test_find_free_port():
    port = find_free_port()
    assert isinstance(port, int)
    assert 1024 <= port <= 65535

    # Verify the port can actually be bound
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))
        assert sock.getsockname()[1] == port

from unittest.mock import patch, MagicMock # noqa: E402
from validate import wait_for_server # noqa: E402
import httpx # noqa: E402

def test_wait_for_server_success():
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.Client.get", return_value=mock_response):
        assert wait_for_server("http://localhost:8000", timeout=0.1)

def test_wait_for_server_timeout():
    mock_request = MagicMock()
    with patch("httpx.Client.get", side_effect=httpx.RequestError("Error", request=mock_request)):
        assert not wait_for_server("http://localhost:8000", timeout=0.1)

def test_wait_for_server_non_200():
    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch("httpx.Client.get", return_value=mock_response):
        assert not wait_for_server("http://localhost:8000", timeout=0.1)

import pytest # noqa: E402
from validate import check_health # noqa: E402

@pytest.mark.asyncio
async def test_check_health_success():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200

    # We must use an AsyncMock-like approach for async methods
    import asyncio
    future = asyncio.Future()
    future.set_result(mock_response)
    mock_client.get.return_value = future

    result = await check_health(mock_client)
    assert result

@pytest.mark.asyncio
async def test_check_health_failure():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 500

    import asyncio
    future = asyncio.Future()
    future.set_result(mock_response)
    mock_client.get.return_value = future

    result = await check_health(mock_client)
    assert not result

@pytest.mark.asyncio
async def test_check_health_exception():
    mock_client = MagicMock()
    mock_client.get.side_effect = Exception("Connection Error")

    result = await check_health(mock_client)
    assert not result

from validate import greedy_action # noqa: E402

def test_greedy_action_submit_fallback():
    observation = {
        "query": "something",
        "available_chunks": [],
        "selected_chunks": [],
        "step_number": 1,
        "total_tokens_used": 0,
        "token_budget": 1000
    }
    result = greedy_action(observation)
    assert result["action_type"] == "submit_report"

def test_greedy_action_set_resolution_plan():
    observation = {
        "query": "something",
        "available_chunks": [],
        "selected_chunks": ["chunk1"],
        "step_number": 3,
        "total_tokens_used": 100,
        "token_budget": 1000
    }
    result = greedy_action(observation)
    assert result["action_type"] == "set_resolution_plan"

def test_greedy_action_summarize_artifact():
    observation = {
        "query": "something",
        "available_chunks": [{"chunk_id": "chunk1", "tokens": 500}],
        "selected_chunks": ["chunk1"],
        "step_number": 1,
        "total_tokens_used": 500,
        "token_budget": 1000,
        "plan_draft": "a draft"
    }
    result = greedy_action(observation)
    assert result["action_type"] == "summarize_artifact"
    assert result["artifact_id"] == "chunk1"

def test_greedy_action_inspect_artifact():
    observation = {
        "query": "error log database",
        "available_chunks": [
            {"chunk_id": "chunk1", "tokens": 50, "keywords": ["unrelated", "stuff"]},
            {"chunk_id": "chunk2", "tokens": 50, "keywords": ["database", "error"]}
        ],
        "selected_chunks": [],
        "step_number": 1,
        "total_tokens_used": 0,
        "token_budget": 1000
    }
    result = greedy_action(observation)
    assert result["action_type"] == "inspect_artifact"
    assert result["artifact_id"] == "chunk2"

from validate import planner_action # noqa: E402

def test_planner_action_success():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"action_type": "inspect_artifact", "artifact_id": "chunk1"}
    mock_client.post.return_value = mock_response

    observation = {"query": "test", "available_chunks": []}
    result = planner_action(mock_client, "http://localhost:8000", observation)

    mock_client.post.assert_called_once_with("http://localhost:8000/optimize-step")
    assert result["action_type"] == "inspect_artifact"
    assert result["artifact_id"] == "chunk1"

def test_planner_action_fallback():
    mock_client = MagicMock()
    mock_client.post.side_effect = httpx.RequestError("Connection Error", request=MagicMock())

    observation = {
        "query": "something",
        "available_chunks": [],
        "selected_chunks": [],
        "step_number": 1,
        "total_tokens_used": 0,
        "token_budget": 1000
    }

    # Should fall back to greedy_action which returns submit_report for empty available_chunks
    result = planner_action(mock_client, "http://localhost:8000", observation)
    assert result["action_type"] == "submit_report"

from validate import run_task # noqa: E402

def test_run_task_success(capsys):
    mock_client = MagicMock()

    # Mock response for reset
    reset_response = MagicMock()
    reset_response.status_code = 200
    reset_response.json.return_value = {
        "observation": {
            "query": "something",
            "available_chunks": [],
            "selected_chunks": [],
            "step_number": 1,
            "total_tokens_used": 0,
            "token_budget": 1000
        }
    }

    # Mock response for step
    step_response = MagicMock()
    step_response.status_code = 200
    step_response.json.return_value = {
        "observation": {
            "query": "something",
            "available_chunks": [],
            "selected_chunks": [],
            "step_number": 2,
            "total_tokens_used": 0,
            "token_budget": 1000
        },
        "done": True,
        "reward": 0.85
    }

    mock_client.post.side_effect = [reset_response, step_response]

    # Use planner_action side_effect to avoid hitting real endpoint if it tries to
    with patch("validate.planner_action", return_value={"action_type": "submit_report"}):
        in_range, final_score = run_task(mock_client, "http://localhost:8000", "test_task")

        assert in_range is True
        assert final_score == 0.85

        captured = capsys.readouterr()
        assert "PASS: task test_task score range - score=0.8500" in captured.out

def test_run_task_reset_failure(capsys):
    mock_client = MagicMock()
    reset_response = MagicMock()
    reset_response.status_code = 500
    reset_response.text = "Internal Server Error"

    mock_client.post.return_value = reset_response

    in_range, final_score = run_task(mock_client, "http://localhost:8000", "test_task")

    assert in_range is False
    assert final_score == 0.0

    captured = capsys.readouterr()
    assert "FAIL: reset test_task - Internal Server Error" in captured.out

def test_run_task_step_failure(capsys):
    mock_client = MagicMock()

    reset_response = MagicMock()
    reset_response.status_code = 200
    reset_response.json.return_value = {
        "observation": {
            "query": "something",
            "available_chunks": [],
            "selected_chunks": [],
            "step_number": 1,
            "total_tokens_used": 0,
            "token_budget": 1000
        }
    }

    step_response = MagicMock()
    step_response.status_code = 500
    step_response.text = "Step Error"

    mock_client.post.side_effect = [reset_response, step_response]

    with patch("validate.planner_action", return_value={"action_type": "submit_report"}):
        in_range, final_score = run_task(mock_client, "http://localhost:8000", "test_task")

        assert in_range is False
        assert final_score == 0.0

        captured = capsys.readouterr()
        assert "FAIL: step test_task - Step Error" in captured.out
