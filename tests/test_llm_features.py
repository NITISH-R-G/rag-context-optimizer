from __future__ import annotations

import asyncio
import json
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

from fastapi.testclient import TestClient

from app import app
from env.environment import RagContextOptimizerEnv
from env.models import RagAction


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def fake_llm_server():
    port = _free_port()
    requests_seen: list[dict[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body)
            system_prompt = payload["messages"][0]["content"]
            user_prompt = payload["messages"][1]["content"]
            requests_seen.append({"system": system_prompt, "user": user_prompt, "path": self.path})

            if "ACTION_PLANNER" in system_prompt:
                response_payload = {
                    "action_type": "inspect_artifact",
                    "artifact_id": "support_003",
                }
            elif "PROMPT_COMPRESSOR" in system_prompt:
                response_payload = {
                    "optimized_prompt": "Verify outage impact and billing history before refund approval [support_003].",
                    "estimated_tokens": 24,
                    "citation_ready": True,
                    "citation_guidance": "ready",
                }
            elif "ANSWER_GRADER" in system_prompt:
                response_payload = {
                    "answer_quality": 0.92,
                    "groundedness": 0.88,
                    "coverage": 0.91,
                    "citation_support": 0.9,
                    "notes": "Evidence-backed response.",
                }
            elif "TOKEN_ESTIMATOR" in system_prompt:
                token_count = 48 if "confirmed outage" in user_prompt.lower() else 24
                response_payload = {"token_count": token_count}
            else:
                raise AssertionError(f"Unexpected system prompt: {system_prompt}")

            encoded = json.dumps(
                {
                    "id": "chatcmpl-llm-features",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "fake-llm",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": json.dumps(response_payload)},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 64, "completion_tokens": 16, "total_tokens": 80},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args):
            return

    server = HTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port, requests_seen
    finally:
        server.shutdown()
        server.server_close()


def test_optimize_step_uses_llm(monkeypatch):
    with fake_llm_server() as (port, requests_seen):
        monkeypatch.setenv("API_BASE_URL", f"http://127.0.0.1:{port}/v1")
        monkeypatch.setenv("API_KEY", "test-proxy-token")
        monkeypatch.delenv("HF_TOKEN", raising=False)

        with TestClient(app) as client:
            reset = client.post("/reset", json={"task_name": "refund_triage_easy"})
            episode_id = reset.json()["episode_id"]
            response = client.post(f"/optimize-step?episode_id={episode_id}")

        assert response.status_code == 200
        assert response.json()["action_type"] == "inspect_artifact"
        assert response.json()["artifact_id"] == "support_003"
        assert any("/v1/chat/completions" == request["path"] for request in requests_seen)
        assert any("ACTION_PLANNER" in request["system"] for request in requests_seen)


def test_optimize_prompt_uses_llm(monkeypatch):
    with fake_llm_server() as (port, requests_seen):
        monkeypatch.setenv("API_BASE_URL", f"http://127.0.0.1:{port}/v1")
        monkeypatch.setenv("API_KEY", "test-proxy-token")
        monkeypatch.delenv("HF_TOKEN", raising=False)

        with TestClient(app) as client:
            response = client.post(
                "/optimize-prompt",
                json={
                    "prompt": "You are handling a billing escalation after a confirmed outage. Explain the policy steps before refunding.",
                    "compression_mode": "grounded",
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["optimized_prompt"] == "Verify outage impact and billing history before refund approval [support_003]."
        assert body["stats"]["original_prompt_tokens"] == 48
        assert body["stats"]["optimized_prompt_tokens"] == 24
        assert any("PROMPT_COMPRESSOR" in request["system"] for request in requests_seen)
        assert any("TOKEN_ESTIMATOR" in request["system"] for request in requests_seen)

