from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_inference_uses_proxy_api_key():
    app_port = _free_port()
    proxy_port = _free_port()
    requests_seen: list[dict[str, str | None]] = []

    class ProxyHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            requests_seen.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "body": body,
                }
            )
            payload = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "proxy-test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action_type": "submit_answer",
                                    "answer": "Proxy verified [support_003]",
                                }
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args):
            return

    proxy_server = HTTPServer(("127.0.0.1", proxy_port), ProxyHandler)
    proxy_thread = threading.Thread(target=proxy_server.serve_forever, daemon=True)
    proxy_thread.start()

    app_process = subprocess.Popen(
        [str(PYTHON), "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(app_port)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                if httpx.get(f"http://127.0.0.1:{app_port}/health", timeout=2).status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)

        env = os.environ.copy()
        env["RAG_ENV_URL"] = f"http://127.0.0.1:{app_port}"
        env["RAG_ENV_TASK"] = "single_domain_qa"
        env["API_BASE_URL"] = f"http://127.0.0.1:{proxy_port}/v1"
        env["API_KEY"] = "proxy-check-token"
        env["HF_TOKEN"] = "legacy-should-not-win"
        result = subprocess.run(
            [str(PYTHON), "inference.py"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0
        assert requests_seen
        assert requests_seen[0]["path"] == "/v1/chat/completions"
        assert requests_seen[0]["authorization"] == "Bearer proxy-check-token"
        assert any(line.startswith("[END]") and "score=" in line for line in result.stdout.splitlines())
    finally:
        proxy_server.shutdown()
        proxy_server.server_close()
        app_process.terminate()
        try:
            app_process.wait(timeout=5)
        except Exception:
            app_process.kill()
