from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent
TASKS = [
    "single_domain_qa",
    "cross_domain_synthesis",
    "adversarial_compression",
]


def print_check(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"{status}: {name}{suffix}")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def wait_for_server(base_url: str, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    with httpx.Client(timeout=2.0) as client:
        while time.time() < deadline:
            try:
                response = client.get(f"{base_url}/health")
                if response.status_code == 200:
                    return True
            except Exception:
                time.sleep(0.5)
    return False


def greedy_action(observation: dict) -> dict:
    query_terms = set(observation["query"].lower().split())
    selected = set(observation.get("selected_chunks", []))
    available = [chunk for chunk in observation["available_chunks"] if chunk["chunk_id"] not in selected]
    remaining_budget = observation["token_budget"] - observation["total_tokens_used"]

    def overlap(chunk: dict) -> tuple[float, int, str]:
        keyword_terms = set(" ".join(chunk["keywords"]).lower().split())
        union = query_terms | keyword_terms
        score = (len(query_terms & keyword_terms) / len(union)) if union else 0.0
        return (-score, chunk["tokens"], chunk["chunk_id"])

    if selected and (
        observation["step_number"] >= 3
        or observation["total_tokens_used"] >= int(observation["token_budget"] * 0.7)
    ):
        return {"action_type": "submit_answer", "answer": "A concise answer synthesized from the selected chunks."}

    if selected:
        heavy = sorted(
            [chunk for chunk in available + observation["available_chunks"] if chunk["chunk_id"] in selected],
            key=lambda chunk: (-chunk["tokens"], chunk["chunk_id"]),
        )
        if heavy and heavy[0]["tokens"] > max(120, observation["token_budget"] // 3):
            return {
                "action_type": "compress_chunk",
                "chunk_id": heavy[0]["chunk_id"],
                "compression_ratio": 0.5,
            }

    for chunk in sorted(available, key=overlap):
        if chunk["tokens"] <= remaining_budget:
            return {"action_type": "select_chunk", "chunk_id": chunk["chunk_id"]}

    return {"action_type": "submit_answer", "answer": "A concise answer synthesized from the selected chunks."}


def planner_action(client: httpx.Client, base_url: str, fallback_observation: dict) -> dict:
    try:
        response = client.post(f"{base_url}/optimize-step")
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return greedy_action(fallback_observation)


def run_task(client: httpx.Client, base_url: str, task_name: str) -> tuple[bool, float]:
    reset = client.post(f"{base_url}/reset", json={"task_name": task_name})
    if reset.status_code != 200:
        print_check(f"reset {task_name}", False, reset.text)
        return False, 0.0

    observation = reset.json()["observation"]
    done = False
    final_score = 0.0
    while not done:
        action = planner_action(client, base_url, observation)
        step = client.post(f"{base_url}/step", json=action)
        if step.status_code != 200:
            print_check(f"step {task_name}", False, step.text)
            return False, 0.0
        body = step.json()
        observation = body["observation"]
        done = body["done"]
        final_score = float(body["reward"])
    in_range = 0.0 <= final_score <= 1.0
    print_check(f"task {task_name} score range", in_range, f"score={final_score:.4f}")
    return in_range, final_score


def run_inference_script(base_url: str) -> bool:
    env = os.environ.copy()
    env["RAG_ENV_URL"] = base_url
    env["ALLOW_BASELINE_FALLBACK"] = "1"
    env["API_BASE_URL"] = "http://127.0.0.1:9/v1"
    env["API_KEY"] = "offline-validation-token"
    process = subprocess.run(
        [sys.executable, "inference.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    stdout = process.stdout or ""
    has_start = "[START]" in stdout
    has_end = "[END]" in stdout
    end_has_score = " score=" in stdout
    return process.returncode == 0 and has_start and has_end and end_has_score


def main() -> int:
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    command = [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(port)]
    process = subprocess.Popen(command, cwd=PROJECT_ROOT)

    try:
        if not wait_for_server(base_url):
            print_check("server startup", False, "Timed out waiting for /health")
            return 1
        print_check("server startup", True)

        all_passed = True
        with httpx.Client(timeout=10.0) as client:
            health = client.get(f"{base_url}/health")
            health_ok = health.status_code == 200 and health.json().get("status") == "ok"
            print_check("GET /health", health_ok)
            all_passed &= health_ok

            reset = client.post(f"{base_url}/reset", json={"task_name": "single_domain_qa"})
            reset_ok = reset.status_code == 200 and "observation" in reset.json()
            print_check("POST /reset", reset_ok)
            all_passed &= reset_ok

            initial_observation = reset.json().get("observation", {})
            first_chunk_id = None
            for chunk in initial_observation.get("available_chunks", []):
                if chunk.get("chunk_id"):
                    first_chunk_id = chunk["chunk_id"]
                    break
            step_payload = {"action_type": "select_chunk", "chunk_id": first_chunk_id} if first_chunk_id else {
                "action_type": "submit_answer",
                "answer": "No chunk available for validation.",
            }
            step = client.post(f"{base_url}/step", json=step_payload)
            step_ok = step.status_code == 200 and isinstance(step.json().get("reward"), float)
            print_check("POST /step", step_ok)
            all_passed &= step_ok

            state = client.get(f"{base_url}/state")
            state_ok = state.status_code == 200 and "selected_chunks" in state.json()
            print_check("GET /state", state_ok)
            all_passed &= state_ok

            optimize_prompt = client.post(
                f"{base_url}/optimize-prompt",
                json={
                    "prompt": "Draft a customer-safe admin compromise update with rollback safeguards and cite evidence.",
                    "corpus_family": "enterprise_v2",
                    "compression_mode": "grounded",
                },
            )
            optimize_body = optimize_prompt.json() if optimize_prompt.status_code == 200 else {}
            optimize_ok = (
                optimize_prompt.status_code == 200
                and "optimized_prompt" in optimize_body
                and "context_tuning" in optimize_body
                and "grounding" in optimize_body
                and optimize_body.get("optimization_mode") == "grounded"
                and bool(optimize_body.get("grounding", {}).get("citation_ready"))
            )
            print_check("POST /optimize-prompt", optimize_ok)
            all_passed &= optimize_ok

            inference_ok = run_inference_script(base_url)
            print_check("python inference.py", inference_ok)
            all_passed &= inference_ok

            for task_name in TASKS:
                passed, _ = run_task(client, base_url, task_name)
                all_passed &= passed

        return 0 if all_passed else 1
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    process.kill()
                else:
                    process.send_signal(signal.SIGKILL)


if __name__ == "__main__":
    raise SystemExit(main())
