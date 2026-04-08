"""
Baseline inference runner for the incident operations HTTP environment.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from typing import Any

import httpx
from openai import OpenAI

from env.models import RagAction

ENV_NAME = "incident-ops-env"
ALLOW_BASELINE_FALLBACK = os.getenv("ALLOW_BASELINE_FALLBACK", "").strip().lower() in {"1", "true", "yes"}
RAG_ENV_TASK = os.getenv("RAG_ENV_TASK", "refund_triage_easy")
RAG_ENV_URL = os.getenv("RAG_ENV_URL", "http://localhost:7860")
TASK_SEQUENCE = [
    "refund_triage_easy",
    "cross_function_brief_medium",
    "executive_escalation_hard",
]

SYSTEM_PROMPT = """You are a baseline incident operations agent.
You are handling an enterprise case through staged workflow actions.
Your job is to inspect the right artifacts, prioritize the evidence that belongs in the working set,
draft a short operational plan, summarize heavy artifacts when needed, and finally submit a grounded report.
Return only valid JSON matching one of these forms:
{"action_type":"inspect_artifact","artifact_id":"support_003"}
{"action_type":"prioritize_artifact","artifact_id":"support_003"}
{"action_type":"summarize_artifact","artifact_id":"support_003","compression_ratio":0.55}
{"action_type":"set_resolution_plan","plan":"Verify outage evidence, confirm the billing ledger, and route manual exceptions to finance review."}
{"action_type":"submit_report","answer":"Proceed to refund review only after outage and billing evidence are confirmed. [support_001] [support_003]"}
Legacy aliases like select_chunk, compress_chunk, and submit_answer are also accepted, but prefer the new workflow actions."""

DEFAULT_LEGACY_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"


def _model_name() -> str:
    return os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)


def _resolve_llm_credentials() -> tuple[str | None, str | None, str | None]:
    api_base_url = os.getenv("API_BASE_URL", DEFAULT_LEGACY_BASE_URL)
    api_key = os.getenv("API_KEY")
    legacy_token = os.getenv("HF_TOKEN")
    if api_key:
        return api_base_url, api_key, "proxy"
    if legacy_token:
        return api_base_url, legacy_token, "legacy"
    return None, None, None


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_reward(value: float | None) -> str:
    return "0.00" if value is None else f"{value:.2f}"


def _format_error(error: str | None) -> str:
    return "null" if not error else error.replace("\n", " ").strip()


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def _format_rewards(rewards: list[float]) -> str:
    return ",".join(f"{reward:.2f}" for reward in rewards)


def _format_action(action: dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=True, separators=(",", ":"))


def _extract_json_object(text: str) -> dict[str, Any]:
    payload = text.strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", payload, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _keyword_overlap(query: str, chunk: dict[str, Any]) -> float:
    query_terms = _tokenize(query)
    keyword_terms = _tokenize(" ".join(chunk.get("keywords", [])))
    if not query_terms or not keyword_terms:
        return 0.0
    union = query_terms | keyword_terms
    return (len(query_terms & keyword_terms) / len(union)) if union else 0.0


def _fallback_report(observation: dict[str, Any]) -> str:
    prioritized = set(observation.get("prioritized_artifacts") or observation.get("selected_chunks", []))
    snippets: list[str] = []
    for chunk in observation.get("available_artifacts") or observation.get("available_chunks", []):
        if chunk.get("chunk_id") in prioritized:
            keywords = ", ".join(chunk.get("keywords", [])[:3])
            snippets.append(f"[{chunk['chunk_id']}] covers {keywords}")
    if not snippets:
        return "The case needs a defensible operational recommendation grounded in reviewed incident artifacts."
    return "; ".join(snippets[:3]) + "."


def _fallback_plan(observation: dict[str, Any]) -> str:
    task_name = observation.get("task_name", "")
    if task_name == "refund_triage_easy":
        return "Verify outage evidence, confirm the billing ledger, and route manual exceptions to finance review."
    if task_name == "cross_function_brief_medium":
        return "Align the incident timeline, customer communications, and rollback guardrails before publishing the brief."
    return "Revoke active risk, protect customers, preserve evidence, and keep change freeze safeguards in place."


def _fallback_action(observation: dict[str, Any]) -> dict[str, Any]:
    reviewed = set(observation.get("reviewed_artifacts", []))
    prioritized = set(observation.get("prioritized_artifacts") or observation.get("selected_chunks", []))
    available = list(observation.get("available_artifacts") or observation.get("available_chunks", []))
    token_budget = observation["token_budget"]
    total_tokens_used = observation["total_tokens_used"]
    remaining_budget = token_budget - total_tokens_used

    ranked = sorted(
        available,
        key=lambda chunk: (-_keyword_overlap(observation["query"], chunk), chunk["tokens"], chunk["chunk_id"]),
    )

    unprioritized_reviewed = [chunk for chunk in ranked if chunk["chunk_id"] in reviewed and chunk["chunk_id"] not in prioritized]
    for chunk in unprioritized_reviewed:
        if chunk["tokens"] <= remaining_budget:
            return {"action_type": "prioritize_artifact", "artifact_id": chunk["chunk_id"]}

    unseen = [chunk for chunk in ranked if chunk["chunk_id"] not in reviewed]
    if unseen:
        if len(reviewed) >= 2:
            unseen = unseen[:1]
        return {"action_type": "inspect_artifact", "artifact_id": unseen[0]["chunk_id"]}

    if prioritized and not observation.get("plan_draft"):
        return {"action_type": "set_resolution_plan", "plan": _fallback_plan(observation)}

    heavy_prioritized = [chunk for chunk in ranked if chunk["chunk_id"] in prioritized and chunk["tokens"] >= max(120, token_budget // 4)]
    if heavy_prioritized and total_tokens_used >= int(token_budget * 0.7):
        return {"action_type": "summarize_artifact", "artifact_id": heavy_prioritized[0]["chunk_id"], "compression_ratio": 0.55}

    return {"action_type": "submit_report", "answer": _fallback_report(observation)}


def _build_user_prompt(observation: dict[str, Any]) -> str:
    payload = {
        "case_id": observation.get("case_id"),
        "case_summary": observation.get("case_summary"),
        "objective": observation.get("objective") or observation.get("query"),
        "workflow_stage": observation.get("workflow_stage"),
        "customer_tier": observation.get("customer_tier"),
        "incident_severity": observation.get("incident_severity"),
        "reviewed_artifacts": observation.get("reviewed_artifacts", []),
        "prioritized_artifacts": observation.get("prioritized_artifacts") or observation.get("selected_chunks", []),
        "plan_draft": observation.get("plan_draft"),
        "report_requirements": observation.get("report_requirements", []),
        "progress_signals": observation.get("progress_signals", {}),
        "total_tokens_used": observation["total_tokens_used"],
        "token_budget": observation["token_budget"],
        "step_number": observation["step_number"],
        "task_name": observation["task_name"],
        "last_action_feedback": observation.get("last_action_feedback"),
        "available_artifacts": [
            {
                "chunk_id": chunk["chunk_id"],
                "domain": chunk["domain"],
                "tokens": chunk["tokens"],
                "keywords": chunk["keywords"],
            }
            for chunk in (observation.get("available_artifacts") or observation.get("available_chunks", []))
        ],
    }
    return json.dumps(payload, ensure_ascii=True)


async def _llm_action(client: OpenAI, observation: dict[str, Any]) -> dict[str, Any]:
    prompt = _build_user_prompt(observation)
    model_name = _model_name()

    def _call() -> Any:
        return client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

    response = await asyncio.to_thread(_call)
    content = response.choices[0].message.content or "{}"
    return _extract_json_object(content)


async def _post_json(http_client: httpx.AsyncClient, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = await http_client.post(f"{RAG_ENV_URL}{path}", json=payload)
    if response.status_code >= 400:
        raise RuntimeError(f"{path} -> {response.status_code}: {response.text}")
    return response.json()


async def _run_task_http(task_name: str) -> tuple[float, list[float], int, bool]:
    rewards: list[float] = []
    steps = 0
    success = False
    score = 0.0
    terminal_error: str | None = None
    fallback_reason: str | None = None
    model_name = _model_name()

    print(f"[START] task={task_name} env={ENV_NAME} model={model_name}")

    api_base_url, client_api_key, auth_mode = _resolve_llm_credentials()
    llm_required = auth_mode in {"proxy", "legacy"}
    openai_client: Any | None = None

    if llm_required:
        openai_client = OpenAI(base_url=api_base_url, api_key=client_api_key)
    elif ALLOW_BASELINE_FALLBACK:
        fallback_reason = "missing_llm_credentials"
        print(
            f"[warn] Missing API_BASE_URL/API_KEY credentials; using deterministic fallback policy for {task_name}.",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            "[warn] Missing API_BASE_URL/API_KEY credentials; aborting model-backed run. "
            "Set ALLOW_BASELINE_FALLBACK=1 only for offline smoke testing.",
            file=sys.stderr,
            flush=True,
        )
        print("[END] success=false steps=0 score=0.000 rewards=")
        return 0.0, [], 0, False

    try:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            reset_payload = await _post_json(http_client, "/reset", {"task_name": task_name})
            observation = reset_payload["observation"]

            while True:
                step_error: str | None = None
                try:
                    if openai_client is None:
                        raise RuntimeError("llm_unavailable")
                    llm_payload = await _llm_action(openai_client, observation)
                    action_payload = RagAction.model_validate(llm_payload).model_dump(exclude_none=True)
                except Exception as exc:
                    fallback_reason = fallback_reason or type(exc).__name__
                    if llm_required or not ALLOW_BASELINE_FALLBACK:
                        terminal_error = f"model_unavailable:{fallback_reason}"
                        print(f"[END] success=false steps={steps} score={_clamp_score(score):.3f} rewards={_format_rewards(rewards)}")
                        return score, rewards, steps, False
                    action_payload = RagAction.model_validate(_fallback_action(observation)).model_dump(exclude_none=True)

                try:
                    step_response = await _post_json(http_client, "/step", action_payload)
                except Exception as exc:
                    steps += 1
                    rewards.append(0.0)
                    terminal_error = str(exc)
                    print(f"[STEP] step={steps} action={_format_action(action_payload)} reward=0.00 done=true error={_format_error(terminal_error)}")
                    break

                steps += 1
                reward_value = step_response.get("reward")
                reward_float = float(reward_value) if reward_value is not None else 0.0
                rewards.append(reward_float)
                done = bool(step_response["done"])
                print(
                    f"[STEP] step={steps} action={_format_action(action_payload)} "
                    f"reward={_format_reward(reward_float)} done={_format_bool(done)} error={_format_error(step_error)}"
                )
                observation = step_response["observation"]
                if done:
                    score = _clamp_score(reward_float)
                    success = terminal_error is None
                    break

            print(f"[END] success={_format_bool(success)} steps={steps} score={score:.3f} rewards={_format_rewards(rewards)}")
            return score, rewards, steps, success
    except Exception:
        print(f"[END] success=false steps={steps} score={_clamp_score(score):.3f} rewards={_format_rewards(rewards)}")
        return score, rewards, steps, False


def run_task(task_name: str) -> tuple[float, list[float], int, bool]:
    return asyncio.run(_run_task_http(task_name))


def main() -> int:
    if RAG_ENV_TASK in TASK_SEQUENCE:
        tasks = [RAG_ENV_TASK] + [task for task in TASK_SEQUENCE if task != RAG_ENV_TASK]
    else:
        tasks = list(TASK_SEQUENCE)
    all_success = True
    for task_name in tasks:
        _score, _rewards, _steps, success = run_task(task_name)
        all_success &= success
    return 0 if all_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
