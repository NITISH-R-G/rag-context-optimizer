"""
Baseline inference runner for the rag-context-optimizer HTTP environment.
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

ENV_NAME = "rag-context-optimizer"
ALLOW_BASELINE_FALLBACK = os.getenv("ALLOW_BASELINE_FALLBACK", "").strip().lower() in {"1", "true", "yes"}
RAG_ENV_TASK = os.getenv("RAG_ENV_TASK", "single_domain_qa")
RAG_ENV_URL = os.getenv("RAG_ENV_URL", "http://localhost:7860")
TASK_SEQUENCE = [
    "single_domain_qa",
    "cross_domain_synthesis",
    "adversarial_compression",
]

SYSTEM_PROMPT = """You are a baseline RAG context optimizer.
Read the query and available chunks using chunk_id, keywords, tokens, and domain.
Select chunks that maximize keyword overlap with the query.
Stay under the token budget.
Compress chunks that are mildly relevant but token-heavy.
Submit a concise answer once enough useful chunks are selected.
When you submit an answer, cite selected chunks inline like [support_003] or [incident_002].
Return only valid JSON matching one of these forms:
{"action_type":"select_chunk","chunk_id":"support_003"}
{"action_type":"deselect_chunk","chunk_id":"support_003"}
{"action_type":"compress_chunk","chunk_id":"support_003","compression_ratio":0.5}
{"action_type":"submit_answer","answer":"Verify outage evidence and the billing ledger before refunding [support_001] [support_003]."}"""

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
    if value is None:
        return "0.00"
    return f"{value:.2f}"


def _format_error(error: str | None) -> str:
    return "null" if not error else error.replace("\n", " ").strip()


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def _format_rewards(rewards: list[float]) -> str:
    return ",".join(f"{reward:.2f}" for reward in rewards)


def _format_action(action: dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=True, separators=(",", ":"))


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
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
    if not union:
        return 0.0
    return len(query_terms & keyword_terms) / len(union)


def _fallback_answer(observation: dict[str, Any]) -> str:
    selected = set(observation.get("selected_chunks", []))
    snippets: list[str] = []
    for chunk in observation.get("available_chunks", []):
        if chunk.get("chunk_id") in selected:
            keywords = ", ".join(chunk.get("keywords", [])[:3])
            if keywords:
                snippets.append(f"[{chunk['chunk_id']}] highlights {keywords}")
    if not snippets:
        return "The most relevant evidence points to a small set of grounded operational practices."
    citation_suffix = " ".join(f"[{chunk.get('chunk_id')}]" for chunk in observation.get("available_chunks", []) if chunk.get("chunk_id") in selected)
    answer = "; ".join(snippets[:3]) + "."
    if citation_suffix:
        answer = answer.rstrip(".") + " " + citation_suffix + "."
    return answer


def _fallback_action(observation: dict[str, Any]) -> dict[str, Any]:
    selected = set(observation.get("selected_chunks", []))
    available = list(observation.get("available_chunks", []))
    token_budget = observation["token_budget"]
    total_tokens_used = observation["total_tokens_used"]
    remaining_budget = token_budget - total_tokens_used

    ranked = sorted(
        available,
        key=lambda chunk: (
            -_keyword_overlap(observation["query"], chunk),
            chunk["tokens"],
            chunk["chunk_id"],
        ),
    )

    if selected and total_tokens_used >= int(token_budget * 0.75):
        heavy_selected = [
            chunk for chunk in ranked if chunk["chunk_id"] in selected and chunk["tokens"] >= max(120, token_budget // 4)
        ]
        if heavy_selected:
            return {
                "action_type": "compress_chunk",
                "chunk_id": heavy_selected[0]["chunk_id"],
                "compression_ratio": 0.5,
            }

    if len(selected) >= 3 or observation["step_number"] >= 3 or remaining_budget <= max(80, token_budget // 6):
        return {"action_type": "submit_answer", "answer": _fallback_answer(observation)}

    for chunk in ranked:
        if chunk["chunk_id"] in selected:
            continue
        if chunk["tokens"] <= remaining_budget:
            return {"action_type": "select_chunk", "chunk_id": chunk["chunk_id"]}

    if selected:
        return {"action_type": "submit_answer", "answer": _fallback_answer(observation)}

    best_chunk = ranked[0] if ranked else None
    if best_chunk is not None:
        return {"action_type": "select_chunk", "chunk_id": best_chunk["chunk_id"]}
    return {"action_type": "submit_answer", "answer": "No relevant chunks were available."}


def _build_user_prompt(observation: dict[str, Any]) -> str:
    payload = {
        "query": observation["query"],
        "task_name": observation["task_name"],
        "step_number": observation["step_number"],
        "selected_chunks": observation["selected_chunks"],
        "total_tokens_used": observation["total_tokens_used"],
        "token_budget": observation["token_budget"],
        "last_action_feedback": observation.get("last_action_feedback"),
        "available_chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "domain": chunk["domain"],
                "tokens": chunk["tokens"],
                "keywords": chunk["keywords"],
            }
            for chunk in observation["available_chunks"]
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
            f"[warn] Missing API_BASE_URL/API_KEY credentials; aborting model-backed run for {task_name}. Set ALLOW_BASELINE_FALLBACK=1 only for offline smoke testing.",
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
                action_payload: dict[str, Any]
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
                        print(
                            f"[END] success=false steps={steps} score={_clamp_score(score):.3f} rewards={_format_rewards(rewards)}",
                        )
                        return score, rewards, steps, False
                    print(
                        f"[warn] Falling back to deterministic policy for {task_name}: {fallback_reason}",
                        file=sys.stderr,
                        flush=True,
                    )
                    action_payload = _fallback_action(observation)
                    action_payload = RagAction.model_validate(action_payload).model_dump(exclude_none=True)

                try:
                    step_response = await _post_json(http_client, "/step", action_payload)
                except Exception as exc:
                    steps += 1
                    rewards.append(0.0)
                    terminal_error = str(exc)
                    print(
                        f"[STEP] step={steps} action={_format_action(action_payload)} "
                        f"reward=0.00 done=true error={_format_error(terminal_error)}"
                    )
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
                    success = terminal_error is None and fallback_reason is None
                    break

            score = _clamp_score(score)
            print(
                f"[END] success={_format_bool(success)} steps={steps} score={score:.3f} rewards={_format_rewards(rewards)}"
            )
            return score, rewards, steps, success
    except Exception:
        score = _clamp_score(score)
        print(
            f"[END] success=false steps={steps} score={score:.3f} rewards={_format_rewards(rewards)}"
        )
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
