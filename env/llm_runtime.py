"""
Shared OpenAI-compatible runtime helpers for LLM-backed benchmark features.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"


@dataclass(frozen=True, slots=True)
class JsonCallResult:
    data: dict[str, Any]
    prompt_tokens: int | None
    completion_tokens: int | None


def model_name() -> str:
    return os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)


def resolve_llm_credentials() -> tuple[str | None, str | None, str | None]:
    api_base_url = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
    api_key = os.getenv("API_KEY")
    legacy_token = os.getenv("HF_TOKEN")

    if api_key:
        return api_base_url, api_key, "proxy"
    if legacy_token:
        return api_base_url, legacy_token, "legacy"
    return None, None, None


def llm_configured() -> bool:
    _base_url, api_key, _auth_mode = resolve_llm_credentials()
    return bool(api_key)


def _extract_json_object(text: str) -> dict[str, Any]:
    payload = text.strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", payload, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


async def call_json(
    *,
    system_prompt: str,
    user_payload: dict[str, Any] | list[Any] | str,
    temperature: float = 0.0,
    max_output_tokens: int = 400,
) -> JsonCallResult:
    api_base_url, client_api_key, _auth_mode = resolve_llm_credentials()
    if not api_base_url or not client_api_key:
        raise RuntimeError("llm_credentials_missing")

    client = OpenAI(base_url=api_base_url, api_key=client_api_key)
    user_content = user_payload if isinstance(user_payload, str) else json.dumps(user_payload, ensure_ascii=True)

    def _call():
        return client.chat.completions.create(
            model=model_name(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_output_tokens,
        )

    response = await asyncio.to_thread(_call)
    content = response.choices[0].message.content or "{}"
    usage = getattr(response, "usage", None)
    return JsonCallResult(
        data=_extract_json_object(content),
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
    )


async def estimate_tokens(text: str) -> int:
    cleaned = text.strip()
    if not cleaned:
        return 0

    if not llm_configured():
        return max(1, len(cleaned) // 4)

    result = await call_json(
        system_prompt=(
            "You are TOKEN_ESTIMATOR. Estimate how many model tokens the provided text would use "
            "for the current chat model. Return JSON with exactly one integer field: "
            '{"token_count": 123}'
        ),
        user_payload={"text": cleaned},
        temperature=0.0,
        max_output_tokens=32,
    )
    token_count = int(result.data.get("token_count", 0))
    return max(1, token_count)
