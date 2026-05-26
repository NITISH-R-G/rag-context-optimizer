from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import (  # noqa: E402
    _build_user_prompt,
    _clamp_score,
    _extract_json_object,
    _fallback_action,
    _fallback_plan,
    _fallback_report,
    _format_action,
    _format_bool,
    _format_error,
    _format_reward,
    _format_rewards,
    _keyword_overlap,
    _model_name,
    _resolve_llm_credentials,
    _tokenize,
)


def test_format_bool():
    assert _format_bool(True) == "true"
    assert _format_bool(False) == "false"


def test_format_reward():
    assert _format_reward(None) == "0.00"
    assert _format_reward(0.0) == "0.00"
    assert _format_reward(0.123) == "0.12"
    assert _format_reward(1.0) == "1.00"


def test_format_error():
    assert _format_error(None) == "null"
    assert _format_error("") == "null"
    assert _format_error("error\nmessage") == "error message"


def test_clamp_score():
    assert _clamp_score(-0.5) == 0.0
    assert _clamp_score(0.5) == 0.5
    assert _clamp_score(1.5) == 1.0


def test_format_rewards():
    assert _format_rewards([]) == ""
    assert _format_rewards([0.1, 0.25, 0.333]) == "0.10,0.25,0.33"


def test_format_action():
    action = {"action_type": "inspect_artifact", "artifact_id": "support_003"}
    assert _format_action(action) == '{"action_type":"inspect_artifact","artifact_id":"support_003"}'


def test_extract_json_object():
    assert _extract_json_object('{"key": "value"}') == {"key": "value"}
    assert _extract_json_object('Some text here {"key": "value"} and more text') == {"key": "value"}
    with pytest.raises(Exception):
        _extract_json_object("invalid json")


def test_tokenize():
    assert _tokenize("Hello, World!") == {"hello", "world"}
    assert _tokenize("123 test-case") == {"123", "test", "case"}
    assert _tokenize("") == set()


def test_keyword_overlap():
    chunk = {"keywords": ["test", "case", "example"]}
    assert _keyword_overlap("test example", chunk) == 0.6666666666666666
    assert _keyword_overlap("no overlap", chunk) == 0.0
    assert _keyword_overlap("", chunk) == 0.0
    assert _keyword_overlap("test case", {"keywords": []}) == 0.0


def test_fallback_report():
    observation = {
        "prioritized_artifacts": ["support_001"],
        "available_artifacts": [
            {"chunk_id": "support_001", "keywords": ["outage", "billing", "refund"]},
            {"chunk_id": "support_002", "keywords": ["unrelated"]},
        ],
    }
    assert _fallback_report(observation) == "[support_001] covers outage, billing, refund."

    observation_empty = {}
    assert _fallback_report(observation_empty) == "The case needs a defensible operational recommendation grounded in reviewed incident artifacts."


def test_fallback_plan():
    assert _fallback_plan({"task_name": "refund_triage_easy"}) == "Verify outage evidence, confirm the billing ledger, and route manual exceptions to finance review."
    assert _fallback_plan({"task_name": "cross_function_brief_medium"}) == "Align the incident timeline, customer communications, and rollback guardrails before publishing the brief."
    assert _fallback_plan({"task_name": "unknown_task"}) == "Revoke active risk, protect customers, preserve evidence, and keep change freeze safeguards in place."


def test_fallback_action():
    # Test prioritize_artifact
    observation1 = {
        "query": "refund",
        "reviewed_artifacts": ["support_001"],
        "prioritized_artifacts": [],
        "available_artifacts": [{"chunk_id": "support_001", "tokens": 10, "keywords": ["refund"]}],
        "token_budget": 100,
        "total_tokens_used": 0,
    }
    assert _fallback_action(observation1) == {"action_type": "prioritize_artifact", "artifact_id": "support_001"}

    # Test inspect_artifact
    observation2 = {
        "query": "refund",
        "reviewed_artifacts": [],
        "prioritized_artifacts": [],
        "available_artifacts": [{"chunk_id": "support_002", "tokens": 10, "keywords": ["refund"]}],
        "token_budget": 100,
        "total_tokens_used": 0,
    }
    assert _fallback_action(observation2) == {"action_type": "inspect_artifact", "artifact_id": "support_002"}

    # Test set_resolution_plan
    observation3 = {
        "query": "refund",
        "reviewed_artifacts": ["support_001"],
        "prioritized_artifacts": ["support_001"],
        "available_artifacts": [{"chunk_id": "support_001", "tokens": 10, "keywords": ["refund"]}],
        "token_budget": 100,
        "total_tokens_used": 10,
        "plan_draft": None,
        "task_name": "refund_triage_easy"
    }
    assert _fallback_action(observation3) == {"action_type": "set_resolution_plan", "plan": "Verify outage evidence, confirm the billing ledger, and route manual exceptions to finance review."}

    # Test summarize_artifact
    observation4 = {
        "query": "refund",
        "reviewed_artifacts": ["support_001"],
        "prioritized_artifacts": ["support_001"],
        "available_artifacts": [{"chunk_id": "support_001", "tokens": 150, "keywords": ["refund"]}],
        "token_budget": 200,
        "total_tokens_used": 150,
        "plan_draft": "Some plan",
    }
    assert _fallback_action(observation4) == {"action_type": "summarize_artifact", "artifact_id": "support_001", "compression_ratio": 0.55}

    # Test submit_report
    observation5 = {
        "query": "refund",
        "reviewed_artifacts": ["support_001"],
        "prioritized_artifacts": ["support_001"],
        "available_artifacts": [{"chunk_id": "support_001", "tokens": 50, "keywords": ["refund"]}],
        "token_budget": 200,
        "total_tokens_used": 50,
        "plan_draft": "Some plan",
    }
    assert _fallback_action(observation5) == {"action_type": "submit_report", "answer": "[support_001] covers refund."}


def test_build_user_prompt():
    observation = {
        "case_id": "123",
        "total_tokens_used": 50,
        "token_budget": 100,
        "step_number": 1,
        "task_name": "task",
        "available_artifacts": [{"chunk_id": "1", "domain": "d", "tokens": 10, "keywords": ["k"]}],
    }
    prompt = _build_user_prompt(observation)
    prompt_dict = json.loads(prompt)
    assert prompt_dict["case_id"] == "123"
    assert prompt_dict["total_tokens_used"] == 50
    assert prompt_dict["token_budget"] == 100
    assert prompt_dict["step_number"] == 1
    assert prompt_dict["task_name"] == "task"
    assert len(prompt_dict["available_artifacts"]) == 1


def test_model_name(monkeypatch):
    monkeypatch.delenv("MODEL_NAME", raising=False)
    assert _model_name() == "Qwen/Qwen2.5-72B-Instruct"

    monkeypatch.setenv("MODEL_NAME", "custom-model")
    assert _model_name() == "custom-model"


def test_resolve_llm_credentials(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)

    base, key, mode = _resolve_llm_credentials()
    assert base is None
    assert key is None
    assert mode is None

    monkeypatch.setenv("API_KEY", "proxy-key")
    monkeypatch.setenv("API_BASE_URL", "http://proxy")
    base, key, mode = _resolve_llm_credentials()
    assert base == "http://proxy"
    assert key == "proxy-key"
    assert mode == "proxy"

    monkeypatch.delenv("API_KEY")
    monkeypatch.setenv("HF_TOKEN", "legacy-token")
    base, key, mode = _resolve_llm_credentials()
    assert base == "http://proxy"
    assert key == "legacy-token"
    assert mode == "legacy"
