import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json  # noqa: E402
from inference import _fallback_plan, _fallback_action, _build_user_prompt  # noqa: E402

def test_fallback_plan():
    assert "Verify outage evidence" in _fallback_plan({"task_name": "refund_triage_easy"})
    assert "Align the incident timeline" in _fallback_plan({"task_name": "cross_function_brief_medium"})
    assert "Revoke active risk" in _fallback_plan({"task_name": "other_task"})
    assert "Revoke active risk" in _fallback_plan({})

def test_fallback_action_prioritize():
    observation = {
        "query": "test query",
        "token_budget": 100,
        "total_tokens_used": 10,
        "reviewed_artifacts": ["chunk1"],
        "prioritized_artifacts": [],
        "available_artifacts": [
            {"chunk_id": "chunk1", "tokens": 20, "keywords": ["test"]},
            {"chunk_id": "chunk2", "tokens": 20, "keywords": ["test"]}
        ]
    }
    action = _fallback_action(observation)
    assert action["action_type"] == "prioritize_artifact"
    assert action["artifact_id"] == "chunk1"

def test_fallback_action_inspect():
    observation = {
        "query": "test query",
        "token_budget": 100,
        "total_tokens_used": 10,
        "reviewed_artifacts": [],
        "prioritized_artifacts": [],
        "available_artifacts": [
            {"chunk_id": "chunk1", "tokens": 20, "keywords": ["test"]},
            {"chunk_id": "chunk2", "tokens": 20, "keywords": ["test"]}
        ]
    }
    action = _fallback_action(observation)
    assert action["action_type"] == "inspect_artifact"

def test_fallback_action_inspect_limit():
    observation = {
        "query": "test query",
        "token_budget": 100,
        "total_tokens_used": 10,
        "reviewed_artifacts": ["chunk2", "chunk3"],
        "prioritized_artifacts": ["chunk2", "chunk3"], # Make both prioritized so prioritize branch is skipped
        "available_artifacts": [
            {"chunk_id": "chunk1", "tokens": 20, "keywords": ["test"]},
            {"chunk_id": "chunk2", "tokens": 20, "keywords": ["test"]},
            {"chunk_id": "chunk3", "tokens": 20, "keywords": ["test"]}
        ]
    }
    action = _fallback_action(observation)
    assert action["action_type"] == "inspect_artifact"
    # Should only return one of the unseen due to slicing
    assert action["artifact_id"] == "chunk1"

def test_fallback_action_set_plan():
    observation = {
        "query": "test query",
        "token_budget": 100,
        "total_tokens_used": 10,
        "reviewed_artifacts": ["chunk1"],
        "prioritized_artifacts": ["chunk1"],
        "available_artifacts": [
            {"chunk_id": "chunk1", "tokens": 20, "keywords": ["test"]},
        ],
        "task_name": "refund_triage_easy"
    }
    action = _fallback_action(observation)
    assert action["action_type"] == "set_resolution_plan"
    assert "Verify outage evidence" in action["plan"]

def test_fallback_action_summarize():
    observation = {
        "query": "test query",
        "token_budget": 100,
        "total_tokens_used": 80, # > 70%
        "reviewed_artifacts": ["chunk1"],
        "prioritized_artifacts": ["chunk1"],
        "plan_draft": "Some plan",
        "available_artifacts": [
            {"chunk_id": "chunk1", "tokens": 120, "keywords": ["test"]}, # Heavy > 120
        ],
    }
    action = _fallback_action(observation)
    assert action["action_type"] == "summarize_artifact"
    assert action["artifact_id"] == "chunk1"

def test_fallback_action_submit():
    observation = {
        "query": "test query",
        "token_budget": 100,
        "total_tokens_used": 10,
        "reviewed_artifacts": ["chunk1"],
        "prioritized_artifacts": ["chunk1"],
        "plan_draft": "Some plan",
        "available_artifacts": [
            {"chunk_id": "chunk1", "tokens": 20, "keywords": ["test"]},
        ],
    }
    action = _fallback_action(observation)
    assert action["action_type"] == "submit_report"

def test_build_user_prompt():
    observation = {
        "case_id": "123",
        "case_summary": "Test summary",
        "objective": "Test objective",
        "workflow_stage": "triage",
        "customer_tier": "standard",
        "incident_severity": "sev3",
        "reviewed_artifacts": [],
        "prioritized_artifacts": [],
        "plan_draft": None,
        "report_requirements": [],
        "progress_signals": {},
        "total_tokens_used": 0,
        "token_budget": 100,
        "step_number": 1,
        "task_name": "test_task",
        "available_artifacts": [
            {"chunk_id": "chunk1", "domain": "test", "tokens": 10, "keywords": ["test"]}
        ]
    }
    prompt = _build_user_prompt(observation)
    parsed = json.loads(prompt)
    assert parsed["case_id"] == "123"
    assert parsed["step_number"] == 1
    assert len(parsed["available_artifacts"]) == 1
