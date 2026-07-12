import pytest
from unittest.mock import AsyncMock, patch

from env.llm_services import suggest_action, judge_answer, rewrite_prompt
from env.models import RagObservation
from env.tasks import Task
from env.llm_runtime import JsonCallResult


@pytest.fixture
def mock_call_json():
    with patch("env.llm_services.call_json", new_callable=AsyncMock) as mock:
        yield mock


@pytest.mark.asyncio
async def test_suggest_action(mock_call_json):
    mock_call_json.return_value = JsonCallResult(
        data={"action_type": "submit_answer", "answer": "test"},
        prompt_tokens=10,
        completion_tokens=10,
    )

    obs = RagObservation(
        task_name="test_task",
        case_id="1",
        case_summary="summary",
        query="query",
        objective="query",
        workflow_stage="triage",
        customer_tier="standard",
        incident_severity="sev3",
        available_chunks=[],
        selected_chunks=[],
        reviewed_artifacts=[],
        available_artifacts=[],
        prioritized_artifacts=[],
        report_requirements=[],
        progress_signals={},
        token_budget=100,
        total_tokens_used=10,
        step_number=1,
        last_action_feedback=None,
        plan_draft=None,
    )

    result = await suggest_action(
        obs, selected_chunk_details=[], suggested_citations=[], top_demo_cases=[]
    )

    assert result == {"action_type": "submit_answer", "answer": "test"}
    mock_call_json.assert_called_once()


@pytest.mark.asyncio
async def test_judge_answer(mock_call_json):
    mock_call_json.return_value = JsonCallResult(
        data={
            "answer_quality": 0.8,
            "groundedness": 0.9,
            "coverage": 1.0,
            "citation_support": 0.5,
            "notes": "Good",
        },
        prompt_tokens=10,
        completion_tokens=10,
    )

    task = Task(
        name="test_task",
        description="desc",
        difficulty="easy",
        query="query",
        token_budget=100,
        max_steps=5,
        required_artifact_ids=[],
        expected_citation_ids=[],
        case_summary="summary",
        customer_tier="standard",
        incident_severity="sev3",
        required_plan_keywords=[],
        required_report_keywords=[],
        report_requirements=[],
        domain_filter=None,
    )

    result = await judge_answer(
        task=task, answer="answer", selected_chunks=[], required_chunks=[]
    )

    assert result["answer_quality"] == pytest.approx(0.8, rel=1e-5)
    assert result["groundedness"] == pytest.approx(0.9, rel=1e-5)
    assert result["coverage"] == pytest.approx(1.0, rel=1e-5)
    assert result["citation_support"] == pytest.approx(0.5, rel=1e-5)
    assert result["notes"] == "Good"


@pytest.mark.asyncio
async def test_rewrite_prompt(mock_call_json):
    mock_call_json.return_value = JsonCallResult(
        data={
            "optimized_prompt": "opt",
            "estimated_tokens": 50,
            "citation_ready": True,
            "citation_guidance": "guide",
        },
        prompt_tokens=10,
        completion_tokens=10,
    )

    result = await rewrite_prompt(
        prompt="prompt",
        mode="test",
        target_tokens=100,
        evidence_notes=[],
        citation_ids=[],
    )

    assert result["optimized_prompt"] == "opt"
    assert result["estimated_tokens"] == 50
    assert result["citation_ready"] is True
    assert result["citation_guidance"] == "guide"
