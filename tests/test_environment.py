from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.environment import RagContextOptimizerEnv
from env.models import RagAction
from env.tasks import ALL_TASKS, TASK_EASY, TASK_HARD


def _run(coro):
    return asyncio.run(coro)


def _find_chunk(observation, chunk_id: str):
    for chunk in observation.available_chunks:
        if chunk.chunk_id == chunk_id:
            return chunk
    raise AssertionError(f"Chunk {chunk_id} not found")


def _smallest_unselected_chunk(observation):
    selected = set(observation.selected_chunks)
    candidates = [chunk for chunk in observation.available_chunks if chunk.chunk_id not in selected]
    return min(candidates, key=lambda chunk: (chunk.tokens, chunk.chunk_id))


def _largest_unselected_chunk(observation):
    selected = set(observation.selected_chunks)
    candidates = [chunk for chunk in observation.available_chunks if chunk.chunk_id not in selected]
    return max(candidates, key=lambda chunk: (chunk.tokens, chunk.chunk_id))


def _average_random_agent_score(task_name: str, runs: int = 5) -> float:
    scores: list[float] = []
    for seed in range(runs):
        rng = random.Random(seed)
        env = RagContextOptimizerEnv(task_name)
        result = _run(env.reset())

        while not result.done:
            observation = result.observation
            selected = set(observation.selected_chunks)
            available = [chunk for chunk in observation.available_chunks if chunk.chunk_id not in selected]
            if observation.step_number >= 2 or len(selected) >= 2 or not available:
                action = RagAction(
                    action_type="submit_answer",
                    answer="A short baseline answer using the currently selected evidence.",
                )
            else:
                choice = rng.choice(available)
                action = RagAction(action_type="select_chunk", chunk_id=choice.chunk_id)
            result = _run(env.step(action))
        scores.append(result.reward)
    return sum(scores) / len(scores)


def test_reset_returns_valid_observation():
    for task in ALL_TASKS:
        env = RagContextOptimizerEnv(task.name)
        result = _run(env.reset())
        assert result.observation.query
        assert result.observation.available_chunks
        assert result.observation.token_budget > 0


def test_select_chunk_within_budget():
    env = RagContextOptimizerEnv(TASK_EASY.name)
    reset_result = _run(env.reset())
    chunk = _smallest_unselected_chunk(reset_result.observation)

    step_result = _run(env.step(RagAction(action_type="inspect_artifact", artifact_id=chunk.chunk_id)))
    assert chunk.chunk_id in step_result.observation.reviewed_artifacts
    prioritized_result = _run(env.step(RagAction(action_type="prioritize_artifact", artifact_id=chunk.chunk_id)))
    assert chunk.chunk_id in prioritized_result.observation.selected_chunks
    assert prioritized_result.observation.total_tokens_used >= chunk.tokens
    assert step_result.reward > 0


def test_select_chunk_over_budget_penalized():
    env = RagContextOptimizerEnv(TASK_HARD.name)
    result = _run(env.reset())

    while True:
        observation = result.observation
        largest = _largest_unselected_chunk(observation)
        if observation.total_tokens_used + largest.tokens > observation.token_budget:
            overflow_chunk = largest
            break
        result = _run(env.step(RagAction(action_type="inspect_artifact", artifact_id=largest.chunk_id)))
        result = _run(env.step(RagAction(action_type="prioritize_artifact", artifact_id=largest.chunk_id)))

    previous_selected = list(result.observation.selected_chunks)
    overflow_result = _run(env.step(RagAction(action_type="inspect_artifact", artifact_id=overflow_chunk.chunk_id)))
    overflow_result = _run(env.step(RagAction(action_type="prioritize_artifact", artifact_id=overflow_chunk.chunk_id)))
    assert overflow_result.reward < 0
    assert overflow_chunk.chunk_id not in overflow_result.observation.selected_chunks
    assert overflow_result.observation.selected_chunks == previous_selected


def test_compress_chunk_reduces_tokens():
    env = RagContextOptimizerEnv(TASK_EASY.name)
    reset_result = _run(env.reset())
    chunk = _smallest_unselected_chunk(reset_result.observation)

    selected_result = _run(env.step(RagAction(action_type="inspect_artifact", artifact_id=chunk.chunk_id)))
    selected_result = _run(env.step(RagAction(action_type="prioritize_artifact", artifact_id=chunk.chunk_id)))
    before_tokens = selected_result.observation.total_tokens_used

    compressed_result = _run(
        env.step(RagAction(action_type="summarize_artifact", artifact_id=chunk.chunk_id, compression_ratio=0.5))
    )
    after_tokens = compressed_result.observation.total_tokens_used
    assert after_tokens <= before_tokens // 2 + 1
    assert after_tokens < before_tokens


def test_submit_answer_ends_episode():
    env = RagContextOptimizerEnv(TASK_EASY.name)
    result = _run(env.reset())
    for chunk_id in TASK_EASY.required_artifact_ids[:2]:
        result = _run(env.step(RagAction(action_type="inspect_artifact", artifact_id=chunk_id)))
        result = _run(env.step(RagAction(action_type="prioritize_artifact", artifact_id=chunk_id)))
    result = _run(
        env.step(RagAction(action_type="set_resolution_plan", plan="Verify outage evidence, confirm the billing ledger, and route manual exceptions to finance review."))
    )

    final_result = _run(
            env.step(
                RagAction(
                    action_type="submit_report",
                    answer="Proceed to refund review only after outage evidence and the billing ledger are confirmed, then route exceptions to finance review. [support_001] [support_003]",
                )
            )
        )
    assert final_result.done is True
    assert 0.0 <= final_result.reward <= 1.0


def test_grader_deterministic():
    def run_sequence():
        env = RagContextOptimizerEnv(TASK_EASY.name)
        result = _run(env.reset())
        result = _run(env.step(RagAction(action_type="inspect_artifact", artifact_id="support_003")))
        result = _run(env.step(RagAction(action_type="prioritize_artifact", artifact_id="support_003")))
        result = _run(env.step(RagAction(action_type="inspect_artifact", artifact_id="support_005")))
        result = _run(env.step(RagAction(action_type="prioritize_artifact", artifact_id="support_005")))
        result = _run(env.step(RagAction(action_type="set_resolution_plan", plan="Verify outage evidence, confirm the billing ledger, and route manual exceptions to finance review.")))
        result = _run(
            env.step(
                RagAction(
                    action_type="submit_report",
                    answer="Support should confirm the outage timeline, verify the charge in the billing ledger, and use the compensation matrix before finance review. [support_003] [support_005]",
                )
            )
        )
        return result.reward

    assert run_sequence() == run_sequence()


def test_all_tasks_reachable():
    for task in ALL_TASKS:
        env = RagContextOptimizerEnv(task.name)
        result = _run(env.reset())
        assert result.observation.task_name == task.name


def test_hard_task_harder_than_easy():
    easy_score = _average_random_agent_score(TASK_EASY.name, runs=5)
    hard_score = _average_random_agent_score(TASK_HARD.name, runs=5)
    assert easy_score > hard_score
