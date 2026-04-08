"""
Higher-level LLM-backed services for planning, grading, and prompt rewriting.
"""

from __future__ import annotations

from typing import Any

from env.corpus import Chunk
from env.llm_runtime import call_json
from env.models import RagAction, RagObservation
from env.tasks import Task


async def suggest_action(
    observation: RagObservation,
    *,
    selected_chunk_details: list[dict[str, Any]],
    suggested_citations: list[str],
    top_demo_cases: list[str],
) -> dict[str, Any]:
    result = await call_json(
        system_prompt=(
            "You are ACTION_PLANNER for a grounded RAG optimization environment. "
            "Return exactly one valid RagAction JSON object. "
            "Choose among select_chunk, deselect_chunk, compress_chunk, or submit_answer. "
            "Prefer selecting the most relevant evidence first, compress only selected chunks, "
            "and submit a concise grounded answer with inline citations once evidence is sufficient."
        ),
        user_payload={
            "observation": observation.model_dump(),
            "selected_chunk_details": selected_chunk_details,
            "suggested_citations": suggested_citations,
            "top_demo_cases": top_demo_cases,
        },
        temperature=0.0,
        max_output_tokens=220,
    )
    return RagAction.model_validate(result.data).model_dump(exclude_none=True)


async def judge_answer(
    *,
    task: Task,
    answer: str,
    selected_chunks: list[Chunk],
    required_chunks: list[Chunk],
) -> dict[str, Any]:
    result = await call_json(
        system_prompt=(
            "You are ANSWER_GRADER for a grounded RAG benchmark. "
            "Evaluate whether the answer addresses the task query, covers the required evidence, "
            "and stays grounded in the provided evidence. "
            "Return JSON with numeric fields in [0,1]: "
            '{"answer_quality": 0.0, "groundedness": 0.0, "coverage": 0.0, "citation_support": 0.0, "notes": "short"}'
        ),
        user_payload={
            "task": {
                "name": task.name,
                "difficulty": task.difficulty,
                "query": task.query,
                "required_chunk_ids": task.required_chunk_ids,
                "expected_citation_ids": task.expected_citation_ids,
            },
            "answer": answer,
            "selected_evidence": [
                {
                    "chunk_id": chunk.chunk_id,
                    "domain": chunk.domain,
                    "keywords": chunk.keywords,
                    "text": chunk.text,
                }
                for chunk in selected_chunks
            ],
            "required_evidence": [
                {
                    "chunk_id": chunk.chunk_id,
                    "domain": chunk.domain,
                    "keywords": chunk.keywords,
                    "text": chunk.text,
                }
                for chunk in required_chunks
            ],
        },
        temperature=0.0,
        max_output_tokens=180,
    )
    payload = result.data
    return {
        "answer_quality": max(0.0, min(1.0, float(payload.get("answer_quality", 0.0)))),
        "groundedness": max(0.0, min(1.0, float(payload.get("groundedness", 0.0)))),
        "coverage": max(0.0, min(1.0, float(payload.get("coverage", 0.0)))),
        "citation_support": max(0.0, min(1.0, float(payload.get("citation_support", 0.0)))),
        "notes": str(payload.get("notes", "")).strip(),
    }


async def rewrite_prompt(
    *,
    prompt: str,
    mode: str,
    target_tokens: int,
    evidence_notes: list[dict[str, str]],
    citation_ids: list[str],
) -> dict[str, Any]:
    result = await call_json(
        system_prompt=(
            "You are PROMPT_COMPRESSOR for grounded prompt optimization. "
            "Rewrite the user's prompt to preserve intent while reducing length and keeping essential constraints. "
            "If evidence notes are provided, use them to keep the rewrite grounded. "
            "Return JSON with exactly these fields: "
            '{"optimized_prompt": "text", "estimated_tokens": 123, "citation_ready": true, "citation_guidance": "short note"}'
        ),
        user_payload={
            "mode": mode,
            "target_tokens": target_tokens,
            "prompt": prompt,
            "evidence_notes": evidence_notes,
            "citation_ids": citation_ids,
        },
        temperature=0.1,
        max_output_tokens=max(220, min(600, target_tokens * 8)),
    )
    payload = result.data
    return {
        "optimized_prompt": str(payload.get("optimized_prompt", "")).strip(),
        "estimated_tokens": max(1, int(payload.get("estimated_tokens", target_tokens))),
        "citation_ready": bool(payload.get("citation_ready", False)),
        "citation_guidance": str(payload.get("citation_guidance", "")).strip() or None,
    }
