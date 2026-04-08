"""
FastAPI server exposing the rag-context-optimizer OpenEnv HTTP API.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
import os
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from env.corpus import list_corpus_families
from env.environment import RagContextOptimizerEnv
from env.llm_runtime import llm_configured
from env.llm_services import suggest_action as suggest_action_with_llm
from env.models import RagAction
from env.prompt_optimizer import CompressionMode, optimize_prompt
from env.tasks import ALL_TASKS, TASKS_BY_NAME


class ResetRequest(BaseModel):
    task_name: Literal["single_domain_qa", "cross_domain_synthesis", "adversarial_compression"] = "single_domain_qa"
    custom_query: str | None = None
    token_budget: int | None = None
    max_steps: int | None = None
    corpus_family: str | None = None


class OptimizePromptRequest(BaseModel):
    prompt: str
    corpus_family: str | None = None
    compression_mode: CompressionMode = "balanced"


class EpisodeStore:
    def __init__(self, max_episodes: int = 16):
        self._episodes: dict[str, RagContextOptimizerEnv] = {}
        self._order: list[str] = []
        self.latest_episode_id: str | None = None
        self._max_episodes = max_episodes

    async def close_all(self) -> None:
        for env in self._episodes.values():
            await env.close()
        self._episodes.clear()
        self._order.clear()
        self.latest_episode_id = None

    async def create(self, env: RagContextOptimizerEnv) -> str:
        episode_id = uuid4().hex
        self._episodes[episode_id] = env
        self._order.append(episode_id)
        self.latest_episode_id = episode_id

        while len(self._order) > self._max_episodes:
            stale_id = self._order.pop(0)
            stale_env = self._episodes.pop(stale_id, None)
            if stale_env is not None:
                await stale_env.close()
            if self.latest_episode_id == stale_id:
                self.latest_episode_id = self._order[-1] if self._order else None
        return episode_id

    def get(self, episode_id: str | None) -> tuple[str, RagContextOptimizerEnv]:
        resolved_id = episode_id or self.latest_episode_id
        if resolved_id is None or resolved_id not in self._episodes:
            raise KeyError("episode_not_found")
        return resolved_id, self._episodes[resolved_id]


def _request_logging_enabled() -> bool:
    return os.getenv("DEBUG_LOG_REQUESTS", "").strip().lower() in {"1", "true", "yes"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.episodes = EpisodeStore()
    yield
    await app.state.episodes.close_all()


app = FastAPI(
    title="rag-context-optimizer",
    version="1.0.0",
    description="RAG pipeline optimization environment - minimize tokens, maximize answer quality",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

UI_TEMPLATE_PATH = Path(__file__).resolve().parent / "server" / "templates" / "ui.html"


@app.middleware("http")
async def log_requests(request: Request, call_next):
    should_log = _request_logging_enabled()
    if should_log:
        print(f"[request] {request.method} {request.url.path}")
    response = await call_next(request)
    if should_log:
        print(f"[response] {request.method} {request.url.path} -> {response.status_code}")
    return response


@app.get("/", response_class=HTMLResponse)
async def home_page():
    return HTMLResponse(
        UI_TEMPLATE_PATH.read_text(encoding="utf-8"),
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
        },
    )


def _serialize_observation(observation: Any) -> dict[str, Any]:
    if hasattr(observation, "model_dump"):
        return observation.model_dump()
    if is_dataclass(observation):
        return asdict(observation)
    return dict(observation)


def _serialize_step_result(result: Any, reset: bool = False, episode_id: str | None = None) -> dict[str, Any]:
    raw_info = result.info or {}
    payload = {
        "observation": _serialize_observation(result.observation),
        "reward": None if reset else result.reward,
        "done": False if reset else result.done,
        "info": {} if reset else {
            "grader_breakdown": raw_info.get("grader"),
            "event": raw_info.get("event"),
            "passed": raw_info.get("passed"),
        },
    }
    if episode_id is not None:
        payload["episode_id"] = episode_id
    return payload


def _is_bad_action_event(event: str | None) -> bool:
    return event in {"chunk_not_found"}


def _episode_store() -> EpisodeStore:
    episodes = getattr(app.state, "episodes", None)
    if episodes is None:
        episodes = EpisodeStore()
        app.state.episodes = episodes
    return episodes


def _resolve_env(episode_id: str | None) -> tuple[str, RagContextOptimizerEnv]:
    try:
        return _episode_store().get(episode_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Episode not found. Call /reset first.") from exc


async def _optimize_prompt_backend(
    prompt: str,
    corpus_family: str | None = None,
    compression_mode: CompressionMode = "balanced",
) -> dict[str, Any]:
    result = await optimize_prompt(prompt, corpus_family=corpus_family, mode=compression_mode)
    return {
        "optimized_prompt": result.optimized_prompt,
        "stats": result.stats,
        "grounding": result.grounding,
        "context_tuning": result.context_tuning,
        "corpus_family": result.corpus_family,
        "selected_keywords": result.selected_keywords,
        "optimization_mode": result.optimization_mode,
    }


def _suggest_action_fallback(env: RagContextOptimizerEnv) -> dict[str, Any]:
    observation = env._build_observation()
    selected = set(observation.selected_chunks)
    remaining_budget = observation.token_budget - observation.total_tokens_used
    tuning = env._last_tuning or env.context_tuner.tune(env.task.query, env._available_chunks)
    score_map = tuning.tuned_scores
    suggested_citations = tuning.suggested_citations or list(selected)[:3]

    selected_chunks = [chunk for chunk in observation.available_chunks if chunk.chunk_id in selected]
    if selected_chunks and (
        observation.total_tokens_used >= int(observation.token_budget * 0.65)
        or observation.step_number >= 3
    ):
        heavy = sorted(
            selected_chunks,
            key=lambda chunk: (
                -(chunk.tokens * (score_map.get(chunk.chunk_id).final_score if score_map.get(chunk.chunk_id) else 0.5)),
                chunk.chunk_id,
            ),
        )
        if heavy and heavy[0].tokens > max(120, observation.token_budget // 4):
            tuned = score_map.get(heavy[0].chunk_id)
            return {
                "action_type": "compress_chunk",
                "chunk_id": heavy[0].chunk_id,
                "compression_ratio": tuned.compression_ratio if tuned is not None else 0.5,
            }

    if len(selected) >= 2 or observation.step_number >= max(2, env.task.max_steps - 2):
        chosen_phrases: list[str] = []
        for chunk in selected_chunks[:3]:
            if chunk.keywords:
                chosen_phrases.append(f"[{chunk.chunk_id}] " + ", ".join(chunk.keywords[:2]))
        answer = (
            "Grounded answer based on selected evidence: " + "; ".join(chosen_phrases[:3])
            if chosen_phrases
            else "Grounded answer based on the currently selected evidence."
        )
        if suggested_citations:
            answer = answer.rstrip(".") + " " + " ".join(f"[{chunk_id}]" for chunk_id in suggested_citations[:3]) + "."
        return {"action_type": "submit_answer", "answer": answer}

    available = [chunk for chunk in observation.available_chunks if chunk.chunk_id not in selected]
    for chunk in sorted(
        available,
        key=lambda chunk: (
            -(score_map.get(chunk.chunk_id).final_score if score_map.get(chunk.chunk_id) else 0.0) / max(chunk.tokens, 1),
            chunk.tokens,
            chunk.chunk_id,
        ),
    ):
        if chunk.tokens <= remaining_budget:
            return {"action_type": "select_chunk", "chunk_id": chunk.chunk_id}

    if selected_chunks:
        return {
            "action_type": "submit_answer",
            "answer": "Optimized answer based on the currently selected evidence.",
        }
    if available:
        smallest_chunk = min(available, key=lambda chunk: (chunk.tokens, chunk.chunk_id))
        return {
            "action_type": "submit_answer",
            "answer": (
                "No chunk fits within the current token budget. "
                f"Increase the budget to at least {smallest_chunk.tokens} tokens or choose a broader budget."
            ),
        }
    return {"action_type": "submit_answer", "answer": "No usable evidence was available."}


async def _suggest_action(env: RagContextOptimizerEnv) -> dict[str, Any]:
    if llm_configured():
        try:
            observation = env._build_observation()
            state = await env.state()
            tuning = env._last_tuning or env.context_tuner.tune(env.task.query, env._available_chunks)
            return await suggest_action_with_llm(
                observation,
                selected_chunk_details=state.get("selected_chunk_details", []),
                suggested_citations=tuning.suggested_citations,
                top_demo_cases=tuning.top_demo_cases,
            )
        except Exception:
            pass
    return _suggest_action_fallback(env)


@app.post("/reset")
async def reset_endpoint(payload: ResetRequest | None = Body(default=None)):
    payload = payload or ResetRequest()
    if payload.task_name not in TASKS_BY_NAME:
        raise HTTPException(status_code=400, detail="Unknown task_name.")

    env = RagContextOptimizerEnv(
        task_name=payload.task_name,
        query_override=payload.custom_query,
        token_budget_override=payload.token_budget,
        max_steps_override=payload.max_steps,
        corpus_family_override=payload.corpus_family,
    )
    result = await env.reset()
    episode_id = await _episode_store().create(env)
    return _serialize_step_result(result, reset=True, episode_id=episode_id)


@app.post("/step")
async def step_endpoint(action: RagAction, episode_id: str | None = None):
    resolved_episode_id, env = _resolve_env(episode_id)
    result = await env.step(action)
    event = (result.info or {}).get("event")
    if _is_bad_action_event(event):
        raise HTTPException(status_code=400, detail=event)
    return _serialize_step_result(result, reset=False, episode_id=resolved_episode_id)


@app.get("/state")
async def state_endpoint(episode_id: str | None = None):
    resolved_episode_id, env = _resolve_env(episode_id)
    state = await env.state()
    state["episode_id"] = resolved_episode_id
    return state


@app.get("/health")
async def health_endpoint():
    return {"status": "ok", "tasks": [task.name for task in ALL_TASKS]}


@app.get("/tasks")
async def tasks_endpoint():
    return [
        {
            "name": task.name,
            "description": task.description,
            "difficulty": task.difficulty,
            "token_budget": task.token_budget,
            "query": task.query,
            "max_steps": task.max_steps,
        }
        for task in ALL_TASKS
    ]


@app.get("/corpus-families")
async def corpus_families_endpoint():
    return {"families": list_corpus_families()}


@app.post("/optimize-step")
async def optimize_step_endpoint(episode_id: str | None = None):
    _resolved_episode_id, env = _resolve_env(episode_id)
    return await _suggest_action(env)


@app.post("/optimize-prompt")
async def optimize_prompt_endpoint(payload: OptimizePromptRequest):
    if not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")
    return await _optimize_prompt_backend(
        payload.prompt,
        corpus_family=payload.corpus_family,
        compression_mode=payload.compression_mode,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
