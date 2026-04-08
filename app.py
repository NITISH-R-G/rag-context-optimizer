"""
FastAPI server exposing the rag-context-optimizer OpenEnv HTTP API.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from env.environment import RagContextOptimizerEnv
from env.models import RagAction
from env.corpus import list_corpus_families
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    env = RagContextOptimizerEnv()
    await env.reset()
    app.state.env = env
    yield
    await app.state.env.close()


app = FastAPI(
    title="rag-context-optimizer",
    version="1.0.0",
    description="RAG pipeline optimization environment — minimize tokens, maximize answer quality",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UI_TEMPLATE_PATH = Path(__file__).resolve().parent / "server" / "templates" / "ui.html"





@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"[request] {request.method} {request.url.path}")
    response = await call_next(request)
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


def _serialize_step_result(result: Any, reset: bool = False) -> dict[str, Any]:
    raw_info = result.info or {}
    if reset:
        return {
            "observation": _serialize_observation(result.observation),
            "reward": None,
            "done": False,
            "info": {},
        }
    return {
        "observation": _serialize_observation(result.observation),
        "reward": result.reward,
        "done": result.done,
        "info": {
            "grader_breakdown": raw_info.get("grader"),
            "event": raw_info.get("event"),
            "passed": raw_info.get("passed"),
        },
    }


def _is_bad_action_event(event: str | None) -> bool:
    return event in {
        "chunk_not_found",
    }


def _tokenize(text: str) -> set[str]:
    import re

    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _content_terms(text: str) -> set[str]:
    return {term for term in _tokenize(text) if len(term) > 2 and term not in _PROMPT_STOPWORDS}


def _clean_output_text(text: str) -> str:
    import re

    cleaned = text.replace("```", " ").replace("---", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"[#*_`]+", "", cleaned)
    cleaned = re.sub(r'\b(title|emoji|colorfrom|colorto|sdk|app_file|pinned)\s*:\s*', "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" -:")


def _compact_text(text: str, max_words: int = 28) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,;:") + " ..."


_PROMPT_STOPWORDS = {
    "a","an","and","are","as","at","be","but","by","can","could","do","does","did",
    "for","from","had","has","have","how","i","if","in","into","is","it","its","me",
    "my","of","on","or","our","should","so","than","that","the","their","them","then",
    "there","these","they","this","to","too","use","using","was","we","were","what",
    "when","where","which","while","with","without","would","you","your",
}


def _approx_tokens(text: str) -> int:
    return max(1, len(text.strip()) // 4) if text.strip() else 0


def _compress_prompt_text(prompt: str, target_tokens: int) -> str:
    import re

    raw = " ".join(prompt.strip().split())
    if not raw:
        return ""

    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_/]*", raw)
    kept: list[str] = []
    seen: set[str] = set()

    # Keep “meaningful” tokens: numbers, identifiers, longer words, and acronyms. Drop stopwords.
    for tok in tokens:
        low = tok.lower()
        is_number = low.isdigit()
        is_identifier = any(ch in tok for ch in ("_", "-", "/")) and len(tok) >= 4
        is_acronym = tok.isupper() and len(tok) <= 8
        is_meaningful = is_number or is_identifier or is_acronym or len(low) >= 4
        if not is_meaningful:
            continue
        if low in _PROMPT_STOPWORDS:
            continue
        if low in seen:
            continue
        seen.add(low)
        kept.append(tok)
        if len(kept) >= max(10, target_tokens):
            break

    if not kept:
        # Fallback: truncated raw prompt.
        words = raw.split()
        return " ".join(words[: max(8, target_tokens)]).rstrip(" ,;:") + (" ..." if len(words) > target_tokens else "")

    # Turn the token list into a copy-paste-ready “goal” sentence.
    goal = " ".join(kept)
    goal = re.sub(r"\s+", " ", goal).strip()
    return goal


_INSTRUCTION_PRIORITY_TERMS = {
    "must","should","only","not","never","always","include","exclude","cite","answer",
    "return","draft","write","summarize","compare","explain","verify","preserve","focus",
    "keep","avoid","report","escalate","rollback","refund","incident","customer","security",
}


def _trim_sentence(sentence: str, max_terms: int) -> str:
    import re

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\\-_/]*|[,:;()]", sentence)
    if not words:
        return ""
    kept: list[str] = []

    for index, token in enumerate(words):
        normalized = re.sub(r"[^A-Za-z0-9]+", "", token).lower()
        if token in {",", ":", ";", "(", ")"}:
            if kept and kept[-1] not in {",", ":", ";", "("}:
                kept.append(token)
            continue
        is_priority = normalized in _INSTRUCTION_PRIORITY_TERMS
        is_meaningful = (
            normalized.isdigit()
            or any(ch in token for ch in ("_", "-", "/"))
            or len(normalized) >= 4
            or is_priority
            or index < 3
        )
        if not is_meaningful:
            continue
        if normalized in _PROMPT_STOPWORDS and not is_priority and index >= 3:
            continue
        kept.append(token)
        if len([word for word in kept if word not in {",", ":", ";", "(", ")"}]) >= max_terms:
            break

    text = " ".join(kept)
    text = re.sub(r"\s+([,:;)])", r"\1", text)
    text = re.sub(r"(\()\s+", r"\1", text)
    return text.strip(" ,;:")


def _rewrite_prompt_text(prompt: str, target_tokens: int) -> str:
    import re

    raw = " ".join(prompt.strip().split())
    if not raw:
        return ""

    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+|\n+", raw) if segment.strip()]
    if not sentences:
        sentences = [raw]

    rewritten: list[str] = []
    used_terms = 0
    max_terms = max(8, target_tokens)
    for index, sentence in enumerate(sentences):
        remaining = max_terms - used_terms
        if remaining <= 0:
            break
        compact = _trim_sentence(sentence, max(4, remaining if index == 0 else min(remaining, 10)))
        if not compact:
            continue
        rewritten.append(compact)
        used_terms += len(compact.split())
        if used_terms >= max_terms:
            break

    if not rewritten:
        fallback = _trim_sentence(raw, max_terms)
        return fallback or raw[: max(16, target_tokens * 4)].strip()

    output = ". ".join(rewritten).strip()
    if len(rewritten) == 1 and not output.endswith("."):
        output += "."
    return output


def _fit_citations_into_prompt(base_prompt: str, citation_ids: list[str], input_tokens: int, target_tokens: int, source_prompt: str) -> tuple[str, bool, str | None]:
    if not citation_ids:
        return base_prompt, False, "No high-confidence evidence anchors were selected."

    citation_suffix = " Evidence: " + " ".join(f"[{chunk_id}]" for chunk_id in citation_ids[:3])
    with_all = (base_prompt.rstrip(".") + "." + citation_suffix).strip()
    if _approx_tokens(with_all) < input_tokens:
        return with_all, True, None

    one_citation_suffix = " Evidence: " + f"[{citation_ids[0]}]"
    with_one = (base_prompt.rstrip(".") + "." + one_citation_suffix).strip()
    if _approx_tokens(with_one) < input_tokens:
        return with_one, True, None

    tighter_target = max(8, target_tokens - 3)
    tighter_prompt = _rewrite_prompt_text(source_prompt, tighter_target)
    tighter_with_one = (tighter_prompt.rstrip(".") + "." + one_citation_suffix).strip()
    if _approx_tokens(tighter_with_one) < input_tokens:
        return tighter_with_one, True, None

    return base_prompt, False, "Citations were omitted to keep the optimized prompt shorter than the original. Use the evidence notes below if explicit anchors are required."


def _summarize_chunk_for_output(chunk: Any, effective_text: str) -> str:
    if getattr(chunk, "domain", "").startswith("Project"):
        keywords = ", ".join(chunk.keywords[:5])
        domain = chunk.domain.replace("Project ", "").lower()
        return _compact_text(f"This benchmark's {domain} covers {keywords}.", 24)
    ranked_sentences = _sentence_rank(" ".join(chunk.keywords), _clean_output_text(effective_text))
    if ranked_sentences:
        return _compact_text(_clean_output_text(ranked_sentences[0]))
    return _compact_text(_clean_output_text(effective_text))


def _sentence_rank(query: str, text: str) -> list[str]:
    import re

    query_terms = _tokenize(query)
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
    if not sentences:
        return []

    ranked: list[tuple[float, str]] = []
    for index, sentence in enumerate(sentences):
        sentence_terms = _tokenize(sentence)
        overlap = len(query_terms & sentence_terms)
        score = (overlap * 2.0) + (0.25 if index == 0 else 0.0)
        ranked.append((score, sentence))

    ranked.sort(key=lambda item: (-item[0], len(item[1])))
    return [sentence for _score, sentence in ranked]


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
    clean_prompt = prompt.strip()
    env = RagContextOptimizerEnv(
        task_name="single_domain_qa",
        query_override=clean_prompt,
        token_budget_override=800,
        max_steps_override=6,
        corpus_family_override=corpus_family,
    )
    await env.reset()

    tuning = env._last_tuning or env.context_tuner.tune(clean_prompt, env._available_chunks)

    ranked_candidates = []
    for chunk in env._available_chunks:
        tuned = tuning.tuned_scores.get(chunk.chunk_id)
        score = tuned.final_score if tuned is not None else env.retriever.hybrid_score(clean_prompt, chunk)
        if score < 0.16:
            continue
        ranked_candidates.append((chunk, score, tuned))
    ranked_candidates.sort(
        key=lambda item: (
            -(item[1] / max(item[0].tokens, 1)),
            -(item[2].citation_prior if item[2] is not None else 0.0),
            -item[1],
            item[0].chunk_id,
        )
    )

    selected_ids: list[str] = []
    token_cap = 360
    running_tokens = 0
    for chunk, score, tuned in ranked_candidates:
        if len(selected_ids) >= 4:
            break
        if score < 0.22 and selected_ids:
            break
        projected = running_tokens + chunk.tokens
        if projected > token_cap and selected_ids:
            continue
        selected_ids.append(chunk.chunk_id)
        env._selected_chunks.append(chunk.chunk_id)
        running_tokens += chunk.tokens

    if not selected_ids and ranked_candidates:
        best_chunk = ranked_candidates[0][0]
        selected_ids.append(best_chunk.chunk_id)
        env._selected_chunks.append(best_chunk.chunk_id)

    for chunk_id in list(selected_ids):
        chunk = env._chunk_map().get(chunk_id)
        if chunk is None:
            continue
        tuned = tuning.tuned_scores.get(chunk_id)
        score = tuned.final_score if tuned is not None else env.retriever.hybrid_score(clean_prompt, chunk)
        ratio = tuned.compression_ratio if tuned is not None else 0.5
        if score >= 0.75:
            ratio = max(ratio, 0.6)
        env._compression_ratios[chunk_id] = ratio

    input_tokens = _approx_tokens(clean_prompt)
    # Target: strictly shorter than input, while preserving more structure for longer prompts.
    if input_tokens <= 24:
        target_ratio = 0.85
    elif input_tokens <= 60:
        target_ratio = 0.75
    elif input_tokens <= 120:
        target_ratio = 0.68
    else:
        target_ratio = 0.62
    target_tokens = max(12, int(input_tokens * target_ratio))
    target_tokens = min(target_tokens, 80)

    compressed_goal = _rewrite_prompt_text(clean_prompt, target_tokens=target_tokens)

    # Optionally add a tiny amount of distilled context, but only if it still stays shorter overall.
    distilled_points: list[tuple[str, str]] = []
    for chunk_id in env._selected_chunks:
        chunk = env._chunk_map().get(chunk_id)
        if chunk is None:
            continue
        best = _summarize_chunk_for_output(chunk, env._effective_chunk_text(chunk_id))
        if best and all(existing_point != best for _existing_chunk_id, existing_point in distilled_points):
            distilled_points.append((chunk_id, best))
        if len(distilled_points) >= (2 if input_tokens < 80 else 3):
            break

    lines: list[str] = []
    lines.append(compressed_goal if compressed_goal else clean_prompt)
    if distilled_points and input_tokens >= 80:
        lines.append("")
        lines.append("Context:")
        lines.extend([f"- [{chunk_id}] {point}" for chunk_id, point in distilled_points])
    optimized_prompt = "\n".join(lines).strip()

    # Hard guarantee: never return an “optimized” prompt longer than the input.
    if input_tokens > 0 and _approx_tokens(optimized_prompt) >= input_tokens:
        # Enforce by character budget (tokens ~= chars/4).
        max_chars = max(12, (input_tokens - 1) * 4)
        optimized_prompt = optimized_prompt[:max_chars].rstrip(" ,;:\n\t")
        if optimized_prompt and not optimized_prompt.endswith("..."):
            optimized_prompt = optimized_prompt + " ..."
        # If still not strictly smaller (very small inputs), trim until it is.
        while input_tokens > 1 and _approx_tokens(optimized_prompt) >= input_tokens and len(optimized_prompt) > 12:
            optimized_prompt = optimized_prompt[:-6].rstrip(" ,;:\n\t") + " ..."
        if input_tokens > 1 and _approx_tokens(optimized_prompt) >= input_tokens:
            optimized_prompt = _rewrite_prompt_text(clean_prompt, target_tokens=max(5, input_tokens - 1))
            if optimized_prompt and not optimized_prompt.endswith("...") and _approx_tokens(optimized_prompt) >= input_tokens:
                optimized_prompt = optimized_prompt[: max(8, (input_tokens - 1) * 4)].strip() + " ..."

    optimized_prompt, citation_ready, citation_guidance = _fit_citations_into_prompt(
        optimized_prompt,
        tuning.suggested_citations or list(env._selected_chunks),
        input_tokens,
        target_tokens,
        clean_prompt,
    )

    original_prompt_tokens = input_tokens
    optimized_prompt_tokens = _approx_tokens(optimized_prompt)
    source_tokens = sum(env._chunk_map()[chunk_id].tokens for chunk_id in env._selected_chunks if chunk_id in env._chunk_map())
    compressed_tokens = sum(env._effective_chunk_tokens(chunk_id) for chunk_id in env._selected_chunks)
    evidence_terms = _content_terms(" ".join(env._effective_chunk_text(chunk_id) for chunk_id in env._selected_chunks))
    prompt_terms = _content_terms(optimized_prompt)
    inline_citations = set(re.findall(r"\[([a-z0-9_]+)\]", optimized_prompt.lower()))
    grounded_overlap = (len(prompt_terms & evidence_terms) / len(prompt_terms)) if prompt_terms else 0.0

    return {
        "optimized_prompt": optimized_prompt,
        "stats": {
            "selected_chunks": len(env._selected_chunks),
            "source_tokens": source_tokens,
            "compressed_context_tokens": compressed_tokens,
            "original_prompt_tokens": original_prompt_tokens,
            "optimized_prompt_tokens": optimized_prompt_tokens,
            "compression_gain": max(0, source_tokens - compressed_tokens),
        },
        "grounding": {
            "citations": tuning.suggested_citations or list(env._selected_chunks),
            "citation_ready": citation_ready and bool(inline_citations),
            "citation_guidance": citation_guidance,
            "grounded_overlap": round(grounded_overlap, 3),
            "evidence_notes": [
                {"chunk_id": chunk_id, "note": note}
                for chunk_id, note in distilled_points
            ],
        },
        "context_tuning": {
            "mode": tuning.mode,
            "top_demo_cases": tuning.top_demo_cases,
            "suggested_citations": tuning.suggested_citations,
            "token_dropout": tuning.token_dropout,
            "leave_one_out": tuning.leave_one_out,
        },
        "corpus_family": env._corpus_family,
        "selected_keywords": [
            keyword
            for chunk_id in env._selected_chunks
            for keyword in (env._chunk_map().get(chunk_id).keywords if env._chunk_map().get(chunk_id) else [])
        ][:10],
    }






def _suggest_action(env: RagContextOptimizerEnv) -> dict[str, Any]:
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
            "Grounded answer based on selected evidence: "
            + "; ".join(chosen_phrases[:3])
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
            -(score_map.get(chunk.chunk_id).final_score if score_map.get(chunk.chunk_id) else 0.0)
            / max(chunk.tokens, 1),
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


@app.post("/reset")
async def reset_endpoint(payload: ResetRequest):
    if payload.task_name not in TASKS_BY_NAME:
        raise HTTPException(status_code=400, detail="Unknown task_name.")
    env = RagContextOptimizerEnv(
        task_name=payload.task_name,
        query_override=payload.custom_query,
        token_budget_override=payload.token_budget,
        max_steps_override=payload.max_steps,
        corpus_family_override=payload.corpus_family,
    )
    app.state.env = env
    result = await env.reset()
    return _serialize_step_result(result, reset=True)


@app.post("/step")
async def step_endpoint(action: RagAction):
    env = getattr(app.state, "env", None)
    if env is None:
        raise HTTPException(status_code=400, detail="Environment is not initialized. Call /reset first.")

    result = await env.step(action)
    event = (result.info or {}).get("event")
    if _is_bad_action_event(event):
        raise HTTPException(status_code=400, detail=event)
    return _serialize_step_result(result, reset=False)


@app.get("/state")
async def state_endpoint():
    env = getattr(app.state, "env", None)
    if env is None:
        raise HTTPException(status_code=400, detail="Environment is not initialized.")
    return await env.state()


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
async def optimize_step_endpoint():
    env = getattr(app.state, "env", None)
    if env is None:
        raise HTTPException(status_code=400, detail="Environment is not initialized. Call /reset first.")
    return _suggest_action(env)


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
