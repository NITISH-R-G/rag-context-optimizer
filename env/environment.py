"""
Main OpenEnv-style environment for incident operations and escalation handling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
import os
from pathlib import Path
import re
from typing import Any

from env.corpus import Chunk, load_corpus, resolve_corpus_path
from env.context_tuner import ContextTunedPlanner
from env.graders import TaskGrader
from env.llm_runtime import estimate_tokens
from env.models import ChunkSummary, RagAction, RagObservation
from env.retriever import HybridRetriever
from env.tasks import ALL_TASKS, TASKS_BY_NAME, Task


@dataclass(slots=True)
class StepResult:
    observation: RagObservation
    reward: float
    done: bool
    info: dict[str, Any]


class RagContextOptimizerEnv:
    _PROJECT_STOPWORDS = {
        "the", "and", "for", "with", "that", "this", "from", "into", "your", "have", "will",
        "using", "used", "use", "into", "they", "them", "their", "about", "while", "where",
        "when", "what", "which", "should", "would", "could", "there", "here", "then", "than",
        "each", "such", "only", "also", "been", "being", "does", "did", "done", "just", "more",
        "most", "very", "over", "under", "like", "same", "across", "because", "through", "make",
        "made", "many", "much", "some", "into", "onto", "must", "need", "needs", "task", "tasks",
        "chunk", "chunks", "query", "prompt", "environment", "agent", "agents", "model", "models",
    }
    _PROJECT_QUERY_HINTS = {
        "openenv", "benchmark", "rag-context-optimizer", "readme", "docker", "fastapi", "api",
        "endpoint", "inference.py", "app.py", "tasks.py", "graders.py", "environment.py", "repo",
        "repository", "codebase", "ui", "frontend", "backend", "space", "validator",
    }

    def __init__(
        self,
        task_name: str = "refund_triage_easy",
        query_override: str | None = None,
        token_budget_override: int | None = None,
        max_steps_override: int | None = None,
        corpus_family_override: str | None = None,
    ):
        if task_name not in TASKS_BY_NAME:
            raise ValueError(f"Unknown task_name: {task_name}")

        self._corpus_family = corpus_family_override or os.getenv("RAG_CORPUS_FAMILY") or "enterprise_v1"
        explicit_path = os.getenv("RAG_CORPUS_PATH")
        self._corpus_path = resolve_corpus_path(explicit_path, family=None if explicit_path else self._corpus_family)
        self._all_chunks = load_corpus(self._corpus_path)
        self._query_overridden = bool(query_override and query_override.strip())
        self._include_project_chunks = os.getenv("ENABLE_PROJECT_CORPUS", "").strip().lower() in {"1", "true", "yes"}
        self._project_chunks = self._load_project_chunks() if self._include_project_chunks else []
        self.retriever = HybridRetriever(self._all_chunks + self._project_chunks)
        self.context_tuner = ContextTunedPlanner(
            self.retriever,
            self._all_chunks + self._project_chunks,
            list(ALL_TASKS),
        )
        self.grader = TaskGrader()
        self.task: Task = self._build_task(
            TASKS_BY_NAME[task_name],
            query_override=query_override,
            token_budget_override=token_budget_override,
            max_steps_override=max_steps_override,
        )

        self._available_chunks: list[Chunk] = []
        self._reviewed_artifacts: list[str] = []
        self._selected_chunks: list[str] = []
        self._compression_ratios: dict[str, float] = {}
        self._step_number = 0
        self._done = False
        self._last_action_feedback: str | None = None
        self._last_answer = ""
        self._plan_draft = ""
        self._workflow_stage: str = "triage"
        self._case_id = f"{self.task.name}-001"
        self._last_tuning = None

    @staticmethod
    def _build_task(
        base_task: Task,
        query_override: str | None = None,
        token_budget_override: int | None = None,
        max_steps_override: int | None = None,
    ) -> Task:
        updated_task = base_task
        if query_override and query_override.strip():
            updated_task = replace(updated_task, query=query_override.strip(), domain_filter=None)
        if token_budget_override is not None and token_budget_override > 0:
            updated_task = replace(updated_task, token_budget=token_budget_override)
        if max_steps_override is not None and max_steps_override > 0:
            updated_task = replace(updated_task, max_steps=max_steps_override)
        return updated_task

    async def reset(self) -> StepResult:
        candidate_chunks = self._filter_chunks_for_task(self.task)
        self._available_chunks = self._rank_chunks_for_query(self.task.query, candidate_chunks)
        if not self._query_overridden:
            chunk_by_id = {chunk.chunk_id: chunk for chunk in candidate_chunks}
            for chunk_id in self.task.required_artifact_ids + self.task.optional_artifact_ids:
                chunk = chunk_by_id.get(chunk_id)
                if chunk and all(existing.chunk_id != chunk_id for existing in self._available_chunks):
                    self._available_chunks.append(chunk)

        self._reviewed_artifacts = []
        self._selected_chunks = []
        self._compression_ratios = {}
        self._step_number = 0
        self._done = False
        self._last_action_feedback = None
        self._last_answer = ""
        self._plan_draft = ""
        self._workflow_stage = "triage"
        self._case_id = f"{self.task.name}-001"

        observation = self._build_observation()
        return StepResult(
            observation=observation,
            reward=0.0,
            done=False,
            info={"task": self.task.name, "event": "reset"},
        )

    async def step(self, action: RagAction) -> StepResult:
        if self._done:
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=True,
                info={"task": self.task.name, "event": "episode_already_done"},
            )

        reward = 0.0
        info: dict[str, Any] = {"task": self.task.name, "action_type": action.action_type}
        artifact_id = action.artifact_id or action.chunk_id or ""

        if action.action_type == "inspect_artifact":
            reward, info = self._handle_inspect(artifact_id, auto_prioritize=False)
        elif action.action_type == "select_chunk":
            reward, info = self._handle_inspect(artifact_id, auto_prioritize=True)
        elif action.action_type == "prioritize_artifact":
            reward, info = self._handle_prioritize(artifact_id)
        elif action.action_type == "deselect_chunk":
            reward, info = self._handle_deprioritize(artifact_id)
        elif action.action_type in {"summarize_artifact", "compress_chunk"}:
            reward, info = self._handle_compress(artifact_id, float(action.compression_ratio or 0.0))
        elif action.action_type == "set_resolution_plan":
            reward, info = self._handle_plan(action.plan or "")
        elif action.action_type in {"submit_report", "submit_answer"}:
            self._last_answer = action.answer or ""
            result = await self._finalize_submission(reason="submit_report")
            self._step_number += 1
            result.observation.step_number = self._step_number
            return result

        self._step_number += 1
        self._update_workflow_stage()

        if self._step_number >= self.task.max_steps:
            return await self._finalize_submission(reason="max_steps_reached")

        observation = self._build_observation()
        return StepResult(observation=observation, reward=reward, done=False, info=info)

    async def state(self) -> dict:
        prioritized_artifact_details = []
        for chunk_id in self._selected_chunks:
            chunk = self._chunk_map().get(chunk_id)
            if chunk is None:
                continue
            prioritized_artifact_details.append(
                {
                    "artifact_id": chunk.chunk_id,
                    "chunk_id": chunk.chunk_id,
                    "domain": chunk.domain,
                    "original_tokens": chunk.tokens,
                    "effective_tokens": self._effective_chunk_tokens(chunk_id),
                    "compression_ratio": round(self._compression_ratios.get(chunk_id, 1.0), 3),
                    "text": self._effective_chunk_text(chunk_id),
                    "keywords": chunk.keywords,
                }
            )
        optimized_prompt = self._build_optimized_prompt()
        optimized_prompt_tokens = await estimate_tokens(optimized_prompt) if optimized_prompt else 0
        return {
            "task": asdict(self.task) if is_dataclass(self.task) else self.task,
            "case_id": self._case_id,
            "case_summary": self.task.case_summary,
            "objective": self.task.query,
            "workflow_stage": self._workflow_stage,
            "customer_tier": self.task.customer_tier,
            "incident_severity": self.task.incident_severity,
            "step_number": self._step_number,
            "done": self._done,
            "reviewed_artifacts": list(self._reviewed_artifacts),
            "prioritized_artifacts": list(self._selected_chunks),
            "selected_chunks": list(self._selected_chunks),
            "compression_ratios": dict(self._compression_ratios),
            "plan_draft": self._plan_draft,
            "report_requirements": list(self.task.report_requirements),
            "progress_signals": self._progress_signals(),
            "total_tokens_used": self._total_tokens_used(),
            "token_budget": self.task.token_budget,
            "last_action_feedback": self._last_action_feedback,
            "last_answer": self._last_answer,
            "corpus_family": self._corpus_family,
            "corpus_path": str(self._corpus_path),
            "available_artifact_ids": [chunk.chunk_id for chunk in self._available_chunks],
            "available_chunk_ids": [chunk.chunk_id for chunk in self._available_chunks],
            "prioritized_artifact_details": prioritized_artifact_details,
            "selected_chunk_details": prioritized_artifact_details,
            "optimized_prompt_preview": optimized_prompt,
            "optimized_prompt_tokens": optimized_prompt_tokens,
            "context_tuning": (
                {
                    "mode": self._last_tuning.mode,
                    "top_demo_cases": self._last_tuning.top_demo_cases,
                    "suggested_citations": self._last_tuning.suggested_citations,
                    "token_dropout": self._last_tuning.token_dropout,
                    "leave_one_out": self._last_tuning.leave_one_out,
                }
                if self._last_tuning is not None
                else None
            ),
        }

    async def close(self):
        self._done = True

    def _filter_chunks_for_task(self, task: Task) -> list[Chunk]:
        domain_mapping = {
            "customer_support_operations": "Customer Support Operations",
            "incident_response_playbooks": "Incident Response Playbooks",
            "platform_reliability_release_engineering": "Platform Reliability & Release Engineering",
        }
        if self._query_overridden:
            if self._include_project_chunks and self._is_project_query(task.query):
                return list(self._all_chunks) + list(self._project_chunks)
            return list(self._all_chunks)
        if task.domain_filter is None:
            return list(self._all_chunks)
        normalized = domain_mapping.get(task.domain_filter, task.domain_filter)
        return [chunk for chunk in self._all_chunks if chunk.domain == normalized]

    def _is_project_query(self, query: str) -> bool:
        lowered = query.lower()
        return any(hint in lowered for hint in self._PROJECT_QUERY_HINTS)

    def _rank_chunks_for_query(self, query: str, chunks: list[Chunk], top_k: int = 20) -> list[Chunk]:
        tuning = self.context_tuner.tune(query, chunks)
        self._last_tuning = tuning
        scored = []
        for chunk in chunks:
            tuned = tuning.tuned_scores.get(chunk.chunk_id)
            score = tuned.final_score if tuned is not None else self.retriever.hybrid_score(query, chunk)
            if self._include_project_chunks and self._query_overridden and chunk.domain.startswith("Project"):
                score = min(1.0, score + 0.08)
            scored.append((chunk, score))
        scored.sort(key=lambda item: (-item[1], item[0].tokens, item[0].chunk_id))
        return [chunk for chunk, _score in scored[: max(1, min(top_k, len(scored)))]]

    def _load_project_chunks(self) -> list[Chunk]:
        root = Path(__file__).resolve().parent.parent
        chunks: list[Chunk] = []
        file_specs = [
            ("Project Documentation", root / "README.md", ["project_docs", "readme"]),
            ("Project Configuration", root / "openenv.yaml", ["project_docs", "config", "openenv_spec"]),
            ("Project API", root / "app.py", ["project_docs", "api", "server"]),
            ("Project Baseline", root / "inference.py", ["project_docs", "baseline", "inference"]),
            ("Project Environment", root / "env" / "environment.py", ["project_docs", "environment", "state_management"]),
            ("Project Retrieval", root / "env" / "retriever.py", ["project_docs", "retrieval", "ranking"]),
            ("Project Grading", root / "env" / "graders.py", ["project_docs", "grading", "reward_design"]),
            ("Project Tasks", root / "env" / "tasks.py", ["project_docs", "tasks", "difficulty"]),
            ("Project Validation", root / "validate.py", ["project_docs", "validation", "testing"]),
        ]
        for domain, path, tags in file_specs:
            if not path.exists():
                continue
            raw_text = path.read_text(encoding="utf-8", errors="ignore")
            sections = self._chunk_project_text(raw_text)
            stem = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_") or "file"
            for index, section in enumerate(sections, start=1):
                keywords = self._extract_project_keywords(section) or [stem, domain.lower()]
                chunks.append(
                    Chunk(
                        chunk_id=f"project_{stem}_{index:03d}",
                        domain=domain,
                        text=section,
                        tokens=max(30, len(section) // 4),
                        keywords=keywords[:5],
                        relevance_tags=tags,
                    )
                )
        return chunks

    def _chunk_project_text(self, raw_text: str, chunk_words: int = 140, stride_words: int = 100) -> list[str]:
        cleaned = " ".join(raw_text.split())
        words = cleaned.split()
        if not words:
            return []
        if len(words) <= chunk_words:
            return [" ".join(words)]
        chunks: list[str] = []
        start = 0
        while start < len(words):
            window = words[start : start + chunk_words]
            if not window:
                break
            chunks.append(" ".join(window))
            if start + chunk_words >= len(words):
                break
            start += stride_words
        return chunks

    def _extract_project_keywords(self, text: str) -> list[str]:
        terms = re.findall(r"[a-z0-9_]+", text.lower())
        counts: dict[str, int] = {}
        for term in terms:
            if len(term) < 4 or term in self._PROJECT_STOPWORDS:
                continue
            counts[term] = counts.get(term, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [term.replace("_", " ") for term, _count in ranked[:8]]

    def _build_observation(self) -> RagObservation:
        available = [
            ChunkSummary(
                chunk_id=chunk.chunk_id,
                domain=chunk.domain,
                tokens=self._effective_chunk_tokens(chunk.chunk_id),
                keywords=chunk.keywords,
            )
            for chunk in self._available_chunks
        ]
        return RagObservation(
            case_id=self._case_id,
            case_summary=self.task.case_summary,
            objective=self.task.query,
            workflow_stage=self._workflow_stage,
            customer_tier=self.task.customer_tier,
            incident_severity=self.task.incident_severity,
            available_artifacts=available,
            reviewed_artifacts=list(self._reviewed_artifacts),
            prioritized_artifacts=list(self._selected_chunks),
            plan_draft=self._plan_draft or None,
            report_requirements=list(self.task.report_requirements),
            progress_signals=self._progress_signals(),
            total_tokens_used=self._total_tokens_used(),
            token_budget=self.task.token_budget,
            step_number=self._step_number,
            task_name=self.task.name,
            last_action_feedback=self._last_action_feedback,
            query=self.task.query,
            available_chunks=available,
            selected_chunks=list(self._selected_chunks),
        )

    def _chunk_map(self) -> dict[str, Chunk]:
        return {chunk.chunk_id: chunk for chunk in self._available_chunks}

    def _effective_chunk_tokens(self, chunk_id: str) -> int:
        chunk = self._chunk_map().get(chunk_id)
        if chunk is None:
            return 0
        ratio = self._compression_ratios.get(chunk_id, 1.0)
        return max(1, int(round(chunk.tokens * ratio)))

    def _total_tokens_used(self) -> int:
        return sum(self._effective_chunk_tokens(chunk_id) for chunk_id in self._selected_chunks)

    def _effective_chunk_text(self, chunk_id: str) -> str:
        chunk = self._chunk_map().get(chunk_id)
        if chunk is None:
            return ""
        ratio = self._compression_ratios.get(chunk_id, 1.0)
        text = " ".join(chunk.text.split())
        if ratio >= 0.999:
            return text

        query_terms = self._query_terms(self.task.query)
        keyword_terms = self._query_terms(" ".join(chunk.keywords))
        sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
        if not sentences:
            return self._truncate_words(text, ratio)

        ranked_sentences: list[tuple[int, float, int, str]] = []
        for index, sentence in enumerate(sentences):
            sentence_terms = self._query_terms(sentence)
            overlap = len(sentence_terms & query_terms)
            keyword_overlap = len(sentence_terms & keyword_terms)
            score = (overlap * 2.0) + keyword_overlap + (0.25 if index == 0 else 0.0)
            ranked_sentences.append((index, score, len(sentence.split()), sentence))

        target_words = max(18, int(len(text.split()) * ratio))
        chosen: list[tuple[int, str]] = []
        used_words = 0
        for index, _score, word_count, sentence in sorted(ranked_sentences, key=lambda item: (-item[1], item[2], item[0])):
            if used_words >= target_words:
                break
            chosen.append((index, sentence))
            used_words += word_count
        if not chosen:
            return self._truncate_words(text, ratio)
        chosen.sort(key=lambda item: item[0])
        compressed = " ".join(sentence for _index, sentence in chosen)
        return self._truncate_words(compressed, ratio)

    @staticmethod
    def _truncate_words(text: str, ratio: float) -> str:
        words = text.split()
        if not words:
            return ""
        keep = max(10, int(len(words) * ratio))
        truncated = " ".join(words[:keep])
        if keep < len(words):
            return truncated + " ..."
        return truncated

    @staticmethod
    def _query_terms(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}

    def _build_optimized_prompt(self) -> str:
        sections = [
            f"Case: {self.task.case_summary}",
            f"Objective: {self.task.query}",
            f"Stage: {self._workflow_stage}",
        ]
        if self._plan_draft:
            sections.extend(["", f"Plan Draft: {self._plan_draft}"])
        if self._selected_chunks:
            sections.extend(["", "Prioritized Evidence:"])
            for chunk_id in self._selected_chunks:
                chunk = self._chunk_map().get(chunk_id)
                if chunk is None:
                    continue
                sections.append(f"[{chunk.chunk_id} | {self._effective_chunk_tokens(chunk_id)} tokens] {self._effective_chunk_text(chunk_id)}")
        return "\n".join(sections).strip()

    def _is_relevant(self, chunk_id: str) -> tuple[bool, float]:
        chunk = self._chunk_map().get(chunk_id)
        if chunk is None:
            return False, 0.0
        score = self.retriever.hybrid_score(self.task.query, chunk)
        return score >= 0.3, score

    def _is_required(self, chunk_id: str) -> bool:
        return chunk_id in set(self.task.required_artifact_ids)

    def _progress_signals(self) -> dict[str, float]:
        required = set(self.task.required_artifact_ids)
        reviewed_hits = len(required & set(self._reviewed_artifacts)) / len(required) if required else 1.0
        prioritized_hits = len(required & set(self._selected_chunks)) / len(required) if required else 1.0
        plan_keywords = sum(1 for keyword in self.task.required_plan_keywords if keyword.lower() in self._plan_draft.lower())
        plan_quality = plan_keywords / len(self.task.required_plan_keywords) if self.task.required_plan_keywords else 1.0
        return {
            "review_coverage": round(reviewed_hits, 3),
            "priority_coverage": round(prioritized_hits, 3),
            "plan_quality": round(plan_quality, 3),
            "budget_headroom": round(max(0.0, 1.0 - (self._total_tokens_used() / self.task.token_budget)), 3),
        }

    def _update_workflow_stage(self) -> None:
        if self._done:
            self._workflow_stage = "submitted"
        elif self._plan_draft.strip():
            self._workflow_stage = "resolution"
        elif self._reviewed_artifacts:
            self._workflow_stage = "analysis"
        else:
            self._workflow_stage = "triage"

    def _handle_inspect(self, chunk_id: str, auto_prioritize: bool) -> tuple[float, dict[str, Any]]:
        chunk = self._chunk_map().get(chunk_id)
        if chunk is None:
            self._last_action_feedback = "artifact_not_found"
            return -0.1, {"event": "artifact_not_found", "artifact_id": chunk_id}
        if chunk_id not in self._reviewed_artifacts:
            self._reviewed_artifacts.append(chunk_id)
        is_relevant, score = self._is_relevant(chunk_id)
        reward = 0.03 + (0.08 if self._is_required(chunk_id) else 0.0) + (0.05 if is_relevant else 0.0)
        info = {"event": "artifact_inspected", "artifact_id": chunk_id, "hybrid_score": score}
        self._last_action_feedback = "artifact_inspected"
        if auto_prioritize:
            priority_reward, priority_info = self._handle_prioritize(chunk_id, inspected=True)
            reward += priority_reward
            info["auto_prioritize"] = priority_info
        return min(reward, 0.2), info

    def _handle_prioritize(self, chunk_id: str, inspected: bool = False) -> tuple[float, dict[str, Any]]:
        chunk = self._chunk_map().get(chunk_id)
        if chunk is None:
            self._last_action_feedback = "artifact_not_found"
            return -0.1, {"event": "artifact_not_found", "artifact_id": chunk_id}
        if chunk_id not in self._reviewed_artifacts and not inspected:
            self._last_action_feedback = "artifact_not_reviewed"
            return -0.05, {"event": "artifact_not_reviewed", "artifact_id": chunk_id}
        if chunk_id in self._selected_chunks:
            self._last_action_feedback = "artifact_already_prioritized"
            return 0.0, {"event": "artifact_already_prioritized", "artifact_id": chunk_id}
        projected_tokens = self._total_tokens_used() + self._effective_chunk_tokens(chunk_id)
        if projected_tokens > self.task.token_budget:
            self._last_action_feedback = "exceeded_budget"
            return -0.1, {"event": "exceeded_budget", "artifact_id": chunk_id}
        self._selected_chunks.append(chunk_id)
        is_relevant, score = self._is_relevant(chunk_id)
        domain_bonus = 0.04 if len({self._chunk_map()[cid].domain for cid in self._selected_chunks if cid in self._chunk_map()}) > 1 else 0.0
        reward = (0.10 if self._is_required(chunk_id) else 0.03) + (0.05 if is_relevant else 0.0) + domain_bonus
        self._last_action_feedback = "artifact_prioritized"
        return min(reward, 0.18), {"event": "artifact_prioritized", "artifact_id": chunk_id, "hybrid_score": score}

    def _handle_deprioritize(self, chunk_id: str) -> tuple[float, dict[str, Any]]:
        if chunk_id not in self._selected_chunks:
            self._last_action_feedback = "artifact_not_prioritized"
            return 0.0, {"event": "artifact_not_prioritized", "artifact_id": chunk_id}
        self._selected_chunks.remove(chunk_id)
        is_required = self._is_required(chunk_id)
        reward = -0.06 if is_required else 0.03
        self._last_action_feedback = "artifact_deprioritized"
        return reward, {"event": "artifact_deprioritized", "artifact_id": chunk_id, "required": is_required}

    def _handle_compress(self, chunk_id: str, compression_ratio: float) -> tuple[float, dict[str, Any]]:
        chunk = self._chunk_map().get(chunk_id)
        if chunk is None:
            self._last_action_feedback = "artifact_not_found"
            return -0.1, {"event": "artifact_not_found", "artifact_id": chunk_id}
        if chunk_id not in self._selected_chunks:
            self._last_action_feedback = "artifact_not_prioritized"
            return -0.04, {"event": "artifact_not_prioritized", "artifact_id": chunk_id}
        self._compression_ratios[chunk_id] = compression_ratio
        is_relevant, score = self._is_relevant(chunk_id)
        reward = 0.04 if is_relevant else 0.0
        if self._is_required(chunk_id) and compression_ratio < 0.45:
            reward -= 0.06
            self._last_action_feedback = "overcompressed_required_artifact"
            return reward, {"event": "overcompressed_required_artifact", "artifact_id": chunk_id, "hybrid_score": score}
        self._last_action_feedback = "artifact_summarized"
        return reward, {"event": "artifact_summarized", "artifact_id": chunk_id, "hybrid_score": score}

    def _handle_plan(self, plan: str) -> tuple[float, dict[str, Any]]:
        self._plan_draft = plan.strip()
        if not self._plan_draft:
            self._last_action_feedback = "empty_plan"
            return -0.05, {"event": "empty_plan"}
        hits = sum(1 for keyword in self.task.required_plan_keywords if keyword.lower() in self._plan_draft.lower())
        coverage = hits / len(self.task.required_plan_keywords) if self.task.required_plan_keywords else 1.0
        reviewed_bonus = min(0.1, 0.02 * len(self._reviewed_artifacts))
        reward = (0.04 + (0.18 * coverage) + reviewed_bonus)
        self._last_action_feedback = "plan_updated"
        return min(reward, 0.26), {"event": "plan_updated", "plan_quality": coverage}

    async def _finalize_submission(self, reason: str) -> StepResult:
        self._done = True
        self._update_workflow_stage()

        if not self._selected_chunks:
            self._last_action_feedback = "no_prioritized_artifacts"
            observation = self._build_observation()
            return StepResult(
                observation=observation,
                reward=0.0,
                done=True,
                info={"event": reason, "grader": None, "passed": False},
            )

        grader_result = self.grader.grade(
            prioritized_artifact_ids=list(self._selected_chunks),
            reviewed_artifact_ids=list(self._reviewed_artifacts),
            answer=self._last_answer,
            plan_draft=self._plan_draft,
            workflow_stage=self._workflow_stage,
            token_budget=self.task.token_budget,
            total_tokens_used=self._total_tokens_used(),
            retriever=self.retriever,
            task=self.task,
        )
        self._last_action_feedback = reason
        observation = self._build_observation()
        return StepResult(
            observation=observation,
            reward=grader_result.score,
            done=True,
            info={"event": reason, "grader": grader_result.breakdown, "passed": grader_result.passed},
        )
