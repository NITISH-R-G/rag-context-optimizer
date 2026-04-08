"""
Typed Pydantic models for the incident operations OpenEnv environment.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ChunkSummary(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "chunk_id": "support_003",
                    "domain": "Customer Support Operations",
                    "tokens": 132,
                    "keywords": ["refund policy", "incident timeline", "billing ledger"],
                }
            ]
        }
    )

    chunk_id: str = Field(..., description="Unique artifact identifier exposed to the agent.")
    domain: str = Field(..., description="High-level source domain for the artifact.")
    tokens: int = Field(..., ge=1, description="Approximate token cost for including the artifact.")
    keywords: list[str] = Field(..., min_length=1, description="Important artifact hints available before inspection.")

    @field_validator("chunk_id", "domain")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value must not be empty.")
        return value

    @field_validator("keywords")
    @classmethod
    def validate_keywords(cls, value: list[str]) -> list[str]:
        cleaned = [keyword.strip() for keyword in value if keyword.strip()]
        if not cleaned:
            raise ValueError("keywords must contain at least one non-empty entry.")
        return cleaned


class RagObservation(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "case_id": "case-refund-triage-001",
                    "case_summary": "A business customer requests a refund after a confirmed outage.",
                    "objective": "Prepare a refund triage memo grounded in support evidence.",
                    "workflow_stage": "triage",
                    "customer_tier": "business",
                    "incident_severity": "sev2",
                    "available_artifacts": [
                        {
                            "chunk_id": "support_003",
                            "domain": "Customer Support Operations",
                            "tokens": 132,
                            "keywords": ["refund policy", "incident timeline", "billing ledger"],
                        }
                    ],
                    "reviewed_artifacts": [],
                    "prioritized_artifacts": [],
                    "plan_draft": None,
                    "report_requirements": ["State whether the case should proceed to refund review."],
                    "total_tokens_used": 0,
                    "token_budget": 850,
                    "step_number": 0,
                    "task_name": "refund_triage_easy",
                    "last_action_feedback": None,
                    "query": "Prepare an incident-linked refund triage memo.",
                    "available_chunks": [],
                    "selected_chunks": [],
                }
            ]
        }
    )

    case_id: str = Field(..., description="Unique identifier for the active simulated incident case.")
    case_summary: str = Field(..., description="Short real-world case summary presented to the agent.")
    objective: str = Field(..., description="The operational deliverable the agent must produce.")
    workflow_stage: Literal["triage", "analysis", "resolution", "submitted"] = Field(
        ..., description="Current workflow stage in the incident operations process."
    )
    customer_tier: Literal["standard", "business", "enterprise"] = Field(
        ..., description="Customer tier for the active case."
    )
    incident_severity: Literal["sev3", "sev2", "sev1"] = Field(
        ..., description="Severity of the active incident."
    )
    available_artifacts: list[ChunkSummary] = Field(
        ..., description="Artifacts that can be inspected, prioritized, or summarized."
    )
    reviewed_artifacts: list[str] = Field(
        default_factory=list,
        description="Artifact ids the agent has inspected so far.",
    )
    prioritized_artifacts: list[str] = Field(
        default_factory=list,
        description="Artifact ids currently included in the working resolution set.",
    )
    plan_draft: Optional[str] = Field(
        default=None,
        description="Current draft of the resolution plan or operational recommendation.",
    )
    report_requirements: list[str] = Field(
        default_factory=list,
        description="Deterministic requirements the final report must satisfy.",
    )
    progress_signals: dict[str, float] = Field(
        default_factory=dict,
        description="Normalized progress metrics for artifact coverage, planning, and workflow readiness.",
    )
    total_tokens_used: int = Field(..., ge=0, description="Current token cost of the prioritized working set.")
    token_budget: int = Field(..., ge=1, description="Maximum allowed token budget for the current task.")
    step_number: int = Field(..., ge=0, description="Current step number in the episode.")
    task_name: str = Field(..., description="Active task identifier.")
    last_action_feedback: Optional[str] = Field(default=None, description="Outcome of the previous action.")

    query: str = Field(..., description="Compatibility mirror of objective for legacy clients.")
    available_chunks: list[ChunkSummary] = Field(
        default_factory=list,
        description="Compatibility mirror of available_artifacts for legacy clients.",
    )
    selected_chunks: list[str] = Field(
        default_factory=list,
        description="Compatibility mirror of prioritized_artifacts for legacy clients.",
    )

    @field_validator("case_id", "case_summary", "objective", "task_name", "query")
    @classmethod
    def validate_required_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value must not be empty.")
        return value

    @field_validator("reviewed_artifacts", "prioritized_artifacts", "selected_chunks")
    @classmethod
    def validate_ids(cls, value: list[str]) -> list[str]:
        cleaned = [artifact_id.strip() for artifact_id in value if artifact_id.strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("Artifact id lists must not contain duplicates.")
        return cleaned

    @field_validator("report_requirements")
    @classmethod
    def validate_report_requirements(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        return cleaned

    @field_validator("last_action_feedback")
    @classmethod
    def validate_feedback(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def validate_budget_and_aliases(self) -> "RagObservation":
        if self.total_tokens_used > self.token_budget:
            raise ValueError("total_tokens_used cannot exceed token_budget.")
        if self.query != self.objective:
            raise ValueError("query must mirror objective.")
        if self.selected_chunks != self.prioritized_artifacts:
            raise ValueError("selected_chunks must mirror prioritized_artifacts.")
        if len(self.available_chunks) != len(self.available_artifacts):
            raise ValueError("available_chunks must mirror available_artifacts.")
        return self


class RagAction(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"action_type": "inspect_artifact", "artifact_id": "support_003"},
                {"action_type": "summarize_artifact", "artifact_id": "support_003", "compression_ratio": 0.55},
                {"action_type": "set_resolution_plan", "plan": "Verify outage evidence and route manual exceptions to finance review."},
                {"action_type": "submit_report", "answer": "Proceed to refund review only after outage and billing evidence are confirmed. [support_001] [support_003]"},
            ]
        }
    )

    action_type: Literal[
        "inspect_artifact",
        "prioritize_artifact",
        "summarize_artifact",
        "set_resolution_plan",
        "submit_report",
        "select_chunk",
        "deselect_chunk",
        "compress_chunk",
        "submit_answer",
    ] = Field(..., description="The environment action the agent wants to perform.")
    artifact_id: Optional[str] = Field(default=None, description="Target artifact id for artifact actions.")
    chunk_id: Optional[str] = Field(default=None, description="Legacy alias for artifact_id.")
    compression_ratio: Optional[float] = Field(default=None, ge=0.3, le=0.9)
    plan: Optional[str] = Field(default=None, description="Draft of the current operational resolution plan.")
    answer: Optional[str] = Field(default=None, description="Final report or resolution memo to submit.")

    @field_validator("artifact_id", "chunk_id", "plan", "answer")
    @classmethod
    def normalize_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def validate_action_semantics(self) -> "RagAction":
        normalized_artifact_id = self.artifact_id or self.chunk_id
        if self.action_type in {"inspect_artifact", "prioritize_artifact", "select_chunk", "deselect_chunk"}:
            if normalized_artifact_id is None:
                raise ValueError("artifact_id or chunk_id is required for artifact selection actions.")
        elif self.action_type in {"summarize_artifact", "compress_chunk"}:
            if normalized_artifact_id is None:
                raise ValueError("artifact_id or chunk_id is required for summarize actions.")
            if self.compression_ratio is None:
                raise ValueError("compression_ratio is required for summarize actions.")
        elif self.action_type == "set_resolution_plan":
            if self.plan is None:
                raise ValueError("plan is required for set_resolution_plan.")
        elif self.action_type in {"submit_report", "submit_answer"}:
            if self.answer is None:
                raise ValueError("answer is required for submit_report/submit_answer.")
        return self


class RagReward(BaseModel):
    total: float = Field(..., ge=0.0, le=1.0)
    token_efficiency: float = Field(..., ge=0.0, le=1.0)
    answer_quality: float = Field(..., ge=0.0, le=1.0)
    retrieval_precision: float = Field(..., ge=0.0, le=1.0)
    penalty: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_total_bound(self) -> "RagReward":
        if self.total > 1.0 or self.total < 0.0:
            raise ValueError("total must remain within [0.0, 1.0].")
        return self
