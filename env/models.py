"""
Typed Pydantic models for the rag-context-optimizer environment.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ChunkSummary(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "chunk_id": "climate_006",
                    "domain": "Climate Policy",
                    "tokens": 505,
                    "keywords": ["battery storage", "renewables", "transmission", "solar", "curtailment"],
                }
            ]
        }
    )

    chunk_id: str = Field(..., description="Unique identifier for the chunk exposed to the agent.")
    domain: str = Field(..., description="Top-level corpus domain for the chunk.")
    tokens: int = Field(..., ge=1, description="Approximate token count for the chunk before compression.")
    keywords: list[str] = Field(
        ...,
        min_length=1,
        description="Important retrieval keywords associated with the chunk.",
    )

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
                    "query": "Which policy design improved carbon tax acceptance among households?",
                    "available_chunks": [
                        {
                            "chunk_id": "climate_001",
                            "domain": "Climate Policy",
                            "tokens": 501,
                            "keywords": ["carbon tax", "rebates", "power sector", "emissions", "permit market"],
                        }
                    ],
                    "selected_chunks": ["climate_001"],
                    "total_tokens_used": 501,
                    "token_budget": 1200,
                    "step_number": 2,
                    "task_name": "easy_climate_query",
                    "last_action_feedback": "Chunk climate_001 selected successfully.",
                }
            ]
        }
    )

    query: str = Field(..., description="The question the agent must answer using retrieval and compression actions.")
    available_chunks: list[ChunkSummary] = Field(
        ...,
        description="All chunk summaries currently available for selection in the episode.",
    )
    selected_chunks: list[str] = Field(
        default_factory=list,
        description="Chunk ids the agent has selected so far for building context.",
    )
    total_tokens_used: int = Field(
        ...,
        ge=0,
        description="Current total token count consumed by the selected chunks.",
    )
    token_budget: int = Field(..., ge=1, description="Maximum token budget allowed for the current task.")
    step_number: int = Field(..., ge=0, description="Current step number in the episode.")
    task_name: str = Field(..., description="Human-readable name of the active benchmark task.")
    last_action_feedback: Optional[str] = Field(
        default=None,
        description="Short explanation of what happened after the previous action.",
    )

    @field_validator("query", "task_name")
    @classmethod
    def validate_required_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value must not be empty.")
        return value

    @field_validator("selected_chunks")
    @classmethod
    def validate_selected_chunks(cls, value: list[str]) -> list[str]:
        cleaned = [chunk_id.strip() for chunk_id in value if chunk_id.strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("selected_chunks must not contain duplicates.")
        return cleaned

    @field_validator("last_action_feedback")
    @classmethod
    def validate_feedback(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def validate_token_budget_relation(self) -> "RagObservation":
        if self.total_tokens_used > self.token_budget:
            raise ValueError("total_tokens_used cannot exceed token_budget in the observation state.")
        return self


class RagAction(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "action_type": "select_chunk",
                    "chunk_id": "climate_001",
                    "compression_ratio": None,
                    "answer": None,
                },
                {
                    "action_type": "compress_chunk",
                    "chunk_id": "climate_001",
                    "compression_ratio": 0.5,
                    "answer": None,
                },
                {
                    "action_type": "submit_answer",
                    "chunk_id": None,
                    "compression_ratio": None,
                    "answer": "Quarterly dividend checks improved household acceptance of the carbon tax.",
                },
            ]
        }
    )

    action_type: Literal["select_chunk", "deselect_chunk", "compress_chunk", "submit_answer"] = Field(
        ...,
        description="The environment action the agent wants to perform.",
    )
    chunk_id: Optional[str] = Field(
        default=None,
        description="Target chunk id for select, deselect, or compress actions.",
    )
    compression_ratio: Optional[float] = Field(
        default=None,
        ge=0.3,
        le=0.9,
        description="Compression ratio to apply during compress_chunk actions.",
    )
    answer: Optional[str] = Field(
        default=None,
        description="Final answer text supplied when the agent submits an answer.",
    )

    @field_validator("chunk_id", "answer")
    @classmethod
    def normalize_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def validate_action_semantics(self) -> "RagAction":
        if self.action_type in {"select_chunk", "deselect_chunk"}:
            if self.chunk_id is None:
                raise ValueError("chunk_id is required for select_chunk and deselect_chunk actions.")
            if self.compression_ratio is not None or self.answer is not None:
                raise ValueError("select_chunk and deselect_chunk actions only accept chunk_id.")
        elif self.action_type == "compress_chunk":
            if self.chunk_id is None:
                raise ValueError("chunk_id is required for compress_chunk.")
            if self.compression_ratio is None:
                raise ValueError("compression_ratio is required for compress_chunk.")
            if self.answer is not None:
                raise ValueError("compress_chunk does not accept answer.")
        elif self.action_type == "submit_answer":
            if self.answer is None:
                raise ValueError("answer is required for submit_answer.")
            if self.chunk_id is not None or self.compression_ratio is not None:
                raise ValueError("submit_answer only accepts answer.")
        return self


class RagReward(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "total": 0.82,
                    "token_efficiency": 0.9,
                    "answer_quality": 0.84,
                    "retrieval_precision": 0.78,
                    "penalty": 0.05,
                }
            ]
        }
    )

    total: float = Field(..., ge=0.0, le=1.0, description="Overall normalized reward for the step or episode.")
    token_efficiency: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Reward component capturing how efficiently the agent used tokens.",
    )
    answer_quality: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Reward component measuring answer correctness and completeness.",
    )
    retrieval_precision: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Reward component measuring how relevant the selected chunks were.",
    )
    penalty: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized deduction applied for wasteful or invalid behavior.",
    )

    @model_validator(mode="after")
    def validate_total_bound(self) -> "RagReward":
        recomposed = self.token_efficiency + self.answer_quality + self.retrieval_precision - self.penalty
        if recomposed < -1e-6:
            raise ValueError("Reward components minus penalty must not be negative.")
        if self.total > 1.0 or self.total < 0.0:
            raise ValueError("total must remain within [0.0, 1.0].")
        return self
