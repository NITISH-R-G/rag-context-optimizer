"""
Task definitions for the rag-context-optimizer benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass(frozen=True, slots=True)
class Task:
    name: str
    description: str
    query: str
    domain_filter: Optional[str]
    token_budget: int
    required_chunk_ids: list[str]
    max_steps: int
    difficulty: Literal["easy", "medium", "hard"]
    expected_citation_ids: list[str] = field(default_factory=list)


TASK_EASY = Task(
    name="single_domain_qa",
    description="Draft a customer refund recommendation using support policy evidence from one domain",
    query="You are handling a billing escalation after a confirmed outage. What support policy steps should an agent follow before issuing a subscription refund?",
    domain_filter="customer_support_operations",
    token_budget=800,
    required_chunk_ids=["support_001", "support_003", "support_005"],
    expected_citation_ids=["support_001", "support_003"],
    max_steps=6,
    difficulty="easy",
)

TASK_MEDIUM = Task(
    name="cross_domain_synthesis",
    description="Prepare a cross-functional outage brief that aligns support triage with incident and release workflows",
    query="You are preparing a cross-functional outage brief. How should support triage procedures align with incident response and release management during a major outage?",
    domain_filter=None,
    token_budget=500,
    required_chunk_ids=["support_004", "incident_002", "reliability_003"],
    expected_citation_ids=["support_004", "incident_002", "reliability_003"],
    max_steps=8,
    difficulty="medium",
)

TASK_HARD = Task(
    name="adversarial_compression",
    description="Draft a terse active-incident brief for a compromised admin account under a very tight budget",
    query="You are responding to a suspected compromised admin account during an active incident. What actions reduce customer harm immediately, and what release-engineering safeguards are analogous?",
    domain_filter=None,
    token_budget=300,
    required_chunk_ids=["incident_003", "support_006", "reliability_004"],
    expected_citation_ids=["incident_003", "support_006", "reliability_004"],
    max_steps=10,
    difficulty="hard",
)


ALL_TASKS: list[Task] = [TASK_EASY, TASK_MEDIUM, TASK_HARD]
TASKS_BY_NAME: dict[str, Task] = {task.name: task for task in ALL_TASKS}
