"""
Task definitions for the incident operations environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass(frozen=True, slots=True)
class Task:
    name: str
    description: str
    query: str
    case_summary: str
    customer_tier: Literal["standard", "business", "enterprise"]
    incident_severity: Literal["sev3", "sev2", "sev1"]
    domain_filter: Optional[str]
    token_budget: int
    required_artifact_ids: list[str]
    expected_citation_ids: list[str]
    required_plan_keywords: list[str]
    required_report_keywords: list[str]
    report_requirements: list[str]
    max_steps: int
    difficulty: Literal["easy", "medium", "hard"]
    objective_type: Literal["refund_triage", "outage_brief", "executive_escalation"] = "refund_triage"
    optional_artifact_ids: list[str] = field(default_factory=list)


TASK_EASY = Task(
    name="refund_triage_easy",
    description="Resolve a billing escalation after an outage by gathering support evidence and drafting a refund-review memo.",
    query="Prepare an incident-linked refund triage memo. Determine whether the case should proceed toward refund review, goodwill credit, or billing queue escalation.",
    case_summary=(
        "A business customer reported a service outage during a billing period and is demanding an immediate refund. "
        "Support already confirmed broad impact but has not yet assembled the policy trail needed for a safe compensation decision."
    ),
    customer_tier="business",
    incident_severity="sev2",
    domain_filter="customer_support_operations",
    token_budget=850,
    required_artifact_ids=["support_001", "support_003", "support_005"],
    expected_citation_ids=["support_001", "support_003", "support_005"],
    required_plan_keywords=["verify outage", "billing ledger", "finance review"],
    required_report_keywords=["refund", "goodwill", "billing queue", "finance review"],
    report_requirements=[
        "State whether the case should proceed to refund review, goodwill credit, or billing queue escalation.",
        "Explain the evidence trail before customer relief is offered.",
        "Cite the policy artifacts that justify the recommendation.",
    ],
    max_steps=7,
    difficulty="easy",
    objective_type="refund_triage",
    optional_artifact_ids=["support_004"],
)

TASK_MEDIUM = Task(
    name="cross_function_brief_medium",
    description="Build a cross-functional outage brief that aligns support operations, incident command, and release engineering.",
    query="Prepare a cross-functional outage response brief that support, incident command, and release engineering can all act on safely.",
    case_summary=(
        "A payment processing outage triggered customer escalations, internal incident response, and release rollback discussions. "
        "Leadership wants one brief that explains customer handling, the incident command discipline, and the release-engineering safeguards to apply next."
    ),
    customer_tier="enterprise",
    incident_severity="sev1",
    domain_filter=None,
    token_budget=620,
    required_artifact_ids=["support_004", "incident_002", "reliability_003"],
    expected_citation_ids=["support_004", "incident_002", "reliability_003"],
    required_plan_keywords=["incident timeline", "confirmed updates", "rollback guardrails"],
    required_report_keywords=["support", "incident", "release", "customer impact", "confirmed updates"],
    report_requirements=[
        "Summarize how support should handle affected customers during the outage.",
        "Describe the incident-command communication discipline.",
        "Include the release-engineering safeguard that prevents unsafe recovery work.",
    ],
    max_steps=8,
    difficulty="medium",
    objective_type="outage_brief",
    optional_artifact_ids=["incident_006", "support_003"],
)

TASK_HARD = Task(
    name="executive_escalation_hard",
    description="Resolve a severe executive escalation involving suspected admin compromise under a tight token budget.",
    query="Prepare an executive escalation note for a suspected compromised admin account that balances immediate customer protection, incident containment, and release safeguards.",
    case_summary=(
        "An enterprise customer believes an admin account was compromised during a live incident. "
        "Executives need a terse but defensible note describing what actions reduce customer harm immediately, "
        "what evidence must be preserved, and what release guardrails are analogous before risky changes continue."
    ),
    customer_tier="enterprise",
    incident_severity="sev1",
    domain_filter=None,
    token_budget=360,
    required_artifact_ids=["incident_003", "incident_004", "support_006", "reliability_004"],
    expected_citation_ids=["incident_003", "support_006", "reliability_004"],
    required_plan_keywords=["revoke sessions", "protect customers", "change freeze"],
    required_report_keywords=["customer harm", "credential rotation", "protective outreach", "release safeguards"],
    report_requirements=[
        "State the immediate customer-protection actions.",
        "Preserve the investigation trail before cleanup.",
        "Name the release-engineering safeguard analogous to the active-incident posture.",
    ],
    max_steps=10,
    difficulty="hard",
    objective_type="executive_escalation",
    optional_artifact_ids=["incident_006", "reliability_003"],
)


ALL_TASKS: list[Task] = [TASK_EASY, TASK_MEDIUM, TASK_HARD]
TASKS_BY_NAME: dict[str, Task] = {task.name: task for task in ALL_TASKS}
