---
title: rag-context-optimizer
emoji: "📚"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - incident-ops
  - benchmark
  - enterprise
---

# rag-context-optimizer

`rag-context-optimizer` is now a real-world OpenEnv environment for enterprise incident operations.

Instead of only picking context chunks, the agent must work through a realistic operational loop:
- inspect incident and support artifacts
- prioritize the evidence that belongs in the working set
- summarize heavy artifacts when token pressure rises
- draft a resolution plan
- submit a grounded final memo or escalation note

This models work that support leads, incident commanders, and release managers actually do during outages and security escalations.

## Why This Is A Real Environment

The environment simulates operational decisions humans make during live enterprise incidents:
- refund triage after a confirmed outage
- cross-functional outage briefing across support, incident response, and release engineering
- executive escalation handling during a suspected admin compromise

That makes it useful for:
- evaluating operational agent behavior
- training evidence prioritization policies
- benchmarking grounded reporting under token pressure
- comparing safe workflow planning against premature free-form answering

## OpenEnv API

Standard endpoints:
- `POST /reset`
- `POST /step`
- `GET /state`

Additional helper endpoints:
- `GET /health`
- `GET /tasks`
- `POST /optimize-step`
- `POST /optimize-prompt`

Metadata lives in [openenv.yaml](/C:/Users/nitis/Downloads/Meta%20OpenEnv/openenv.yaml).

## Observation Space

`RagObservation` includes:

| Field | Type | Description |
| --- | --- | --- |
| `case_id` | `str` | Unique simulated case identifier |
| `case_summary` | `str` | Real-world case context |
| `objective` | `str` | Deliverable the agent must produce |
| `workflow_stage` | `triage \| analysis \| resolution \| submitted` | Current stage |
| `customer_tier` | `standard \| business \| enterprise` | Customer criticality |
| `incident_severity` | `sev3 \| sev2 \| sev1` | Incident severity |
| `available_artifacts` | `List[ChunkSummary]` | Artifacts available for inspection or prioritization |
| `reviewed_artifacts` | `List[str]` | Artifacts the agent has inspected |
| `prioritized_artifacts` | `List[str]` | Artifacts in the working set |
| `plan_draft` | `Optional[str]` | Current operational plan |
| `report_requirements` | `List[str]` | Final memo requirements |
| `progress_signals` | `Dict[str, float]` | Partial progress metrics |
| `total_tokens_used` | `int` | Current working-set token cost |
| `token_budget` | `int` | Allowed token budget |

Compatibility mirrors are also present for legacy clients:
- `query`
- `available_chunks`
- `selected_chunks`

## Action Space

Canonical actions:

| Action Type | Parameters | Effect |
| --- | --- | --- |
| `inspect_artifact` | `artifact_id` | Review an artifact without yet committing it to the working set |
| `prioritize_artifact` | `artifact_id` | Add a reviewed artifact to the working set |
| `summarize_artifact` | `artifact_id`, `compression_ratio` | Compress a prioritized artifact to reduce token cost |
| `set_resolution_plan` | `plan` | Draft the operational plan before submission |
| `submit_report` | `answer` | Submit the final grounded memo and end the episode |

Legacy aliases are still accepted for compatibility:
- `select_chunk`
- `deselect_chunk`
- `compress_chunk`
- `submit_answer`

## Tasks

| Task | Difficulty | Max Steps | Token Budget | Description |
| --- | --- | --- | --- | --- |
| `refund_triage_easy` | easy | `7` | `850` | Build a refund-review memo from support policy evidence after an outage |
| `cross_function_brief_medium` | medium | `8` | `620` | Prepare a cross-functional outage brief spanning support, incident command, and release controls |
| `executive_escalation_hard` | hard | `10` | `360` | Draft a terse executive escalation note for a suspected admin compromise |

Task definitions live in [env/tasks.py](/C:/Users/nitis/Downloads/Meta%20OpenEnv/env/tasks.py).

## Reward Design

The environment provides shaped signal across the trajectory:
- positive reward for inspecting required evidence
- positive reward for prioritizing the right artifacts
- positive reward for multi-domain coverage on cross-functional tasks
- positive reward for high-quality operational plans
- positive reward for safe token compression
- penalty for over-compressing critical evidence
- penalty for deprioritizing required artifacts
- final deterministic score in `[0, 1]` based on:
  - artifact coverage
  - review coverage
  - domain coverage
  - plan quality
  - report quality
  - citation accuracy
  - token efficiency
  - workflow readiness
  - unsupported claim penalty

The grader is deterministic and task-specific.

## LLM-backed Helpers

The environment includes optional LLM-backed helpers:
- `/optimize-step` proposes the next workflow action
- `/optimize-prompt` rewrites prompts under budget while preserving grounding

The authoritative grader remains deterministic for reproducibility.

## Local Setup

### Requirements

- Python 3.11+ recommended
- Docker
- `openenv-core`

### Install

```bash
pip install -r requirements.txt
pip install openenv-core
```

### Run

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Validate

```bash
docker build .
openenv validate
python validate.py
```

## API Examples

### Reset

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name":"refund_triage_easy"}'
```

### Inspect

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"inspect_artifact","artifact_id":"support_003"}'
```

### Prioritize

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"prioritize_artifact","artifact_id":"support_003"}'
```

### Plan

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"set_resolution_plan","plan":"Verify outage evidence, confirm the billing ledger, and route exceptions to finance review."}'
```

### Submit

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"submit_report","answer":"Proceed to refund review only after outage evidence and the billing ledger are confirmed. [support_001] [support_003]"}'
```

## Baseline Inference

The baseline runner is [inference.py](/C:/Users/nitis/Downloads/Meta%20OpenEnv/inference.py).

Submission-critical requirements satisfied:
- file name is exactly `inference.py`
- located at the project root
- uses the OpenAI client
- reads:
  - `API_BASE_URL` with a default
  - `MODEL_NAME` with a default
  - `HF_TOKEN` as the published credential path
  - `API_KEY` when validator proxy credentials are injected
- emits strict `[START]`, `[STEP]`, `[END]` stdout logs

### Environment variables

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `API_BASE_URL` | no | `https://router.huggingface.co/v1` | OpenAI-compatible endpoint |
| `MODEL_NAME` | no | `Qwen/Qwen2.5-72B-Instruct` | Model used for baseline inference |
| `HF_TOKEN` | yes | none | Primary token |
| `API_KEY` | no | none | Validator-injected proxy key; overrides `HF_TOKEN` |
| `RAG_ENV_URL` | no | `http://localhost:7860` | Environment base URL |
| `RAG_ENV_TASK` | no | `refund_triage_easy` | Preferred starting task |

## Baseline Scores

Current local validation run:

| Policy | refund_triage_easy | cross_function_brief_medium | executive_escalation_hard |
| --- | --- | --- | --- |
| baseline script | reproducible via `python validate.py` | reproducible via `python validate.py` | reproducible via `python validate.py` |

## Deployment

Live deployment:
- Space URL: [nitishrg15102007-rag-context-optimizer.hf.space](https://nitishrg15102007-rag-context-optimizer.hf.space)
- Space repo: [NITISHRG15102007/rag-context-optimizer](https://huggingface.co/spaces/NITISHRG15102007/rag-context-optimizer)

Recommended pre-submission flow:

```bash
docker build .
openenv validate
python validate.py
```
