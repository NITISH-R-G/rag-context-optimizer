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
  - rag
  - benchmark
  - retrieval
---

# rag-context-optimizer

## Overview

`rag-context-optimizer` is an OpenEnv-compatible environment for enterprise context selection under token constraints.

It models a real workflow that teams actually face in production AI systems:
- customer support agents deciding which policy evidence to retrieve
- incident responders deciding which runbook details to include
- reliability and release teams deciding what context is worth paying to send to a model

Instead of rewarding naive retrieve-everything behavior, the environment rewards agents that choose relevant evidence, manage token budget carefully, compress when appropriate, and produce grounded answers.

The current benchmark is organized around three enterprise knowledge workflows:
- support policy triage
- outage and incident synthesis
- high-pressure adversarial compression during a security-flavored active incident

## Why This Is A Real Task

This is not a toy prompt-shortening demo. It represents a production control problem:
- too little context causes low-quality or unsafe answers
- too much context increases latency and cost
- irrelevant context increases hallucination risk
- uncited answers are harder to trust in support and incident workflows

That makes the environment useful for:
- benchmarking OpenEnv agents
- training PyTorch or bandit-style controllers over deterministic feedback
- evaluating retrieval and compression policies before production rollout
- comparing grounded answering strategies against unsupported free-form behavior

## OpenEnv API

The environment exposes the standard HTTP interface:
- `POST /reset`
- `POST /step`
- `GET /state`

It also includes:
- `GET /health`
- `GET /tasks`
- `POST /optimize-step`
- `POST /optimize-prompt`

Metadata lives in [openenv.yaml](/C:/Users/User/Documents/META%20HACKATHON/rag-context-optimizer/openenv.yaml).

## Observation Space

`RagObservation` is returned by `/reset` and `/step`.

| Field | Type | Description |
| --- | --- | --- |
| `query` | `str` | The task query the agent must answer. |
| `available_chunks` | `List[ChunkSummary]` | Candidate evidence chunks the agent can act on. |
| `selected_chunks` | `List[str]` | Chunk IDs currently selected into working context. |
| `total_tokens_used` | `int` | Current context token cost after compression. |
| `token_budget` | `int` | Maximum allowed token budget for the episode. |
| `step_number` | `int` | Current episode step count. |
| `task_name` | `str` | Active task identifier. |
| `last_action_feedback` | `Optional[str]` | Human-readable feedback for the previous action. |

`ChunkSummary` includes:

| Field | Type | Description |
| --- | --- | --- |
| `chunk_id` | `str` | Unique evidence identifier. |
| `domain` | `str` | Chunk source domain. |
| `tokens` | `int` | Approximate token cost. |
| `keywords` | `List[str]` | Retrieval hints for the agent. |

## Action Space

The agent acts through `RagAction`.

| Action Type | Parameters | Effect |
| --- | --- | --- |
| `select_chunk` | `chunk_id` | Adds a chunk if it fits the remaining budget. |
| `deselect_chunk` | `chunk_id` | Removes a selected chunk. |
| `compress_chunk` | `chunk_id`, `compression_ratio` | Reduces token cost for a selected chunk. |
| `submit_answer` | `answer` | Ends the episode and triggers grading. |

Answering guidance:
- final answers may cite evidence with chunk IDs such as `[support_003]`
- citation accuracy is rewarded
- unsupported claims are penalized
- task definitions include expected citation targets for deterministic grading

## Tasks

The benchmark contains three tasks with increasing difficulty.

| Task | Difficulty | Max Steps | Token Budget | Description |
| --- | --- | --- | --- | --- |
| `single_domain_qa` | easy | `6` | `800` | Draft a customer refund recommendation using support policy evidence. |
| `cross_domain_synthesis` | medium | `8` | `500` | Produce a cross-functional outage brief spanning support, incident, and reliability evidence. |
| `adversarial_compression` | hard | `10` | `300` | Produce a tightly budgeted active-incident brief for a compromised admin account. |

These tasks are implemented in [env/tasks.py](/C:/Users/User/Documents/META%20HACKATHON/rag-context-optimizer/env/tasks.py) and graded in [env/graders.py](/C:/Users/User/Documents/META%20HACKATHON/rag-context-optimizer/env/graders.py).

## Reward Design

The final score is normalized to `[0.0, 1.0]` and combines:
- retrieval precision
- token efficiency
- answer quality
- required chunk coverage
- citation accuracy
- hallucination penalty

The environment also emits trajectory-level shaping rewards for:
- selecting useful evidence
- compressing cautiously
- avoiding wasteful budget use
- removing unhelpful context

This produces denser signal than a submit-only environment.

## Prompt Optimization Modes

The `/optimize-prompt` helper supports three modes:
- `balanced`: default mode, preserves constraints while shortening
- `grounded`: prefers inline citations and evidence anchoring, even if the final prompt is slightly less compressed
- `aggressive`: prioritizes maximum shortening over richer evidence phrasing

Example:

```bash
curl -X POST http://localhost:7860/optimize-prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Draft a customer-safe incident update and cite evidence.","corpus_family":"enterprise_v2","compression_mode":"grounded"}'
```

## Local Setup

### Requirements

- Python 3.11 recommended
- Docker
- `openenv-core`

### Install

```bash
pip install -r requirements.txt
pip install openenv-core
```

### Run locally

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Test locally

```bash
docker build .
openenv validate
python validate.py
```

## API Usage

### Reset

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name":"single_domain_qa"}'
```

### Step

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"select_chunk","chunk_id":"support_003"}'
```

### Submit

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"submit_answer","answer":"Verify outage evidence and billing records before issuing an immediate refund. [support_001] [support_003]"}'
```

### State

```bash
curl http://localhost:7860/state
```

### Health

```bash
curl http://localhost:7860/health
```

## Baseline Inference

The baseline runner is [inference.py](/C:/Users/User/Documents/META%20HACKATHON/rag-context-optimizer/inference.py).

Submission-critical requirements:
- file name is exactly `inference.py`
- located at the project root
- uses the OpenAI client
- reads:
  - `API_BASE_URL`
  - `MODEL_NAME` with a default
  - `API_KEY`
- emits strict stdout lines only in this format:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
```

Important runtime behavior:
- `API_BASE_URL` and `API_KEY` are required for submission runs
- `HF_TOKEN` is supported only as a legacy local credential for manual smoke tests
- `ALLOW_BASELINE_FALLBACK=1` enables deterministic offline fallback for local validation only
- offline fallback is intended for smoke testing, not benchmark claims

### Environment variables

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `API_BASE_URL` | yes for submission | none | LiteLLM / OpenAI-compatible endpoint for model calls |
| `MODEL_NAME` | no | `Qwen/Qwen2.5-72B-Instruct` | Model used for baseline inference |
| `API_KEY` | yes for submission | none | API key for the injected LiteLLM proxy |
| `HF_TOKEN` | no | none | Optional legacy local token for manual smoke tests |
| `RAG_ENV_URL` | no | `http://localhost:7860` | Environment base URL |
| `RAG_ENV_TASK` | no | `single_domain_qa` | Preferred starting task order |
| `ALLOW_BASELINE_FALLBACK` | no | unset | Optional offline smoke-test mode |
| `RAG_CORPUS_PATH` | no | unset | Optional alternate corpus file |
| `RAG_CORPUS_FAMILY` | no | unset | Optional built-in corpus pack selector |

## Baseline Scores

Recorded local scores for the current baseline:

| Policy | single_domain_qa | cross_domain_synthesis | adversarial_compression |
| --- | --- | --- | --- |
| baseline script | `0.1524` | `0.1644` | `0.0000` |
| deterministic validator policy | `0.4821` | `0.3270` | `0.2740` |

These scores are intentionally modest. Strong performance requires balancing relevance, grounding, and token budget rather than simply selecting the largest plausible evidence set.

## Deployment

This repository is designed for a Docker-based Hugging Face Space.

Live deployment:
- Space URL: [nitishrg15102007-rag-context-optimizer.hf.space](https://nitishrg15102007-rag-context-optimizer.hf.space)
- Space repo: [NITISHRG15102007/rag-context-optimizer](https://huggingface.co/spaces/NITISHRG15102007/rag-context-optimizer)

Suggested validation flow before submission:

```bash
docker build .
openenv validate
python validate.py
```

External validator example:

```bash
./validate-submission.sh https://nitishrg15102007-rag-context-optimizer.hf.space ./rag-context-optimizer
```

FastAPI docs are available at `/docs` when the server is running.
