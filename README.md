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

## Environment Description

RAG Context Optimization models a real production workflow: deciding what evidence should enter a prompt when context windows, latency budgets, and inference cost all matter. The current benchmark is organized around enterprise support operations, incident response, and release engineering. Teams building support assistants, internal knowledge copilots, analyst memo generators, engineering copilots, and compliance tools routinely face the same operational question: how do you include enough context to stay accurate without overpaying for irrelevant tokens or drowning the model in noise?

This environment turns that practical decision problem into an interactive benchmark. An agent must retrieve evidence, choose what to keep, decide when compression is worth the loss in detail, and then answer under an explicit token budget. That makes it useful both for evaluation and for training policies that behave more like real RAG orchestration systems instead of naive retrieve-everything pipelines.

It is not intended as a toy chat demo. It focuses on a real control problem in production AI systems: context allocation under uncertainty. The same decision pattern appears when a support assistant chooses which policy documents to include, when an internal copilot selects code or runbook context, when a trust-and-safety workflow chooses which moderation guidelines to retrieve, or when an analyst assistant decides which reports are worth paying to send to a model.

## Observation Space

`RagObservation` is the state payload returned by `/reset` and `/step`.

| Field | Type | Description |
| --- | --- | --- |
| `query` | `str` | The question the agent must answer. |
| `available_chunks` | `List[ChunkSummary]` | All chunk summaries the agent can choose from. |
| `selected_chunks` | `List[str]` | Chunk IDs currently selected into the working context. |
| `total_tokens_used` | `int` | Current token cost of the selected chunks after compression. |
| `token_budget` | `int` | Maximum tokens allowed for the active task. |
| `step_number` | `int` | Current episode step count. |
| `task_name` | `str` | Active task identifier. |
| `last_action_feedback` | `Optional[str]` | Feedback describing the previous action result. |

`ChunkSummary` contains:

| Field | Type | Description |
| --- | --- | --- |
| `chunk_id` | `str` | Unique chunk identifier. |
| `domain` | `str` | Source domain for the chunk. |
| `tokens` | `int` | Approximate token cost of the chunk. |
| `keywords` | `List[str]` | Lightweight retrieval hints for the agent. |

## Action Space

The agent interacts through `RagAction`.

| Action Type | Parameters | Effect |
| --- | --- | --- |
| `select_chunk` | `chunk_id` | Adds a chunk if it fits within the token budget. |
| `deselect_chunk` | `chunk_id` | Removes a previously selected chunk. |
| `compress_chunk` | `chunk_id`, `compression_ratio` | Reduces a selected chunk's token cost. |
| `submit_answer` | `answer` | Finalizes the episode and triggers grading. |

Answering guidance:

- final answers can include evidence citations using chunk IDs such as `[support_003]`
- citation use is rewarded when the cited chunks were actually selected
- unsupported answer content is penalized to discourage hallucinated policy claims
- task definitions now include explicit expected citation targets so citation quality is measured against task-specific gold evidence

Parameter notes:

| Parameter | Type | Description |
| --- | --- | --- |
| `chunk_id` | `Optional[str]` | Required for `select_chunk`, `deselect_chunk`, and `compress_chunk`. |
| `compression_ratio` | `Optional[float]` | Required for `compress_chunk`, valid range `0.3` to `0.9`. |
| `answer` | `Optional[str]` | Required for `submit_answer`. |

## Tasks

The benchmark includes three fixed tasks with progressively tighter tradeoffs. Together they test the same decisions production RAG pipelines make: single-domain support policy retrieval, cross-functional outage synthesis, and aggressive budget management during a security-flavored active incident.

| Task | Difficulty | Token Budget | Description |
| --- | --- | --- | --- |
| `single_domain_qa` | easy | `800` | Draft a customer refund recommendation using support policy evidence from one domain. |
| `cross_domain_synthesis` | medium | `500` | Prepare a cross-functional outage brief that aligns support triage with incident and release workflows. |
| `adversarial_compression` | hard | `300` | Draft a terse active-incident brief for a compromised admin account under a very tight budget. |

Scoring criteria:

- Retrieval precision: how well selected chunks match the relevant evidence set.
- Token efficiency: how much of the budget remains unused while still producing a valid answer.
- Answer quality: lexical overlap between the answer and the required chunk content.
- Required chunk coverage: fraction of mandatory chunks that were selected.
- Citation accuracy: whether the final answer cites selected evidence chunks.
- Unsupported-claim penalty: whether the answer introduces claims not grounded in the selected evidence.

This combination is intentionally operational rather than purely academic. A system that retrieves everything, ignores cost, or answers without the required evidence should not score well, because those are exactly the failure modes that make production RAG systems expensive or untrustworthy.

## Reward Function

The final grader score is normalized to `0.0` to `1.0` and uses this weighted average:

```text
score =
  0.25 * retrieval_precision +
  0.25 * token_efficiency +
  0.35 * answer_quality +
  0.15 * required_chunks_hit +
  0.10 * citation_accuracy -
  0.15 * hallucination_penalty
```

During the episode, the environment may also return small shaping rewards for useful selections, cautious compression, and efficient deselection behavior.

## Why This Is Useful

This benchmark is aimed at a real deployment gap in modern AI systems: most teams can retrieve documents, but far fewer can reliably decide which evidence deserves to consume prompt budget. That makes the environment useful for:

- benchmarking OpenEnv-compatible agents that need to trade off relevance, coverage, and cost
- training PyTorch policy or bandit-style controllers around deterministic environment feedback
- testing retrieval orchestration ideas before wiring them into production copilots
- evaluating whether an agent can stay faithful without resorting to retrieve-everything behavior
- comparing grounded, cited answer policies against unsupported free-form answering

The environment can also swap between enterprise corpus packs while keeping the same API and task identities. This makes it possible to compare whether a policy generalizes across differently worded support, incident, and release documents rather than overfitting one fixed wording of the corpus.

## Setup & Usage

### Docker build

```bash
docker build -t rag-context-optimizer .
```

### Docker run

```bash
docker run --rm -p 7860:7860 rag-context-optimizer
```

### Reset an episode

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name":"single_domain_qa"}'
```

### Take a step

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"select_chunk","chunk_id":"support_003"}'
```

### Submit an answer

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"submit_answer","answer":"Verify the outage timeline and billing ledger before issuing an immediate refund, and use goodwill credit only when the compensation matrix supports it. [support_001] [support_003] [support_005]"}'
```

### Inspect state

```bash
curl http://localhost:7860/state
```

### List tasks

```bash
curl http://localhost:7860/tasks
```

### Health check

```bash
curl http://localhost:7860/health
```

### Prompt optimization modes

The `/optimize-prompt` helper supports three user-facing modes:

- `balanced`: default rewrite path that shortens prompts while keeping key constraints.
- `grounded`: prefers inline citations and stronger evidence anchoring, even if the final prompt is slightly less compressed.
- `aggressive`: prioritizes maximum shortening over richer evidence phrasing.

Example:

```bash
curl -X POST http://localhost:7860/optimize-prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Draft a customer-safe incident update and cite evidence.","corpus_family":"enterprise_v2","compression_mode":"grounded"}'
```

## Baseline Scores

The following scores were recorded from the current `inference.py` baseline against the local FastAPI server on April 7, 2026. By default, the baseline expects a real model-backed run. If you explicitly set `ALLOW_BASELINE_FALLBACK=1`, it can fall back to a deterministic heuristic policy for offline validation, and that mode is intended for local smoke testing rather than benchmark claims.

| Model | single_domain_qa | cross_domain_synthesis | adversarial_compression |
| --- | --- | --- | --- |
| `Qwen/Qwen2.5-72B-Instruct` baseline script | `0.1524` | `0.1644` | `0.0000` |
| Deterministic greedy validator | `0.1203` | `0.0426` | `0.0000` |

These scores are intentionally modest. The benchmark is designed so that strong performance requires making tradeoffs between relevance, coverage, and token use rather than simply selecting the largest set of plausible chunks.

## Deployment

This repository is designed to run as a Docker-based Hugging Face Space. After creating a Space with Docker SDK enabled, push this repository and Spaces will build the container from `Dockerfile`.

Suggested Space URL format:

```text
https://huggingface.co/spaces/<your-username>/rag-context-optimizer
```

Environment variables to configure in the Space settings:

| Variable | Purpose |
| --- | --- |
| `HF_TOKEN` | Primary API key used by the baseline script for OpenAI-compatible model calls through the configured router. |
| `OPENAI_API_KEY` | Optional fallback key for local compatibility if `HF_TOKEN` is not set. |
| `API_BASE_URL` | OpenAI-compatible endpoint, default `https://router.huggingface.co/v1`. |
| `MODEL_NAME` | Baseline model, default `Qwen/Qwen2.5-72B-Instruct`. |
| `RAG_ENV_URL` | Environment base URL, usually `http://localhost:7860` inside the container. |
| `RAG_ENV_TASK` | Optional starting task for the baseline runner. |
| `ALLOW_BASELINE_FALLBACK` | Optional offline mode for `inference.py`. When set to `1`, the script may use a deterministic fallback policy if no model key is available. |
| `RAG_CORPUS_PATH` | Optional path to an alternate JSONL corpus so the same environment can run against another enterprise policy pack. |
| `RAG_CORPUS_FAMILY` | Optional built-in corpus selector. Current families: `enterprise_v1`, `enterprise_v2`. |

FastAPI also exposes interactive API documentation at `/docs` once the app is running.
