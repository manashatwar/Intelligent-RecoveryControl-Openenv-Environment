# IRCE: Learning Recovery Under Tool Failure

**A deterministic OpenEnv benchmark for budget-aware recovery decisions in production-style tool workflows.**

IRCE focuses on one question:

> When a tool call fails, should the agent retry, modify the request, switch tools, replan, or escalate?

The project is fully implemented and includes:

- a working OpenEnv environment
- 3 task levels: easy, medium, hard
- deterministic graders with scores in `[0.0, 1.0]`
- a reproducible baseline inference script
- Docker and OpenEnv validation support

## Why This Problem Matters

Many LLM agents still recover from failure with simple rules such as:

- retry up to 3 times
- switch tools after N errors
- stop after a fixed budget

Those rules are easy to ship, but they are weak in real systems.

Real tool failures are not all the same:

- some are transient and worth retrying
- some are structural and need input repair
- some come from rate limits and need a cooldown or tool switch
- some produce ambiguous outputs that look partly useful but are not fully trustworthy
- some get worse over time as reliability drifts

The cost of bad recovery is real:

- wasted API spend
- wasted tokens
- higher latency
- repeated calls into degraded systems
- poor user experience from obvious agent thrashing

IRCE turns that practical recovery problem into a clean, explainable benchmark.

## Core Idea

IRCE is not mainly about tool selection.

It is about **recovery policy**:

- retry when the failure looks transient
- modify when the request looks structurally wrong
- switch when the active tool path looks degraded or rate-limited
- replan when the signal is ambiguous
- escalate when continued action is wasteful

The benchmark asks the agent to make recovery decisions under partial information, budget pressure, ambiguous outcomes, and changing failure modes.

## Why RL Is a Good Fit

This is a sequential decision problem with exactly the properties that make hand-written rules brittle:

- **Partial observability**: the agent sees only observations, not the true hidden failure cause
- **Non-stationarity**: the hard task can drift from transient issues into harder failures and rate limits
- **Delayed reward**: good recovery choices often pay off a few steps later
- **Action consequences**: switching tools, consuming backup routes, or escalating all change later options
- **Rule brittleness**: a rule like "retry twice, then switch" ignores context such as budget, repeated error history, and noisy observations

A fixed `max_retries` policy cannot handle those tradeoffs cleanly.

## Environment Design

### Action Space

The action space is intentionally small:

| Action | Meaning |
| --- | --- |
| `RETRY` | Try the current path again |
| `MODIFY` | Change the request to address structural failure |
| `SWITCH` | Move between primary and backup tool paths |
| `REPLAN` | Reframe the next attempt, especially after ambiguity or cooldown pressure |
| `ESCALATE` | Stop early and hand off |

### Observation Space

Each observation is compact and LLM-friendly:

- `goal`
- `tool_result`
- `error_type`
- `same_error_count`
- `budget_remaining`
- `step_count`
- `last_action_error`
- `active_tool`
- `cooldown_remaining`
- `progress_hint`
- `history_tail`
- `status_summary`

Important detail:

- `tool_result` can be `SUCCESS`, `ERROR`, or `AMBIGUOUS`
- `progress_hint` is useful but not perfectly trustworthy on noisier tasks
- `history_tail` gives a short recent trace instead of a full trajectory dump
- `status_summary` gives a one-line, judge-readable decision context for LLM agents

### Hidden State

The hidden state keeps the environment realistic while staying lightweight:

- current hidden failure type
- current tool path (`primary` or `backup`)
- remaining cooldown from rate limits
- unresolved ambiguous outcomes
- cumulative task progress
- remaining budget

### Core Mechanics

IRCE adds two simple but high-value mechanics that make it feel less like a toy benchmark:

1. **Ambiguous outcomes**

Some actions do not cleanly fail or succeed. They return `AMBIGUOUS`, give partial progress, and force the agent to decide whether to retry, modify, or replan.

2. **Tool tradeoffs and cooldowns**

The backup tool path is often more reliable, but it costs more budget. Rate-limited states expose a cooldown signal, so repeatedly retrying the same path is meaningfully bad.

These mechanics are cheap to simulate, deterministic with a seed, and easy for judges to understand.

## Task Suite

IRCE has three deterministic task profiles with clear difficulty progression.

### Task 1: Easy

- transient and hard failures only
- generous budget
- almost no observation noise
- low ambiguity

This task checks whether the agent can make basic repair decisions without overreacting.

### Task 2: Medium

- adds rate limits and cooldown behavior
- moderate ambiguity
- moderate budget pressure
- primary versus backup tradeoff matters

This task tests whether the agent can stop bad retries and route around degraded tool paths.

### Task 3: Hard

- lower budget
- noisier error labels
- drifting hidden failure modes
- cascading penalties on repeated failure
- costly backup routing

This task asks the agent to stay stable under uncertainty while still finishing efficiently.

## Reward Design

The reward is dense, deterministic, and aligned with recovery quality.

Positive signals:

- `+0.3` for useful progress
- `+0.15` for ambiguous partial progress
- `+1.0` for task completion

Negative signals:

- `-0.1` per step
- `-0.2` for repeated same-error failures
- `-0.3` for bad retries on `HARD` or `RATE_LIMIT`
- switch and backup-cost penalty
- extra cascade penalty on harder tasks
- early escalation penalty when the agent gives up too soon

This reward encourages agents to move quickly, avoid thrashing, and treat ambiguity as something to resolve rather than ignore.

## Grading

Grading is deterministic and returns a final score in `[0.0, 1.0]`.

The current grader combines four components:

- **Completion**: full credit for success, partial credit for controlled escalation after meaningful progress
- **Efficiency**: fewer steps is better
- **Cost**: preserving budget is better
- **Recovery quality**: fewer bad retries, better ambiguity handling, and cleaner exits

Formula:

```text
score =
    0.45 * completion
  + 0.25 * efficiency
  + 0.15 * cost
  + 0.15 * recovery_quality
```

This makes the benchmark more faithful to real-world recovery behavior than scoring on completion alone.

## Example Episode

Example trajectory from the current implementation:

| Step | Action | Tool Result | Error Type | Tool | Budget | Reward | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | reset | ERROR | HARD | primary | 1.00 | 0.000 | Workflow starts in a hard-failure state |
| 1 | MODIFY | SUCCESS | TRANSIENT | primary | 0.90 | 0.199 | Input repair helps and progress jumps |
| 2 | RETRY | AMBIGUOUS | TRANSIENT | primary | 0.80 | 0.049 | Partial progress, but signal is unclear |
| 3 | REPLAN | AMBIGUOUS | TRANSIENT | primary | 0.70 | 0.049 | Agent stabilizes the next attempt |
| 4 | REPLAN | SUCCESS | TRANSIENT | primary | 0.60 | 0.899 | Ambiguity resolves and the task completes |

This is the central IRCE behavior: the agent is judged on what it does after failure and uncertainty, not just on whether it can call a tool.

### Hard-Task Snapshot

The hard task is designed to look like a degraded production incident rather than a clean failure:

```text
Goal: Recover an unstable workflow under drift, ambiguity, and tight budget limits.
Status: result=ERROR; error=RATE_LIMIT; tool=primary; budget=0.22; step=4/7; cooldown=1; repeat_errors=1; progress_hint=0.68; recent=SWITCH->SUCCESS [...] | REPLAN->ERROR [...]
```

In this state, a strong policy should avoid immediate retry and prefer `SWITCH` or `REPLAN`, because cooldown pressure, low budget, and noisy signals make repeated retries wasteful.

## Baseline Results

The provided `inference.py` uses the OpenAI client for model calls and falls back to a deterministic rule-based policy when credentials or model requests are unavailable. The fallback policy uses:

- `error_type`
- `same_error_count`
- `budget_remaining`
- `step_count`
- `active_tool`
- `cooldown_remaining`
- `progress_hint`
- short memory of recent outcomes

Current reproducible baseline with the default seed:

```text
task_1 score: 0.815
task_2 score: 0.898
task_3 score: 0.874
average score: 0.863
```

Across a 100-seed sweep during development, the upgraded baseline improved the average score from `0.751` to `0.805`.

## Why IRCE Is Novel

Benchmarks such as ToolBench and API-Bank mostly focus on whether an agent can choose and use tools correctly to finish a task.

IRCE focuses on a different and highly practical question:

> What should the agent do after a tool failure or ambiguous outcome?

That difference matters in production systems, where many agent failures come from weak recovery behavior rather than weak tool coverage.

IRCE is therefore a benchmark for:

- recovery under uncertainty
- cost-aware tool fallback
- ambiguity resolution
- stability under drift and cooldown pressure

## Setup

### Install

```bash
uv sync
```

Activate the project environment before running `openenv` directly:

```bash
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
```

### Run the Environment Server

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Run the Baseline Inference Script

Set the model environment variables first:

```bash
cp .env.example .env
```

Required variables:

- `OPENAI_API_KEY`
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

If the model credentials are missing or the request fails, `inference.py` falls back to the deterministic baseline policy so the script still completes reproducibly.

```bash
uv run python inference.py
```

### Quick API Check

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/reset
curl -X POST http://127.0.0.1:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action":{"action_type":"RETRY"}}'
```

## Deployment

### Docker

```bash
docker build -t irce:dev .
docker run --rm -p 8000:8000 irce:dev
```

### OpenEnv and Hugging Face Space Readiness

This project is packaged for OpenEnv deployment:

- `openenv.yaml` is included
- `Dockerfile` is included
- root `server` entrypoints are present
- `openenv validate` passes

Validation:

```bash
openenv validate .
```

## Future Work

The current benchmark is intentionally lightweight. Good next steps are:

- learned policies instead of heuristic control
- multi-agent recovery roles
- replay from real API failure traces
- richer escalation policies and handoff objectives

## Why IRCE Is Useful

IRCE is small enough to run cheaply, but the decision problem is real.

In production agent systems, the most expensive mistake is often not the first failure. It is what the agent does next.

IRCE gives that recovery problem a deterministic, OpenEnv-compatible benchmark that is easy to validate, easy to explain, and hard enough to be interesting.
