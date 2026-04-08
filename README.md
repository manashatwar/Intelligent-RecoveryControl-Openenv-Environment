---
title: AI Pipeline Recovery
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# IRCE: Training Agents to Recover When Workflows Break

**Every production AI workflow today handles tool failure with hard-coded rules — retry twice, then stop. Those rules are cheap to write but expensive in production: a misconfigured retry policy burns tokens, exhausts context windows, and creates the infinite loop problem that keeps engineers up at night. IRCE is an RL training environment for the recovery policy itself — so that instead of a hard-coded config, an agent learns when to retry, when to switch tools, when to modify its request, and when to stop.**


IRCE focuses on one question:

> When a step in an AI workflow fails, should the orchestrating agent retry, modify the request, switch tools, replan, or escalate to a human?

Simple Example workflow:
User request → fetch data → call LLM → send result

If the LLM call fails, the agent must decide whether to retry, switch providers, or replan the request.

The project is fully implemented and includes:

- a working OpenEnv environment
- 3 task levels: easy, medium, hard
- deterministic graders with scores in `[0.0, 1.0]`
- a reproducible baseline inference script
- Docker and OpenEnv validation support

## Why This Problem Matters

AI workflow orchestrators — agents that complete tasks by chaining multiple tool calls — are everywhere. LangChain agents, AutoGen pipelines, Google ADK workflows, OpenAI Assistants with tools. Every one of them needs to handle failure.

Today, every team handles failure the same way for example:
```python
if fail:
    retry(max_attempts=3)
else:
    stop()
```

This is the wrong approach. Real tool failures inside workflows are not all the same:

- some are transient and worth retrying immediately
- some are structural and need the input repaired before retrying
- some come from rate limits and need a cooldown or a different tool
- some produce ambiguous partial outputs that look useful but cannot be trusted
- some drift over time as a dependency degrades under load

A fixed `max_retries=3` rule treats all of these the same. The result in production is wasted API spend, exhausted context windows, repeated calls into already-degraded systems, and workflows that loop until they are killed manually.

IRCE turns that recovery decision into a trainable problem. Instead of a rule, the agent learns a policy — one that reads the actual context (error type, budget remaining, cooldown status, recent history) and makes the right call.

Existing systems rely on fixed retry and fallback heuristics; IRCE instead treats recovery as a learnable policy.

## Real-World Scenarios

Each task is designed to require a qualitatively different kind of reasoning, not just tolerance for more noise.

- **Task 1: Easy — Single-Signal Decisions** 
A simple workflow where a single tool call occasionally fails, and the signal is clean.
Each step presents one dominant signal. TRANSIENT errors call for RETRY. HARD errors call for MODIFY or SWITCH. The budget is generous and error labels are reliable. The agent only needs to read one signal correctly and act on it.

This task checks whether the agent understands the basic meaning of each error type and can map it to the correct recovery action without overreacting or overthinking.

- **Task 2: Medium — Conflicting-Signal Decisions**
A multi-step workflow where failures introduce competing signals and tradeoffs.
The correct action now depends on combining multiple signals. A RATE_LIMIT error with cooldown remaining and low budget points to SWITCH — but a RATE_LIMIT error with high budget and partial progress may justify REPLAN instead. Neither option is obviously correct from a single signal alone.

This task tests whether the agent can reason across error type, budget, cooldown, and tool state together — rather than reacting to whichever signal is most obvious.

- **Task 3: Hard — Adaptive Decisions Under a Shifting Policy**
A production-like workflow where system behavior changes during execution.
The correct recovery policy shifts mid-episode. What works in early steps stops working later as the hidden failure mode drifts from TRANSIENT to HARD to RATE_LIMIT. Error signals become noisier, and feedback is less reliable.
The agent must recognize that its current strategy is no longer valid and adapt — without clear confirmation that the new strategy is correct.

This task tests whether the agent can handle non-stationarity and update its behavior in real time, rather than executing a fixed strategy from start to finish.

## Core Idea

IRCE is not about tool selection.

It is about **recovery policy** — what an AI workflow orchestrator does *after* a step fails:

- retry when the failure looks transient
- modify when the request looks structurally wrong
- switch when the active tool path looks degraded or rate-limited
- replan when the signal is ambiguous
- escalate when continued action is wasteful and a human needs to intervene

The benchmark asks the agent to make these decisions under partial information, budget pressure, ambiguous outcomes, and changing failure modes — conditions that hard-coded rules cannot handle cleanly.

## Why RL Is a Good Fit

This is a sequential decision problem with exactly the properties that make hand-written rules brittle:

- **Partial observability**: the agent sees only observations, not the true hidden failure cause
- **Non-stationarity**: the hard task can drift from transient issues into harder failures and rate limits
- **Delayed reward**: good recovery choices often pay off a few steps later
- **Action consequences**: switching tools, consuming backup routes, or escalating all change later options
- **Rule brittleness**: a rule like "retry twice, then switch" ignores context such as budget, repeated error history, and noisy observations

A fixed `max_retries` policy cannot handle those tradeoffs cleanly. A learned policy can.

## Environment Design

### Action Space

The action space is intentionally small and maps directly to decisions a workflow orchestrator makes:

| Action | Meaning |
| --- | --- |
| `RETRY` | Try the current tool path again |
| `MODIFY` | Fix the request input to address a structural failure |
| `SWITCH` | Move to the backup tool path |
| `REPLAN` | Reframe the next attempt, especially after ambiguity or cooldown pressure |
| `ESCALATE` | Stop and hand off to a human or fallback system |

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

Key details:

- `tool_result` can be `SUCCESS`, `ERROR`, or `AMBIGUOUS`
- `progress_hint` is useful but not perfectly trustworthy on noisier tasks
- `history_tail` gives a short recent trace instead of a full trajectory dump
- `status_summary` gives a one-line, human-readable decision context for LLM agents

### Hidden State

The hidden state keeps the environment realistic while staying lightweight:

- current hidden failure type
- current tool path (`primary` or `backup`)
- remaining cooldown from rate limits
- unresolved ambiguous outcomes
- cumulative task progress
- remaining budget

### Core Mechanics

IRCE adds two mechanics that capture failure modes every production orchestration engineer has seen:

1. **Ambiguous outcomes**

Some tool calls do not cleanly fail or succeed. They return `AMBIGUOUS`, give partial progress, and force the agent to decide whether to retry, modify, or replan. This mirrors real cases where an API returns partial data, an LLM output passes schema validation but is semantically wrong, or a downstream service responds slowly with an incomplete result.

2. **Tool tradeoffs and cooldowns**

The backup tool path is often more reliable but costs more budget. Rate-limited states expose a cooldown signal, making repeated retries on the same path meaningfully bad. The agent has to weigh cost against reliability in real time.

These mechanics are cheap to simulate, deterministic with a seed, and directly reflect the failure patterns teams encounter when running LLM-powered workflows in production.

## Task Suite

IRCE has three deterministic task profiles with clear difficulty progression.

### Task 1: Easy — Single API, Transient Failures

- transient and hard failures only
- generous budget
- almost no observation noise
- low ambiguity

A single tool inside the workflow fails occasionally. The agent must make basic repair decisions without overreacting to noise.

### Task 2: Medium — Rate Limits and Routing Tradeoffs

- adds rate limits and cooldown behavior
- moderate ambiguity
- moderate budget pressure
- primary versus backup tradeoff matters

The workflow hits rate limits mid-execution. Retrying the same provider wastes budget. The agent must recognize the pattern and route around degraded paths.

### Task 3: Hard — Drifting Reliability Under Pressure

- lower budget
- noisier error labels
- drifting hidden failure modes
- cascading penalties on repeated failure
- costly backup routing

A third-party tool degrades step by step during execution. Error signals are noisy. The backup path is expensive. The agent must stay stable and efficient under conditions that look nothing like the easy case.

## Reward Design

The reward is dense, deterministic, and aligned with recovery quality.

Each step produces a structured reward composed of multiple components:

- step cost to discourage unnecessary actions
- progress bonus for meaningful forward movement
- ambiguity bonus for resolving partial outcomes
- completion bonus for finishing the task
- penalties for repeated failures, bad retries, and inefficient actions

Key signals:

- `+0.3` for strong progress
- `+0.15` for resolving ambiguity
- `+0.9` for task completion
- `-0.1` per step (efficiency pressure)
- `-0.2` for repeated same-error failures
- `-0.3` for bad retries on `HARD` or `RATE_LIMIT`
- switching cost and backup routing penalties
- cascade penalties on harder tasks
- early escalation penalty

Each step reward is computed using a structured breakdown (see `RewardBreakdown` in code), ensuring transparency and deterministic evaluation.

## Grading

Grading is deterministic and returns a final score in `[0.0, 1.0]`.

The grader evaluates four components:

- **Completion**
  - Full credit for successful completion
  - Partial credit for controlled escalation after meaningful progress

- **Efficiency**
  - Fewer steps → higher score

- **Cost**
  - Preserving budget → higher score

- **Recovery quality**
  - Penalizes bad retries
  - Rewards resolving ambiguity
  - Penalizes exhausting budget
  - Rewards controlled escalation

Formula:

score =
    0.45 * completion
  + 0.25 * efficiency
  + 0.15 * cost
  + 0.15 * recovery_quality

Recovery quality explicitly measures how well the agent handles failure patterns, not just whether it completes the task.

## Example Episode

Example trajectory from the current implementation:

| Step | Action | Tool Result | Error Type | Tool | Budget | Reward | Notes |
| ---  | ---    | ---         | ---        | ---  | ---    | ---    | ---   |
| 0    | reset  | ERROR       | HARD       | primary | 1.00 | 0.000 | Workflow starts in a hard-failure state |
| 1    | MODIFY | SUCCESS     | TRANSIENT  | primary | 0.90 | 0.199 | Input repair helps and progress jumps |
| 2    | RETRY  | AMBIGUOUS   | TRANSIENT  | primary | 0.80 | 0.049 | Partial progress, but signal is unclear |
| 3    | REPLAN | AMBIGUOUS   | TRANSIENT  | primary | 0.70 | 0.049 | Agent stabilizes the next attempt |
| 4    | REPLAN | SUCCESS     | TRANSIENT  | primary | 0.60 | 0.899 | Ambiguity resolves and the task completes |

This is the central IRCE behavior: the agent is judged on what it does after failure and uncertainty, not just on whether it can call a tool.

### Hard-Task Snapshot

The hard task is designed to look like a degraded production incident rather than a clean failure:

```text
Goal: Recover an unstable workflow under drift, ambiguity, and tight budget limits.
Status: result=ERROR; error=RATE_LIMIT; tool=primary; budget=0.22; step=4/7; cooldown=1; repeat_errors=1; progress_hint=0.68; recent=SWITCH->SUCCESS [...] | REPLAN->ERROR [...]
```

In this state, a strong policy should avoid immediate retry and prefer `SWITCH` or `REPLAN`, because cooldown pressure, low budget, and noisy signals make repeated retries wasteful.

## Baseline Results

The provided `inference.py` uses the OpenAI client with a strict rule-based prompting strategy.

The model is instructed to follow a priority-ordered decision policy:

- escalate under low budget or repeated failure
- switch on rate limits or persistent hard failures
- modify inputs for structural issues
- replan under ambiguity or cooldown constraints
- retry only when safe (transient + no cooldown)

The script enforces:

- strict parsing of model output into valid actions
- rejection of invalid or ambiguous outputs
- deterministic execution with a fixed seed

Each step logs:

- action taken
- reward
- error type
- tool result (SUCCESS / ERROR / AMBIGUOUS)

Example log format:

[STEP] step=2 action=MODIFY reward=0.20 done=false error=TRANSIENT result=SUCCESS

This ensures reproducibility and compatibility with OpenEnv evaluation.

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
uv run uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
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
curl http://127.0.0.1:7860/health
curl -X POST http://127.0.0.1:7860/reset
curl -X POST http://127.0.0.1:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action":{"action_type":"RETRY"}}'
```

## Deployment

### Docker

```bash
docker build -t irce:dev .
docker run --rm -p 7860:7860 irce:dev
```

### Hugging Face Spaces Deployment

This project is fully configured for Hugging Face Spaces deployment.

#### Prerequisites

1. Create a Hugging Face account at https://huggingface.co
2. Install the HF CLI:

```bash
pip install huggingface-hub
huggingface-cli login
```

#### Deploy to Hugging Face Spaces

1. **Create a new Space on Hugging Face**

   Visit https://huggingface.co/new-space and select:
   - Space name: `irce` (or your preferred name)
   - License: OpenRAIL-M (or your choice)
   - Space SDK: `Docker`

2. **Push your code to the Space**

   ```bash
   git remote add space https://huggingface.co/spaces/{your-username}/irce
   git push space main
   ```

   Or clone and push:

   ```bash
   git clone https://huggingface.co/spaces/{your-username}/irce
   cd irce
   git remote add upstream {your-original-repo}
   git pull upstream main
   git push origin main
   ```

3. **Set up Environment Variables (Secrets)**

   In your Space settings (https://huggingface.co/spaces/{your-username}/irce/settings):
   
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `API_BASE_URL`: Your API endpoint (if using custom base)
   - `MODEL_NAME`: Model name (default: gpt-4o)
   - `HF_TOKEN`: Your Hugging Face token

4. **Configuration**

   The Space will automatically:
   - Read the `Dockerfile`
   - Build and deploy on port `7860`
   - Expose the FastAPI app at the Space URL

#### Space YAML Configuration (Optional)

If you want to customize Space behavior, create a `.huggingface/space_config.yaml`:

```yaml
title: IRCE - Intelligent Recovery Control Environment
description: >
  An RL training environment for recovery policy in AI workflows.
  Agents learn when to retry, modify, switch tools, replan, or escalate.
app_port: 7860
sdk: docker
emoji: 🤖
```

#### Quick Push to HF Spaces

Once your Space is created on Hugging Face:

```bash
# Replace {your-username} with your actual Hugging Face username
git remote add space https://huggingface.co/spaces/{your-username}/irce

# Push your code to the Space
git push space main
```

Or if you already have an origin remote, replace it:

```bash
git remote set-url space https://huggingface.co/spaces/{your-username}/irce
git push space main
```

**After pushing:**
1. Go to your Space URL: `https://huggingface.co/spaces/{your-username}/irce`
2. Wait for the build to complete (watch logs in Space Settings)
3. Once live, add secrets in Space Settings → Repository secrets
4. The app will automatically restart with your secrets

#### Verify OpenEnv Compatibility

This project is also packaged for OpenEnv deployment:

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
