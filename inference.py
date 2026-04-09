import asyncio
import os
import re
import sys
from typing import List, Optional

# Ensure local src/ package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from dotenv import load_dotenv
from openai import OpenAI

try:
    from ai_pipeline_recovery.client import IRCEEnvClient
    from ai_pipeline_recovery.models import IRCEAction, IRCEObservation
except ImportError as e:
    print(f"[DEBUG] Import error: {e}", flush=True)
    sys.exit(1)

load_dotenv()

# ===== CONFIG =====
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct:novita")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
IMAGE_NAME = os.getenv("IMAGE_NAME")

SUPPORTED_ACTIONS = {"RETRY", "MODIFY", "SWITCH", "REPLAN", "ESCALATE"}

# Match actual task max_steps from tasks.py (6, 6, 7) with small buffer
TASKS = {
    1: {"name": "easy",   "max_steps": 8},
    2: {"name": "medium", "max_steps": 8},
    3: {"name": "hard",   "max_steps": 9},
}

# Theoretical max cumulative reward per episode (completion 0.9 + progress 0.3 - costs)
# Using 1.5 as a safe normalizer so score stays in [0,1] for good episodes
MAX_TOTAL_REWARD = 1.5

SUCCESS_SCORE_THRESHOLD = 0.1

SYSTEM_PROMPT = """You are an AI workflow recovery agent.

Given an observation, output EXACTLY one word: RETRY, MODIFY, SWITCH, REPLAN, or ESCALATE.

Apply these rules in priority order — use the FIRST matching rule:

1. ESCALATE  — if budget_remaining < 0.15 AND progress < 0.50
2. ESCALATE  — if same_error_count >= 4
3. SWITCH    — if error_type = RATE_LIMIT
4. SWITCH    — if error_type = HARD AND same_error_count >= 2
5. MODIFY    — if error_type = HARD AND same_error_count < 2
6. REPLAN    — if tool_result = AMBIGUOUS
7. RETRY     — if error_type = TRANSIENT AND cooldown_remaining = 0
8. REPLAN    — if cooldown_remaining > 0
9. MODIFY    — default

Output ONLY one word. No explanation."""


# ===== LOGGING =====
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ===== ACTION PARSER =====
def parse_action(text: str) -> str:
    if not text or not text.strip():
        return "MODIFY"
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"[`*_#>\-]", "", cleaned).strip().upper()
    cleaned = cleaned.replace(".", "").replace(",", "")
    first_line = cleaned.split("\n")[0].strip()
    if first_line in SUPPORTED_ACTIONS:
        return first_line
    for action in SUPPORTED_ACTIONS:
        if re.search(rf"\b{action}\b", first_line):
            return action
    for action in SUPPORTED_ACTIONS:
        if re.search(rf"\b{action}\b", cleaned):
            return action
    return "MODIFY"


# ===== MODEL CALL =====
def get_action(client: OpenAI, obs: IRCEObservation, step: int) -> str:
    prompt = (
        f"Step: {step}\n"
        f"Goal: {obs.goal}\n\n"
        f"tool_result: {obs.tool_result}\n"
        f"error_type: {obs.error_type}\n"
        f"same_error_count: {obs.same_error_count}\n"
        f"cooldown_remaining: {obs.cooldown_remaining}\n"
        f"budget_remaining: {obs.budget_remaining:.2f} ({obs.budget_remaining:.0%})\n"
        f"active_tool: {obs.active_tool}\n"
        f"progress: {obs.progress_hint:.2f} ({obs.progress_hint:.0%})\n"
        f"step: {obs.step_count}\n\n"
        f"Status: {obs.status_summary}\n"
        f"Context: {obs.decision_context}\n"
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        raw = completion.choices[0].message.content or ""
        return parse_action(raw)
    except Exception as e:
        print(f"[DEBUG] Model request failed: {e}", flush=True)
        return "MODIFY"


# ===== TASK RUNNER =====
async def run_task(task_id: int, seed: int, env: IRCEEnvClient, client: OpenAI) -> None:
    task_name = TASKS[task_id]["name"]
    max_steps = TASKS[task_id]["max_steps"]

    log_start(task=f"task_{task_id}_{task_name}", env="ai_pipeline_recovery", model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    completed = False

    try:
        result = await env.reset(task_id=task_id, seed=seed)
        obs = result.observation

        for step in range(1, max_steps + 1):
            if result.done:
                break

            action_str = get_action(client, obs, step)

            result = await env.step(IRCEAction(action_type=action_str))
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step

            log_step(step, action_str, reward, done, None)

            if done:
                # Check completion: progress >= 1.0 means task fully solved
                completed = obs.progress_hint >= 1.0
                break

        # Normalize: clamp sum of rewards to [0, 1]
        raw_score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(raw_score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)
        score = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ===== MAIN =====
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if IMAGE_NAME:
        env = await IRCEEnvClient.from_docker_image(IMAGE_NAME)
    else:
        env = IRCEEnvClient(base_url="http://localhost:7860")

    try:
        for task_id in sorted(TASKS):
            await run_task(task_id, 42, env, client)
    except Exception as e:
        print(f"[DEBUG] main error: {e}", flush=True)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())