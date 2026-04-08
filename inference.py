import argparse
import os
import re
import sys
import random
from typing import Optional, List

from dotenv import load_dotenv
from openai import OpenAI

from ai_pipeline_recovery.environment import IRCEEnv
from ai_pipeline_recovery.models import IRCEAction, IRCEObservation
from ai_pipeline_recovery.tasks import build_task_registry
from ai_pipeline_recovery.grading import grade_episode

load_dotenv()

# ===== ENV CONFIG =====
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

if not API_BASE_URL or not MODEL_NAME or not API_KEY:
    print(
        "[FATAL] Missing required env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN/OPENAI_API_KEY",
        flush=True,
    )
    sys.exit(1)

SUPPORTED_ACTIONS = {"RETRY", "MODIFY", "SWITCH", "REPLAN", "ESCALATE"}

SYSTEM_PROMPT = """You are an AI workflow recovery agent for the AI Pipeline Recovery openENV Environment benchmark.

Given an observation, output EXACTLY one action word: RETRY, MODIFY, SWITCH, REPLAN, or ESCALATE.

Apply these rules in order — use the FIRST rule whose condition is true:

1. ESCALATE  — if budget_remaining < 0.15 AND progress < 50%
2. ESCALATE  — if same_error_count >= 4
3. SWITCH    — if error_type = RATE_LIMIT
4. SWITCH    — if error_type = HARD AND same_error_count >= 2
5. MODIFY    — if error_type = HARD AND same_error_count < 2
6. REPLAN    — if tool_result = AMBIGUOUS
7. RETRY     — if error_type = TRANSIENT AND cooldown_remaining = 0
8. REPLAN    — if cooldown_remaining > 0
9. MODIFY    — default (when no other rule matches)

Do NOT explain. Output ONLY one word."""

# ===== LOGGING =====
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
    tool_result: Optional[str] = None,
) -> None:
    err = error if error is not None else "null"
    result = tool_result if tool_result is not None else "unknown"

    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err} result={result}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float], score: float) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"rewards={rewards_str} score={score:.3f}",
        flush=True,
    )


# ===== PARSER =====
def parse_action(text: str) -> str:
    if not text or not text.strip():
        raise ValueError("Model returned empty output")

    cleaned = text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r"[`*_#>\-]", "", cleaned).strip()
    cleaned = cleaned.upper().replace(".", "").replace(",", "").strip()

    first_line = cleaned.split("\n")[0].strip()

    if first_line in SUPPORTED_ACTIONS:
        return first_line

    for action in SUPPORTED_ACTIONS:
        if re.search(rf"\b{action}\b", first_line):
            return action

    for action in SUPPORTED_ACTIONS:
        if re.search(rf"\b{action}\b", cleaned):
            return action

    raise ValueError(f"Invalid action: {text[:50]}")


# ===== MODEL CALL =====
def get_action(client: OpenAI, observation: IRCEObservation, step: int) -> str:
    progress_pct = f"{observation.progress_hint:.0%}"
    budget_pct = f"{observation.budget_remaining:.0%}"

    prompt = (
        f"Step: {step}\n"
        f"Goal: {observation.goal}\n\n"
        f"tool_result: {observation.tool_result}\n"
        f"error_type: {observation.error_type}\n"
        f"same_error_count: {observation.same_error_count}\n"
        f"cooldown_remaining: {observation.cooldown_remaining}\n"
        f"budget_remaining: {observation.budget_remaining:.2f} ({budget_pct})\n"
        f"active_tool: {observation.active_tool}\n"
        f"progress: {observation.progress_hint:.2f} ({progress_pct})\n"
        f"step: {observation.step_count}\n\n"
        f"{observation.status_summary}\n"
        f"{observation.decision_context}\n"
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

        raw_text = completion.choices[0].message.content or ""
        return parse_action(raw_text)
    except Exception as e:
        print(f"[WARNING] API or parsing error: {e}", flush=True)
        return "MODIFY"


# ===== TASK RUNNER =====
def run_task(task_id: int, seed: int, client: OpenAI) -> None:
    env = IRCEEnv(task_id=task_id, seed=seed)
    task_config = build_task_registry()[task_id]
    task_name = f"task_{task_id}"

    rewards: List[float] = []
    steps = 0

    log_start(task=task_name, env="openENV", model=MODEL_NAME)

    try:
        obs = env.reset(seed=seed, task_id=task_id)

        for step in range(1, task_config.max_steps + 1):
            if obs.done:
                break

            action_str = get_action(client, obs, step)
            obs = env.step(IRCEAction(action_type=action_str))

            reward = float(obs.reward or 0.0)
            done = bool(obs.done)

            rewards.append(reward)
            steps = step

            log_step(step, action_str, reward, done, obs.error_type, obs.tool_result)

            if done:
                break

        success = bool(obs.done)
        score = grade_episode(env.episode_log)

        log_end(success=success, steps=steps, rewards=rewards, score=score)

    except Exception as e:
        print(f"[ERROR] Task {task_id} failed: {e}", flush=True)
        log_end(success=False, steps=steps, rewards=rewards, score=0.0)


# ===== ENTRY POINT =====
def main() -> None:
    parser = argparse.ArgumentParser(description="AI Pipeline Recovery inference runner")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed = args.seed

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_id in sorted(build_task_registry()):
        run_task(task_id, seed, client)


if __name__ == "__main__":
    main()