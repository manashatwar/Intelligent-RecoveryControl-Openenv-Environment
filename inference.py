"""
inference.py — AI Pipeline Recovery openENV agent

Communicates with the env server over HTTP (OpenEnv standard API):
  POST {ENV_BASE_URL}/api/reset  -> observation dict
  POST {ENV_BASE_URL}/api/step   -> observation dict

External dependencies (from requirements.txt only):
  openai, python-dotenv, requests
"""

import argparse
import os
import re
import sys
from typing import Optional, List

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ===== ENV CONFIG =====
# ENV_BASE_URL: URL of the running env container (e.g. http://localhost:7860)
# API_BASE_URL: LLM provider base URL
# MODEL_NAME:   model to call
# API_KEY:      HF_TOKEN or OPENAI_API_KEY

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "").rstrip("/")
API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")

TASK_IDS = [1, 2, 3]
TASK_MAX_STEPS = {1: 6, 2: 6, 3: 7}

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


# ===== HTTP ENV CLIENT =====
def env_reset(task_id: int, seed: int) -> dict:
    """Call POST /api/reset on the env container."""
    url = f"{ENV_BASE_URL}/api/reset"
    resp = requests.post(url, json={"task_id": task_id, "seed": seed}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action: str) -> dict:
    """Call POST /api/step on the env container."""
    url = f"{ENV_BASE_URL}/api/step"
    resp = requests.post(url, json={"action_type": action}, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ===== ACTION PARSER =====
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
def get_action(client: OpenAI, obs: dict, step: int) -> str:
    progress = obs.get("progress_hint", 0.0)
    budget = obs.get("budget_remaining", 1.0)
    progress_pct = f"{progress:.0%}"
    budget_pct = f"{budget:.0%}"

    prompt = (
        f"Step: {step}\n"
        f"Goal: {obs.get('goal', '')}\n\n"
        f"tool_result: {obs.get('tool_result', 'unknown')}\n"
        f"error_type: {obs.get('error_type', 'unknown')}\n"
        f"same_error_count: {obs.get('same_error_count', 0)}\n"
        f"cooldown_remaining: {obs.get('cooldown_remaining', 0)}\n"
        f"budget_remaining: {budget:.2f} ({budget_pct})\n"
        f"active_tool: {obs.get('active_tool', 'unknown')}\n"
        f"progress: {progress:.2f} ({progress_pct})\n"
        f"step: {obs.get('step_count', step)}\n\n"
        f"{obs.get('status_summary', '')}\n"
        f"{obs.get('decision_context', '')}\n"
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
        print(f"[WARNING] API or parsing error at step {step}: {e}", flush=True)
        return "MODIFY"


# ===== TASK RUNNER =====
def run_task(task_id: int, seed: int, client: OpenAI) -> None:
    task_name = f"task_{task_id}"
    max_steps = TASK_MAX_STEPS.get(task_id, 6)

    rewards: List[float] = []
    steps = 0
    score = 0.0

    log_start(task=task_name, env="openENV", model=MODEL_NAME)

    try:
        obs = env_reset(task_id=task_id, seed=seed)

        for step in range(1, max_steps + 1):
            if obs.get("done", False):
                break

            action_str = get_action(client, obs, step)

            try:
                obs = env_step(action_str)
            except Exception as e:
                print(f"[WARNING] env step error at step {step}: {e}", flush=True)
                break

            reward = float(obs.get("reward") or 0.0)
            done = bool(obs.get("done", False))

            rewards.append(reward)
            steps = step

            log_step(
                step,
                action_str,
                reward,
                done,
                obs.get("error_type"),
                obs.get("tool_result"),
            )

            if done:
                break

        success = bool(obs.get("done", False))
        # Score from response if available, else compute from rewards
        score = float(obs.get("score") or (sum(rewards) / max(len(rewards), 1)))
        log_end(success=success, steps=steps, rewards=rewards, score=score)

    except Exception as e:
        print(f"[ERROR] Task {task_id} failed: {e}", flush=True)
        log_end(success=False, steps=steps, rewards=rewards, score=0.0)


# ===== ENTRY POINT =====
def main() -> None:
    try:
        parser = argparse.ArgumentParser(description="AI Pipeline Recovery inference runner")
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args()
        seed = args.seed

        # Validate required env vars
        missing = []
        if not ENV_BASE_URL:
            missing.append("ENV_BASE_URL")
        if not API_BASE_URL:
            missing.append("API_BASE_URL")
        if not MODEL_NAME:
            missing.append("MODEL_NAME")
        if not API_KEY:
            missing.append("HF_TOKEN or OPENAI_API_KEY")

        if missing:
            print(f"[FATAL] Missing required env vars: {', '.join(missing)}", flush=True)
            sys.exit(1)

        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

        for task_id in TASK_IDS:
            run_task(task_id, seed, client)

    except SystemExit:
        raise
    except Exception as e:
        print(f"[FATAL] Unexpected error in main: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()