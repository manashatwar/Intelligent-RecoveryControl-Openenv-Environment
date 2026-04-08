"""
inference.py — AI Pipeline Recovery openENV Environment
Uses the OpenEnv HTTP API (/reset, /step) so no local package imports are needed.
All dependencies (openai, requests, python-dotenv) are in requirements.txt.
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
API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")
MODEL_NAME   = os.getenv("MODEL_NAME", "")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or ""

SUPPORTED_ACTIONS = {"RETRY", "MODIFY", "SWITCH", "REPLAN", "ESCALATE"}

# Task IDs and max steps mirror openenv.yaml
TASKS = {
    1: {"name": "easy",   "max_steps": 6},
    2: {"name": "medium", "max_steps": 6},
    3: {"name": "hard",   "max_steps": 7},
}

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
def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=openENV model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str], tool_result: Optional[str] = None) -> None:
    err    = error       if error       is not None else "null"
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
    """Call POST /reset on the env container."""
    url = f"{API_BASE_URL}/reset"
    resp = requests.post(url, json={"task_id": task_id, "seed": seed}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action_type: str) -> dict:
    """Call POST /step on the env container."""
    url = f"{API_BASE_URL}/step"
    resp = requests.post(url, json={"action_type": action_type}, timeout=30)
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


# ===== RULE-BASED FALLBACK (no LLM needed) =====
def rule_based_action(obs: dict) -> str:
    """Deterministic fallback that mirrors the SYSTEM_PROMPT rules."""
    budget      = float(obs.get("budget_remaining", 1.0))
    progress    = float(obs.get("progress_hint",   0.0))
    same_err    = int(obs.get("same_error_count",  0))
    cooldown    = int(obs.get("cooldown_remaining", 0))
    error_type  = str(obs.get("error_type",  "") or "").upper()
    tool_result = str(obs.get("tool_result", "") or "").upper()

    if budget < 0.15 and progress < 0.5:
        return "ESCALATE"
    if same_err >= 4:
        return "ESCALATE"
    if error_type == "RATE_LIMIT":
        return "SWITCH"
    if error_type == "HARD" and same_err >= 2:
        return "SWITCH"
    if error_type == "HARD" and same_err < 2:
        return "MODIFY"
    if tool_result == "AMBIGUOUS":
        return "REPLAN"
    if error_type == "TRANSIENT" and cooldown == 0:
        return "RETRY"
    if cooldown > 0:
        return "REPLAN"
    return "MODIFY"


# ===== MODEL CALL =====
def get_action(client: OpenAI, obs: dict, step: int) -> str:
    progress_pct = f"{float(obs.get('progress_hint', 0)):.0%}"
    budget_pct   = f"{float(obs.get('budget_remaining', 1)):.0%}"

    prompt = (
        f"Step: {step}\n"
        f"Goal: {obs.get('goal', '')}\n\n"
        f"tool_result: {obs.get('tool_result')}\n"
        f"error_type: {obs.get('error_type')}\n"
        f"same_error_count: {obs.get('same_error_count')}\n"
        f"cooldown_remaining: {obs.get('cooldown_remaining')}\n"
        f"budget_remaining: {float(obs.get('budget_remaining', 1)):.2f} ({budget_pct})\n"
        f"active_tool: {obs.get('active_tool')}\n"
        f"progress: {float(obs.get('progress_hint', 0)):.2f} ({progress_pct})\n"
        f"step: {obs.get('step_count')}\n\n"
        f"{obs.get('status_summary', '')}\n"
        f"{obs.get('decision_context', '')}\n"
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        raw_text = completion.choices[0].message.content or ""
        return parse_action(raw_text)
    except Exception as e:
        print(f"[WARNING] LLM error at step {step}: {e} — using rule-based fallback", flush=True)
        return rule_based_action(obs)


# ===== TASK RUNNER =====
def run_task(task_id: int, seed: int, client: Optional[OpenAI]) -> None:
    task_name = TASKS[task_id]["name"]
    max_steps = TASKS[task_id]["max_steps"]
    rewards: List[float] = []
    steps = 0

    log_start(task=f"task_{task_id}_{task_name}", model=MODEL_NAME or "rule-based")

    try:
        obs = env_reset(task_id=task_id, seed=seed)

        for step in range(1, max_steps + 1):
            if obs.get("done"):
                break

            if client:
                action_str = get_action(client, obs, step)
            else:
                action_str = rule_based_action(obs)

            obs = env_step(action_str)

            reward = float(obs.get("reward") or 0.0)
            done   = bool(obs.get("done"))

            rewards.append(reward)
            steps = step

            log_step(step, action_str, reward, done,
                     obs.get("error_type"), obs.get("tool_result"))

            if done:
                break

        score   = float(obs.get("score") or sum(rewards))
        success = bool(obs.get("done"))
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

        if not API_BASE_URL:
            print("[FATAL] API_BASE_URL env var is not set.", flush=True)
            sys.exit(1)

        # Use LLM if credentials are present, otherwise fall back to rule-based
        if MODEL_NAME and API_KEY:
            client: Optional[OpenAI] = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
            print(f"[INFO] Using LLM: {MODEL_NAME}", flush=True)
        else:
            client = None
            print("[INFO] No LLM credentials — running rule-based policy.", flush=True)

        for task_id in sorted(TASKS):
            run_task(task_id, seed, client)

    except SystemExit:
        raise
    except Exception as e:
        print(f"[FATAL] Unexpected error in main: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()