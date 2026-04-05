<<<<<<< HEAD
import argparse
=======
"""
inference.py
------------
Hackathon entry point for the IRCE OpenEnv benchmark.

Runs the IRCE environment across all three tasks (easy / medium / hard) and
prints per-task and average scores. Must be run from the project root.

Inference flow
--------------
1. Load API credentials from .env (HF_TOKEN, API_BASE_URL, MODEL_NAME).
2. Build an OpenAI-compatible client pointing at the configured LLM API.
3. For each task (1 → 2 → 3):
   a. Reset the environment.
   b. At every step, build a chat prompt (system + state summary).
   c. Call the LLM via client.chat.completions.create().
   d. Parse the one-word action from the response.
   e. Step the environment; repeat until done.
4. Print scores and average.

Fallback policy
---------------
If the LLM API is unavailable (no API key, network error, etc.) the agent
falls back to a deterministic rule-based policy (select_fallback_action)
that achieves ~0.86 average without any API calls — ensuring the script
always produces a valid score.

Usage
-----
    python inference.py                           # API mode (uses .env)
    python inference.py --seed 42                 # different eval seed
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
>>>>>>> 9b34442430b98e9efb51d76a40f653bc7bde7b4d
import os
import re
import sys
from typing import Optional, List

from dotenv import load_dotenv
from openai import OpenAI

from irce.environment import IRCEEnv
from irce.models import IRCEAction, IRCEObservation
from irce.tasks import build_task_registry
from irce.grading import grade_episode

<<<<<<< HEAD
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

# Priority-ordered decision rules the model must follow top-to-bottom.
# Each rule is a concrete condition → action mapping. REPLAN is NOT a default.
SYSTEM_PROMPT = """You are an AI workflow recovery agent for the IRCE benchmark.

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

Do NOT explain. Do NOT add punctuation or extra text. Output ONLY the single action word."""
=======
SUPPORTED_ACTIONS = ("RETRY", "MODIFY", "SWITCH", "REPLAN", "ESCALATE")
SYSTEM_PROMPT = (
    "You are controlling a recovery policy for IRCE. "
    "Choose the single best recovery action based on the current error_type and context.\n\n"
    "Decision rules (follow in order):\n"
    "  1. error=HARD        → MODIFY  (never RETRY on HARD — it always fails)\n"
    "  2. error=RATE_LIMIT  → SWITCH  (cooldown — switching tool path clears it)\n"
    "  3. repeat_errors>=2  → SWITCH  (same error twice means path is stuck)\n"
    "  4. result=AMBIGUOUS  → REPLAN  (partial signal — reframe before next attempt)\n"
    "  5. error=TRANSIENT   → RETRY   (transient failures are safe to retry once)\n"
    "  6. budget<0.15       → ESCALATE (too little left to keep trying)\n\n"
    "Respond with EXACTLY one word: RETRY, MODIFY, SWITCH, REPLAN, or ESCALATE\n"
    "No explanation. One word only."
)
ObservationT = TypeVar("ObservationT")
>>>>>>> 9b34442430b98e9efb51d76a40f653bc7bde7b4d


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


<<<<<<< HEAD
def log_end(success: bool, steps: int, rewards: List[float], score: float) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"rewards={rewards_str} score={score:.3f}",
        flush=True,
    )
=======
def build_openai_client() -> tuple[Any | None, str, str]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or ""
    base_url = os.getenv("API_BASE_URL", "").strip()
    model_name_env = os.getenv("MODEL_NAME", "").strip()

    if not api_key or not model_name_env:
        return None, model_name_env, "OPENAI_API_KEY/HF_TOKEN or MODEL_NAME missing"

    try:
        from openai import OpenAI
    except Exception as exc:  # noqa: BLE001
        return None, model_name_env, f"OpenAI SDK import failed: {exc}"

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    return OpenAI(**client_kwargs), model_name_env, ""
>>>>>>> 9b34442430b98e9efb51d76a40f653bc7bde7b4d


# ===== PARSER =====
def parse_action(text: str) -> str:
    """Parse LLM output into a supported action. Raises ValueError if unparsable."""
    if not text or not text.strip():
        raise ValueError("Model returned empty output — cannot parse action")

    cleaned = text.strip()

    # Strip chain-of-thought / reasoning tokens (Qwen, DeepSeek style)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    # Strip markdown noise
    cleaned = re.sub(r"[`*_#>\-]", "", cleaned).strip()

    # Normalise: uppercase, remove trailing punctuation
    cleaned = cleaned.upper().replace(".", "").replace(",", "").strip()

    # Try the first line only (model may add an explanation after)
    first_line = cleaned.split("\n")[0].strip()

    if first_line in SUPPORTED_ACTIONS:
        return first_line

    for action in SUPPORTED_ACTIONS:
        if re.search(rf"\b{action}\b", first_line):
            return action

    # Full-text fallback
    for action in SUPPORTED_ACTIONS:
        if re.search(rf"\b{action}\b", cleaned):
            return action

    raise ValueError(
        f"Could not parse a valid action from model output: '{text.strip()[:100]}'"
    )


# ===== MODEL CALL =====
def get_action(client: OpenAI, observation: IRCEObservation, step: int) -> str:
    """Call the LLM and return a parsed action. Raises on API or parse errors."""
    progress_pct = f"{observation.progress_hint:.0%}"
    budget_pct = f"{observation.budget_remaining:.0%}"

    prompt = (
        f"Step: {step}\n"
        f"Goal: {observation.goal}\n\n"
        f"=== Observation ===\n"
        f"tool_result:       {observation.tool_result}\n"
        f"error_type:        {observation.error_type}\n"
        f"same_error_count:  {observation.same_error_count}\n"
        f"cooldown_remaining:{observation.cooldown_remaining}\n"
        f"budget_remaining:  {observation.budget_remaining:.2f} ({budget_pct})\n"
        f"active_tool:       {observation.active_tool}\n"
        f"progress:          {observation.progress_hint:.2f} ({progress_pct})\n"
        f"step:              {observation.step_count}\n\n"
        f"=== Context ===\n"
        f"Status:      {observation.status_summary}\n"
        f"Constraints: {observation.decision_context}\n\n"
        f"Apply the priority rules and output EXACTLY one action word."
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=20,
    )

    raw_text = completion.choices[0].message.content or ""
    return parse_action(raw_text)


# ===== TASK RUNNER =====
def run_task(task_id: int, seed: int, client: OpenAI) -> None:
    env = IRCEEnv(task_id=task_id, seed=seed)
    task_config = build_task_registry()[task_id]
    task_name = f"task_{task_id}"

    rewards: List[float] = []
    steps = 0

    log_start(task=task_name, env="IRCE", model=MODEL_NAME)

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


<<<<<<< HEAD
# ===== ENTRY POINT =====
def main() -> None:
    parser = argparse.ArgumentParser(description="IRCE inference runner")
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for the environment (default: 7)",
    )
    args = parser.parse_args()
    seed: int = args.seed

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"[INFO] Connecting to {API_BASE_URL} with model {MODEL_NAME}...", flush=True)
    print(f"[INFO] Using seed={seed}", flush=True)
    try:
        test = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        test_response = (test.choices[0].message.content or "").strip()
        print(f"[INFO] LLM connectivity verified: '{test_response}'", flush=True)
    except Exception as exc:
        print(f"[FATAL] Cannot connect to LLM: {exc}", flush=True)
        sys.exit(1)

    for task_id in sorted(build_task_registry()):
        run_task(task_id, seed, client)
=======
def _parse_inference_args() -> Any:
    import argparse
    p = argparse.ArgumentParser(description="IRCE inference runner.")

    p.add_argument("--seed", type=int, default=7, help="Evaluation seed (default: 7).")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = _parse_inference_args()

    client, model_name, client_status = build_openai_client()
    if client is None:
        print(f"Client not configured ({client_status}). Using deterministic fallback policy.")
    else:
        print(f"OpenAI client configured for model '{model_name}'.")

    seed = args.seed
    task_registry = build_task_registry()
    scores = [
        run_task(task_id=task_id, seed=seed, client=client, model_name=model_name)
        for task_id in sorted(task_registry)
    ]
    average_score = sum(scores) / len(scores)
    print(f"average score: {average_score:.3f}")
>>>>>>> 9b34442430b98e9efb51d76a40f653bc7bde7b4d


if __name__ == "__main__":
    main()