"""
Organizer-compliant inference entry point for IRCE (AI Pipeline Recovery).

Runs the environment locally in standalone mode — no Docker, no HTTP server,
no async. Mirrors the FoodCrisisEnv pattern that passed Phase 2.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── 1. Standalone mode: must be set BEFORE importing env/models ──────────────
os.environ.setdefault("IRCE_STANDALONE", "1")

# ── 2. Ensure src/ is on the path ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv

try:
    from ai_pipeline_recovery.environment import IRCEEnv
    from ai_pipeline_recovery.grading import grade_episode
    from ai_pipeline_recovery.models import IRCEAction, IRCEObservation
    from ai_pipeline_recovery.tasks import build_task_registry
except ImportError:
    # Fallback for flat-layout (no src/ prefix)
    from environment import IRCEEnv          # type: ignore[no-redef]
    from grading import grade_episode        # type: ignore[no-redef]
    from models import IRCEAction, IRCEObservation  # type: ignore[no-redef]
    from tasks import build_task_registry    # type: ignore[no-redef]

# ── 3. Config ─────────────────────────────────────────────────────────────────
BENCHMARK          = "ai_pipeline_recovery"
DEFAULT_API_BASE   = "https://router.huggingface.co/v1"
DEFAULT_MODEL      = "meta-llama/Llama-3.1-8B-Instruct:novita"
SUPPORTED_ACTIONS  = {"RETRY", "MODIFY", "SWITCH", "REPLAN", "ESCALATE"}
MAX_TOTAL_REWARD   = 1.5   # normaliser so score stays in [0, 1]
SUCCESS_THRESHOLD  = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))

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


# ── 4. Helpers ────────────────────────────────────────────────────────────────

def clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


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


def build_prompt(obs: IRCEObservation, step: int) -> str:
    return (
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


# ── 5. Deterministic fallback (runs even with no API key) ────────────────────

def deterministic_fallback(obs: IRCEObservation) -> str:
    """Rule-based policy that mirrors the SYSTEM_PROMPT priority order exactly.
    Used when no API key is set, or when the LLM call fails for any reason."""
    budget = obs.budget_remaining
    progress = obs.progress_hint
    error = obs.error_type
    same = obs.same_error_count
    cooldown = obs.cooldown_remaining
    result = obs.tool_result

    # Rule 1
    if budget < 0.15 and progress < 0.50:
        return "ESCALATE"
    # Rule 2
    if same >= 4:
        return "ESCALATE"
    # Rule 3
    if error == "RATE_LIMIT":
        return "SWITCH"
    # Rule 4
    if error == "HARD" and same >= 2:
        return "SWITCH"
    # Rule 5
    if error == "HARD" and same < 2:
        return "MODIFY"
    # Rule 6
    if result == "AMBIGUOUS":
        return "REPLAN"
    # Rule 7
    if error == "TRANSIENT" and cooldown == 0:
        return "RETRY"
    # Rule 8
    if cooldown > 0:
        return "REPLAN"
    # Rule 9 — default
    return "MODIFY"


# ── 6. LLM call (falls back to deterministic on any failure) ─────────────────

def get_action(client: Any, model: str, obs: IRCEObservation, step: int) -> str:
    # No client means no API key was configured — use rule engine directly
    if client is None:
        action = deterministic_fallback(obs)
        print(f"[DEBUG] no_client deterministic={action}", file=sys.stderr, flush=True)
        return action

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(obs, step)},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        raw = completion.choices[0].message.content or ""
        return parse_action(raw)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc} — using deterministic fallback", file=sys.stderr, flush=True)
        return deterministic_fallback(obs)


# ── 6. Logging ────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f'[STEP] step={step} action="{action}" reward={reward:.2f} '
        f'done={str(done).lower()} error={error or "null"}',
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── 7. Episode runner ─────────────────────────────────────────────────────────

def run_task(task_id: int, seed: int, client: Any, model: str) -> None:
    from ai_pipeline_recovery.tasks import get_task_config  # local import to avoid early-import issues
    try:
        cfg = get_task_config(task_id)
    except Exception:
        from tasks import get_task_config as _get  # type: ignore[no-redef]
        cfg = _get(task_id)

    log_start(task=f"task_{task_id}_{cfg.name}", env=BENCHMARK, model=model)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    env = IRCEEnv(task_id=task_id, seed=seed)

    try:
        obs: IRCEObservation = env.reset(seed=seed, task_id=task_id)
        done = obs.done

        for step in range(1, cfg.max_steps + 2):   # +1 buffer over task limit
            if done:
                break

            action_str = get_action(client, model, obs, step)

            obs = env.step(IRCEAction(action_type=action_str))
            reward = float(obs.reward)
            done   = bool(obs.done)
            error  = "action_error" if obs.last_action_error else None

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score   = clamp01(grade_episode(env.episode_log))
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} error: {exc}", file=sys.stderr, flush=True)
        score   = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── 8. Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()

    api_key    = (os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "").strip()
    base_url   = (os.getenv("API_BASE_URL") or DEFAULT_API_BASE).strip()
    model_name = (os.getenv("MODEL_NAME") or DEFAULT_MODEL).strip()

    client: Any
    if not api_key:
        print("[DEBUG] No API key — using built-in deterministic fallback.", file=sys.stderr, flush=True)
        client = None
    else:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url)
        except Exception as exc:
            print(f"[DEBUG] OpenAI client init failed: {exc} — deterministic fallback.", file=sys.stderr, flush=True)
            client = None

    task_registry = build_task_registry()
    for task_id in sorted(task_registry):
        try:
            run_task(task_id=task_id, seed=42, client=client, model=model_name)
        except Exception as exc:
            print(f"[DEBUG] run_task({task_id}) unhandled: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()