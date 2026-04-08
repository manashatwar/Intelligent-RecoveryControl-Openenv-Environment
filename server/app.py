"""
IRCE Playground Server
----------------------
Custom FastAPI application that serves the interactive Playground UI
and exposes API endpoints for manual stepping and LLM simulation.
"""
from __future__ import annotations

import os

# Standalone mode – previously avoided hard openenv dependency.
# Now we use openenv_core so it passes the grader.
# os.environ.setdefault("IRCE_STANDALONE", "1")

import sys
import json
import uuid
import asyncio
import re
import random
from pathlib import Path
from typing import Optional, Any, List

import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_pipeline_recovery.environment import IRCEEnv
from ai_pipeline_recovery.models import IRCEAction, IRCEObservation
from ai_pipeline_recovery.tasks import build_task_registry, get_task_config
from ai_pipeline_recovery.grading import (
    grade_episode,
    grade_completion,
    grade_efficiency,
    grade_cost,
    grade_recovery_quality,
)

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# ── FastAPI app ──────────────────────────────────────────────────────────────

def _create_env():
    return IRCEEnv()

try:
    from openenv.core.env_server import create_fastapi_app
    app = create_fastapi_app(_create_env, IRCEAction, IRCEObservation)
except ImportError:
    try:
        from openenv_core.env_server import create_fastapi_app
        app = create_fastapi_app(_create_env, IRCEAction, IRCEObservation)
    except ImportError:
        app = FastAPI(title="IRCE – Intelligent Recovery Control Environment")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session storage ─────────────────────────────────────────────────────────

sessions: dict[str, IRCEEnv] = {}

# ── Request / Response models ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: int = 1
    seed: int = 42


class StepRequest(BaseModel):
    session_id: str
    action_type: str = "RETRY"


# ── Helpers ─────────────────────────────────────────────────────────────────

def _task_config_dict(cfg) -> dict:
    return {
        "task_id": cfg.task_id,
        "name": cfg.name,
        "description": cfg.description,
        "goal": cfg.goal,
        "initial_budget": cfg.initial_budget,
        "max_steps": cfg.max_steps,
        "noise_level": cfg.noise_level,
        "ambiguity_rate": cfg.ambiguity_rate,
        "allow_rate_limit": cfg.allow_rate_limit,
        "rate_limit_cooldown": cfg.rate_limit_cooldown,
        "drift_enabled": cfg.drift_enabled,
        "drift_after_step": cfg.drift_after_step,
        "cascade_penalty": cfg.cascade_penalty,
        "initial_errors": list(cfg.initial_errors),
    }


def _obs_dict(obs: IRCEObservation) -> dict:
    return {
        "goal": obs.goal,
        "tool_result": obs.tool_result,
        "error_type": obs.error_type,
        "same_error_count": obs.same_error_count,
        "budget_remaining": obs.budget_remaining,
        "step_count": obs.step_count,
        "last_action_error": obs.last_action_error,
        "active_tool": obs.active_tool,
        "cooldown_remaining": obs.cooldown_remaining,
        "progress_hint": obs.progress_hint,
        "history_tail": obs.history_tail,
        "status_summary": obs.status_summary,
        "decision_context": obs.decision_context,
        "reward": obs.reward,
        "done": obs.done,
    }


def _grade_dict(log: list) -> dict:
    return {
        "total": round(grade_episode(log), 4),
        "completion": round(grade_completion(log), 4),
        "efficiency": round(grade_efficiency(log), 4),
        "cost": round(grade_cost(log), 4),
        "recovery_quality": round(grade_recovery_quality(log), 4),
    }


# ── Baseline policy (mirrors inference.py system prompt rules) ──────────────

def baseline_policy(obs: IRCEObservation) -> str:
    """Deterministic rule-based recovery policy."""
    if obs.budget_remaining < 0.15 and obs.progress_hint < 0.5:
        return "ESCALATE"
    if obs.same_error_count >= 4:
        return "ESCALATE"
    if obs.error_type == "RATE_LIMIT":
        return "SWITCH"
    if obs.error_type == "HARD" and obs.same_error_count >= 2:
        return "SWITCH"
    if obs.error_type == "HARD" and obs.same_error_count < 2:
        return "MODIFY"
    if obs.tool_result == "AMBIGUOUS":
        return "REPLAN"
    if obs.error_type == "TRANSIENT" and obs.cooldown_remaining == 0:
        return "RETRY"
    if obs.cooldown_remaining > 0:
        return "REPLAN"
    return "MODIFY"


# ── LLM action (optional, uses OpenAI-compatible endpoint) ──────────────────

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

Do NOT explain. Output ONLY one word."""

SUPPORTED_ACTIONS = {"RETRY", "MODIFY", "SWITCH", "REPLAN", "ESCALATE"}


def _parse_action(text: str) -> str:
    if not text or not text.strip():
        raise ValueError("Empty")
    cleaned = text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r"[`*_#>\-]", "", cleaned).strip()
    cleaned = cleaned.upper().replace(".", "").replace(",", "").strip()
    first_line = cleaned.split("\n")[0].strip()
    if first_line in SUPPORTED_ACTIONS:
        return first_line
    for a in SUPPORTED_ACTIONS:
        if re.search(rf"\b{a}\b", first_line):
            return a
    for a in SUPPORTED_ACTIONS:
        if re.search(rf"\b{a}\b", cleaned):
            return a
    raise ValueError(f"Cannot parse: {text[:50]}")


async def _get_llm_action(obs: IRCEObservation, step: int) -> str:
    """Call an OpenAI-compatible endpoint. Falls back to baseline on error."""
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("API_BASE_URL")
    model = os.getenv("MODEL_NAME")

    if not api_key or not base_url or not model:
        return baseline_policy(obs)

    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)
        prompt = (
            f"Step: {step}\n"
            f"Goal: {obs.goal}\n\n"
            f"tool_result: {obs.tool_result}\n"
            f"error_type: {obs.error_type}\n"
            f"same_error_count: {obs.same_error_count}\n"
            f"cooldown_remaining: {obs.cooldown_remaining}\n"
            f"budget_remaining: {obs.budget_remaining:.2f}\n"
            f"active_tool: {obs.active_tool}\n"
            f"progress: {obs.progress_hint:.2f}\n"
            f"step: {obs.step_count}\n\n"
            f"{obs.status_summary}\n"
            f"{obs.decision_context}\n"
        )

        loop = asyncio.get_event_loop()
        completion = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=10,
            ),
        )
        raw = completion.choices[0].message.content or ""
        return _parse_action(raw)
    except Exception:
        return baseline_policy(obs)


# ── API Routes ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "ai_pipeline_recovery", "version": "0.1.0"}


@app.get("/api/tasks")
async def api_get_tasks():
    registry = build_task_registry()
    return {
        str(tid): _task_config_dict(cfg)
        for tid, cfg in registry.items()
    }


@app.get("/api/readme")
async def api_get_readme():
    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        # Strip YAML front-matter
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                text = parts[2].strip()
        return {"content": text}
    return {"content": "# README not found"}


@app.post("/api/reset")
async def api_reset(req: ResetRequest):
    env = IRCEEnv(task_id=req.task_id, seed=req.seed)
    obs = env.reset(seed=req.seed, task_id=req.task_id)
    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = env

    # Limit max sessions in memory
    if len(sessions) > 50:
        oldest = list(sessions.keys())[0]
        del sessions[oldest]

    return {
        "session_id": session_id,
        "observation": _obs_dict(obs),
        "task_config": _task_config_dict(env.task_config),
    }


@app.post("/api/step")
async def api_step(req: StepRequest):
    env = sessions.get(req.session_id)
    if not env:
        return JSONResponse(
            status_code=404,
            content={"error": "Session not found. Please reset first."},
        )

    action = IRCEAction(action_type=req.action_type)
    obs = env.step(action)

    result: dict[str, Any] = {
        "observation": _obs_dict(obs),
        "episode_log": env.episode_log,
    }

    if obs.done:
        result["grade"] = _grade_dict(env.episode_log)

    return result


@app.get("/api/state/{session_id}")
async def api_get_state(session_id: str):
    env = sessions.get(session_id)
    if not env:
        return JSONResponse(status_code=404, content={"error": "Session not found."})
    return {
        "state": {
            "task_id": env.task_id,
            "task_name": env.task_config.name,
            "step_count": env._state.step_count,
            "budget_remaining": round(env._state.budget_remaining, 2),
            "progress": round(env._state.progress, 2),
            "tool_state": env._state.tool_state,
            "current_error_type": env._state.current_error_type,
            "cooldown_remaining": env._state.cooldown_remaining,
        },
        "episode_log": env.episode_log,
    }


# ── SSE streaming for simulation mode ──────────────────────────────────────

@app.get("/api/llm/stream")
async def api_llm_stream(
    task_ids: str = Query("1,2,3"),
    seed: int = Query(42),
    use_llm: bool = Query(False),
):
    """Run simulation across one or more tasks, streaming step data via SSE."""

    parsed_ids = [int(t.strip()) for t in task_ids.split(",") if t.strip()]

    async def event_generator():
        all_results = []

        for task_id in parsed_ids:
            try:
                config = get_task_config(task_id)
            except ValueError:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Unknown task {task_id}'})}\n\n"
                continue

            env = IRCEEnv(task_id=task_id, seed=seed)
            obs = env.reset(seed=seed, task_id=task_id)

            yield f"data: {json.dumps({'type': 'task_start', 'task_id': task_id, 'task_name': config.name, 'goal': config.goal, 'max_steps': config.max_steps, 'initial_budget': config.initial_budget})}\n\n"
            await asyncio.sleep(0.3)

            for step in range(1, config.max_steps + 1):
                if obs.done:
                    break

                if use_llm:
                    action_str = await _get_llm_action(obs, step)
                else:
                    action_str = baseline_policy(obs)

                obs = env.step(IRCEAction(action_type=action_str))

                step_data = {
                    "type": "step",
                    "task_id": task_id,
                    "task_name": config.name,
                    "step": step,
                    "max_steps": config.max_steps,
                    "action": action_str,
                    "observation": _obs_dict(obs),
                    "log_entry": env.episode_log[-1] if env.episode_log else {},
                }
                yield f"data: {json.dumps(step_data)}\n\n"
                await asyncio.sleep(0.6)

                if obs.done:
                    break

            grade = _grade_dict(env.episode_log)
            task_result = {
                "type": "task_complete",
                "task_id": task_id,
                "task_name": config.name,
                "grade": grade,
                "episode_log": env.episode_log,
                "steps": len(env.episode_log),
            }
            all_results.append(task_result)
            yield f"data: {json.dumps(task_result)}\n\n"
            await asyncio.sleep(0.5)

        yield f"data: {json.dumps({'type': 'all_complete', 'results': [{'task_id': r['task_id'], 'task_name': r['task_name'], 'grade': r['grade'], 'steps': r['steps']} for r in all_results]})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Static files & UI ──────────────────────────────────────────────────────

STATIC_DIR = ROOT / "static"

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file), media_type="text/html")
    return HTMLResponse("<h1>IRCE</h1><p>Static files not found. Create static/index.html.</p>")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Entry point ─────────────────────────────────────────────────────────────

__all__ = ["app", "main"]


def main() -> None:
    uvicorn.run(
        "server.app:app",
        host=os.getenv("IRCE_HOST", "0.0.0.0"),
        port=int(os.getenv("IRCE_PORT", "7860")),
        reload=True,
    )


if __name__ == "__main__":
    main()
