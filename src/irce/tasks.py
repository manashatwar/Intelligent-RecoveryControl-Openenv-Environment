from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TaskConfig:
    task_id: int
    name: str
    description: str
    goal: str
    initial_budget: float
    max_steps: int
    noise_level: float
    ambiguity_rate: float
    allow_rate_limit: bool
    rate_limit_cooldown: int
    drift_after_step: int | None
    drift_enabled: bool
    cascade_penalty: float
    initial_errors: tuple[str, ...]
    retry_success: float
    modify_success: float
    switch_success: float
    replan_success: float
    backup_success_bonus: float
    backup_budget_cost: float
    ambiguity_progress: float


def build_task_registry() -> dict[int, TaskConfig]:
    return {
        1: TaskConfig(
            task_id=1,
            name="easy",
            description="Single-workflow recovery with transient and hard failures, generous budget, and clean signals.",
            goal="Recover a stalled production workflow with minimal wasted attempts.",
            initial_budget=1.0,
            max_steps=6,
            noise_level=0.0,
            ambiguity_rate=0.05,
            allow_rate_limit=False,
            rate_limit_cooldown=0,
            drift_after_step=None,
            drift_enabled=False,
            cascade_penalty=0.0,
            initial_errors=("TRANSIENT", "HARD"),
            retry_success=0.68,
            modify_success=0.76,
            switch_success=0.52,
            replan_success=0.45,
            backup_success_bonus=0.12,
            backup_budget_cost=0.03,
            ambiguity_progress=0.15,
        ),
        2: TaskConfig(
            task_id=2,
            name="medium",
            description="Two-tool recovery with rate limits, moderate ambiguity, and noticeable budget pressure.",
            goal="Recover a mixed-failure tool workflow without wasting API budget.",
            initial_budget=0.8,
            max_steps=6,
            noise_level=0.12,
            ambiguity_rate=0.12,
            allow_rate_limit=True,
            rate_limit_cooldown=1,
            drift_after_step=None,
            drift_enabled=False,
            cascade_penalty=0.04,
            initial_errors=("TRANSIENT", "HARD", "RATE_LIMIT"),
            retry_success=0.58,
            modify_success=0.66,
            switch_success=0.64,
            replan_success=0.45,
            backup_success_bonus=0.16,
            backup_budget_cost=0.06,
            ambiguity_progress=0.18,
        ),
        3: TaskConfig(
            task_id=3,
            name="hard",
            description="Drifting reliability, noisy error labels, cooldown pressure, and costly backup routing.",
            goal="Recover an unstable workflow under drift, ambiguity, and tight budget limits.",
            initial_budget=0.6,
            max_steps=7,
            noise_level=0.25,
            ambiguity_rate=0.2,
            allow_rate_limit=True,
            rate_limit_cooldown=2,
            drift_after_step=2,
            drift_enabled=True,
            cascade_penalty=0.08,
            initial_errors=("TRANSIENT", "HARD", "RATE_LIMIT"),
            retry_success=0.5,
            modify_success=0.62,
            switch_success=0.62,
            replan_success=0.56,
            backup_success_bonus=0.18,
            backup_budget_cost=0.08,
            ambiguity_progress=0.2,
        ),
    }


TASKS = build_task_registry()


def get_task_config(task_id: int = 1) -> TaskConfig:
    try:
        return TASKS[int(task_id)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported task_id: {task_id}") from exc
