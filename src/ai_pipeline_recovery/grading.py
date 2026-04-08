from __future__ import annotations

from typing import Any


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def grade_completion(log: list[dict[str, Any]]) -> float:
    if not log:
        return 0.0

    final_step = log[-1]

    if final_step.get("completed", False):
        steps = len(log)
        bad_retries = sum(1 for step in log if step.get("bad_retry"))

        # Penalize messy completion slightly
        penalty = min(0.2, 0.04 * bad_retries)

        return max(0.8, 1.0 - penalty)

    if final_step.get("action") == "ESCALATE" and final_step.get("progress", 0.0) >= 0.5:
        return 0.4

    return 0.0


def grade_efficiency(log: list[dict[str, Any]]) -> float:
    if not log:
        return 0.0

    max_steps = max(1, int(log[-1].get("max_steps", len(log))))
    steps = max(1, int(log[-1].get("step_count", len(log))))
    return _clamp01(1.0 - ((steps - 1) / max_steps))


def grade_cost(log: list[dict[str, Any]]) -> float:
    if not log:
        return 0.0

    return _clamp01(log[-1].get("budget_remaining", 0.0))


def grade_recovery_quality(log: list[dict[str, Any]]) -> float:
    if not log:
        return 0.0

    steps = max(1, len(log))
    bad_retries = sum(1 for step in log if step.get("bad_retry"))
    ambiguous_steps = sum(1 for step in log if step.get("tool_result") == "AMBIGUOUS")
    resolved_ambiguity = sum(1 for step in log if step.get("resolved_ambiguity"))
    exhausted_budget = 1 if log[-1].get("budget_remaining", 0.0) <= 0.0 else 0
    controlled_escalation = (
        1
        if log[-1].get("action") == "ESCALATE" and log[-1].get("budget_remaining", 0.0) > 0.0
        else 0
    )

    ambiguity_score = 1.0
    if ambiguous_steps:
        ambiguity_score = resolved_ambiguity / ambiguous_steps

    quality = (
        1.0
        - 0.45 * (bad_retries / steps)
        - 0.2 * exhausted_budget
        + 0.15 * ambiguity_score
        + 0.05 * controlled_escalation
    )
    return _clamp01(quality)


def grade_episode(log: list[dict[str, Any]]) -> float:
    completion = grade_completion(log)
    efficiency = grade_efficiency(log)
    cost = grade_cost(log)
    recovery_quality = grade_recovery_quality(log)
    return _clamp01(
        0.45 * completion
        + 0.25 * efficiency
        + 0.15 * cost
        + 0.15 * recovery_quality
    )
