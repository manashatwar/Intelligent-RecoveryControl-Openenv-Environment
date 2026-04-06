from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    step_cost: float
    progress_bonus: float
    ambiguity_bonus: float
    completion_bonus: float
    repeated_error_penalty: float
    bad_retry_penalty: float
    switch_cost_penalty: float
    escalation_penalty: float
    cascade_penalty: float
    total: float


def compute_step_reward(
    *,
    action_type: str,
    tool_result: str,
    previous_error_type: str,
    same_error_count: int,
    progress_delta: float,
    completed: bool,
    switched_tools: bool,
    backup_budget_cost: float,
    cascade_penalty: float,
    escalated_early: bool,
) -> RewardBreakdown:
    step_cost = -0.1
    progress_bonus = 0.3 if progress_delta >= 0.45 else 0.0
    ambiguity_bonus = 0.15 if tool_result == "AMBIGUOUS" and progress_delta > 0.0 else 0.0
    completion_bonus = 0.9 if completed else 0.0
    repeated_error_penalty = -0.2 if tool_result == "ERROR" and same_error_count > 0 else 0.0
    bad_retry_penalty = (
        -0.3
        if action_type == "RETRY" and previous_error_type in {"HARD", "RATE_LIMIT"}
        else 0.0
    )
    switch_cost_penalty = -0.08 if switched_tools else -0.02 * backup_budget_cost
    escalation_penalty = -0.2 if escalated_early else 0.0
    cascade_penalty_value = -cascade_penalty * min(same_error_count, 2) if tool_result == "ERROR" else 0.0

    total = (
        step_cost
        + progress_bonus
        + ambiguity_bonus
        + completion_bonus
        + repeated_error_penalty
        + bad_retry_penalty
        + switch_cost_penalty
        + escalation_penalty
        + cascade_penalty_value
    )

    return RewardBreakdown(
        step_cost=step_cost,
        progress_bonus=progress_bonus,
        ambiguity_bonus=ambiguity_bonus,
        completion_bonus=completion_bonus,
        repeated_error_penalty=repeated_error_penalty,
        bad_retry_penalty=bad_retry_penalty,
        switch_cost_penalty=switch_cost_penalty,
        escalation_penalty=escalation_penalty,
        cascade_penalty=cascade_penalty_value,
        total=round(total, 3),
    )
