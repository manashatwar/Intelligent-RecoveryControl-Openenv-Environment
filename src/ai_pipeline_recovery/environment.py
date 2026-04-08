from __future__ import annotations

import os
import random
from typing import Generic, TypeVar

if os.getenv("IRCE_STANDALONE") == "1":
    ActionT = TypeVar("ActionT")
    ObservationT = TypeVar("ObservationT")
    StateT = TypeVar("StateT")

    class Environment(Generic[ActionT, ObservationT, StateT]):
        """Lightweight Environment base for standalone inference."""

        SUPPORTS_CONCURRENT_SESSIONS = True

        def __init__(self) -> None:
            pass

        def _apply_transform(self, observation: ObservationT) -> ObservationT:
            return observation

        def _reset_rubric(self) -> None:
            return None
else:
    try:
        from openenv.core.env_server import Environment
    except ImportError:  # pragma: no cover
        from openenv_core.env_server import Environment

try:
    from ai_pipeline_recovery.models import IRCEAction, IRCEObservation, IRCEState
    from ai_pipeline_recovery.rewards import compute_step_reward
    from ai_pipeline_recovery.tasks import TaskConfig, get_task_config
except ImportError:  # pragma: no cover
    from models import IRCEAction, IRCEObservation, IRCEState
    from rewards import compute_step_reward
    from tasks import TaskConfig, get_task_config


class IRCEEnv(Environment[IRCEAction, IRCEObservation, IRCEState]):
    """Deterministic OpenEnv benchmark for recovery decisions under uncertainty."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task_id: int = 1, seed: int | None = 7) -> None:
        super().__init__()
        self.task_id = task_id
        self.task_config = get_task_config(task_id)
        self._default_seed = 7 if seed is None else seed
        self._rng = random.Random(self._default_seed)
        self._episode_index = 0
        self.episode_log: list[dict[str, object]] = []
        self._state = IRCEState(episode_id=self._build_episode_id(self._default_seed))

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: object,
    ) -> IRCEObservation:
        requested_task_id = kwargs.pop("task_id", self.task_id)
        self.task_id = int(requested_task_id)
        self.task_config = get_task_config(self.task_id)
        self._reset_rubric()
        episode_seed = self._default_seed if seed is None else seed
        self._rng = random.Random(episode_seed)
        self._episode_index += 1
        self.episode_log = []
        self._state = IRCEState(
            episode_id=episode_id or self._build_episode_id(episode_seed),
            task_name=self.task_config.name,
            goal=self.task_config.goal,
            current_error_type=self._rng.choice(self.task_config.initial_errors),
            same_error_count=0,
            budget_remaining=self.task_config.initial_budget,
            progress=0.0,
            step_count=0,
            tool_state="primary",
            replan_bonus=0.0,
            cooldown_remaining=0,
            ambiguous_count=0,
            last_tool_result="ERROR",
            history=[],
            last_action="RESET",
            last_reward=0.0,
        )
        return self._observation(tool_result="ERROR", last_action_error=False, reward=0.0)

    def step(
        self,
        action: IRCEAction,
        timeout_s: float | None = None,
        **kwargs: object,
    ) -> IRCEObservation:
        del timeout_s, kwargs

        state = self._state
        config = self.task_config
        previous_error_type = state.current_error_type
        previous_outcome = state.last_tool_result
        previous_progress = state.progress
        previous_tool = state.tool_state
        state.step_count += 1

        tool_result = "ERROR"
        last_action_error = True
        action_type = action.action_type
        switched_tools = False
        bad_retry = False
        escalated_early = False

        if not action.is_supported:
            state.same_error_count += 1
        elif action_type == "ESCALATE":
            escalated_early = state.progress < 0.5
            state.budget_remaining = max(0.0, state.budget_remaining - 0.03)
            return self._finalize_step(
                action_type=action_type,
                previous_error_type=previous_error_type,
                tool_result="ERROR",
                hidden_error_type=state.current_error_type,
                last_action_error=False,
                progress_delta=0.0,
                bad_retry=False,
                switched_tools=False,
                escalated_early=escalated_early,
                done=True,
            )
        else:
            if action_type == "SWITCH":
                switched_tools = True
                state.tool_state = "backup" if state.tool_state == "primary" else "primary"
                state.cooldown_remaining = max(0, state.cooldown_remaining - 1)
            elif action_type == "REPLAN":
                state.replan_bonus = min(0.25, state.replan_bonus + 0.12)
                state.cooldown_remaining = max(0, state.cooldown_remaining - 1)
            elif action_type == "MODIFY" and state.cooldown_remaining > 0:
                state.cooldown_remaining = max(0, state.cooldown_remaining - 1)

            success_probability = self._attempt_probability(action_type, previous_error_type)

            if success_probability <= 0.0:
                tool_result = "ERROR"
                last_action_error = True
                state.same_error_count += 1
                bad_retry = action_type == "RETRY" and previous_error_type in {"HARD", "RATE_LIMIT"}
                self._advance_error_on_failure(action_type, previous_error_type)
            else:
                resolved_ambiguity = self._is_ambiguous_outcome(action_type, previous_error_type)
                if self._rng.random() < min(0.95, success_probability):
                    tool_result = "SUCCESS"
                    last_action_error = False
                    state.progress = min(1.0, state.progress + 0.5)
                    state.same_error_count = 0
                    state.ambiguous_count = 0
                    state.cooldown_remaining = max(0, state.cooldown_remaining - 1)
                    state.current_error_type = "TRANSIENT"
                    state.replan_bonus = 0.0
                elif resolved_ambiguity:
                    tool_result = "AMBIGUOUS"
                    last_action_error = False
                    state.progress = min(1.0, state.progress + config.ambiguity_progress)
                    state.same_error_count = 0
                    state.ambiguous_count += 1
                    state.cooldown_remaining = max(0, state.cooldown_remaining - 1)
                    if previous_error_type == "HARD" and action_type in {"MODIFY", "REPLAN"}:
                        state.current_error_type = "TRANSIENT"
                else:
                    tool_result = "ERROR"
                    last_action_error = True
                    state.same_error_count += 1
                    self._advance_error_on_failure(action_type, previous_error_type)
                    self._apply_task_drift()

        state.last_action = action_type
        state.budget_remaining = max(
            0.0,
            state.budget_remaining - (0.1 + self._active_tool_cost()),
        )
        progress_delta = round(state.progress - previous_progress, 2)
        done = (
            state.progress >= 1.0
            or state.budget_remaining <= 0.0
            or state.step_count >= config.max_steps
        )

        return self._finalize_step(
            action_type=action_type,
            previous_error_type=previous_error_type,
            tool_result=tool_result,
            hidden_error_type=state.current_error_type,
            last_action_error=last_action_error,
            progress_delta=progress_delta,
            bad_retry=bad_retry,
            switched_tools=switched_tools,
            escalated_early=escalated_early,
            done=done,
        )

    @property
    def state(self) -> IRCEState:
        return self._state

    def _attempt_probability(self, action_type: str, previous_error_type: str) -> float:
        config = self.task_config
        tool_bonus = config.backup_success_bonus if self._state.tool_state == "backup" else 0.0
        planning_bonus = self._state.replan_bonus

        if action_type == "RETRY":
            if previous_error_type == "TRANSIENT" and self._state.cooldown_remaining == 0:
                return config.retry_success + (tool_bonus * 0.25) + planning_bonus
            return 0.0

        if action_type == "MODIFY":
            if previous_error_type == "HARD":
                return config.modify_success + (tool_bonus * 0.2) + planning_bonus
            if previous_error_type == "TRANSIENT":
                return 0.28 + planning_bonus
            return 0.12 + planning_bonus

        if action_type == "SWITCH":
            if previous_error_type == "RATE_LIMIT":
                return config.switch_success + tool_bonus
            if previous_error_type == "HARD":
                return 0.38 + tool_bonus
            return 0.5 + (tool_bonus * 0.5)

        if action_type == "REPLAN":
            if previous_error_type == "RATE_LIMIT":
                return 0.3 + planning_bonus
            if previous_error_type == "HARD":
                return config.replan_success + (planning_bonus * 0.5)
            return 0.42 + planning_bonus

        return 0.0

    def _is_ambiguous_outcome(self, action_type: str, previous_error_type: str) -> bool:
        if action_type == "ESCALATE":
            return False

        if previous_error_type == "RATE_LIMIT" and action_type == "RETRY":
            return False

        ambiguity_bonus = 0.05 if action_type in {"REPLAN", "MODIFY"} else 0.0
        return self._rng.random() < min(0.4, self.task_config.ambiguity_rate + ambiguity_bonus)

    def _apply_task_drift(self) -> None:
        if not self.task_config.drift_enabled or self.task_config.drift_after_step is None:
            return

        if self._state.step_count < self.task_config.drift_after_step:
            return

        if self._state.tool_state != "primary" or self._rng.random() >= 0.35:
            return

        if self._state.current_error_type == "TRANSIENT":
            self._state.current_error_type = "HARD"
        elif self.task_config.allow_rate_limit and self._state.current_error_type == "HARD":
            self._state.current_error_type = "RATE_LIMIT"
            self._state.cooldown_remaining = max(
                self._state.cooldown_remaining,
                self.task_config.rate_limit_cooldown,
            )

    def _advance_error_on_failure(self, action_type: str, previous_error_type: str) -> None:
        state = self._state
        config = self.task_config

        if previous_error_type == "RATE_LIMIT":
            state.current_error_type = "TRANSIENT" if action_type == "SWITCH" else "RATE_LIMIT"
            state.cooldown_remaining = max(state.cooldown_remaining, config.rate_limit_cooldown)
            return

        if previous_error_type == "HARD":
            if action_type in {"MODIFY", "REPLAN"} and state.same_error_count == 1:
                state.current_error_type = "TRANSIENT"
            elif config.allow_rate_limit and state.same_error_count >= 2:
                state.current_error_type = "RATE_LIMIT"
                state.cooldown_remaining = max(state.cooldown_remaining, config.rate_limit_cooldown)
            else:
                state.current_error_type = "HARD"
            return

        if config.allow_rate_limit and action_type == "RETRY" and state.same_error_count >= 2:
            state.current_error_type = "RATE_LIMIT"
            state.cooldown_remaining = max(state.cooldown_remaining, config.rate_limit_cooldown)
        elif state.same_error_count >= 1:
            state.current_error_type = "HARD"
        else:
            state.current_error_type = "TRANSIENT"

    def _active_tool_cost(self) -> float:
        return self.task_config.backup_budget_cost if self._state.tool_state == "backup" else 0.0

    def _finalize_step(
        self,
        action_type: str,
        previous_error_type: str,
        tool_result: str,
        hidden_error_type: str,
        last_action_error: bool,
        progress_delta: float,
        bad_retry: bool,
        switched_tools: bool,
        escalated_early: bool,
        done: bool,
    ) -> IRCEObservation:
        completed = self._state.progress >= 1.0
        resolved_ambiguity = self._state.last_tool_result == "AMBIGUOUS" and tool_result == "SUCCESS"
        reward_breakdown = compute_step_reward(
            action_type=action_type,
            tool_result=tool_result,
            previous_error_type=previous_error_type,
            same_error_count=self._state.same_error_count,
            progress_delta=progress_delta,
            completed=completed,
            switched_tools=switched_tools,
            backup_budget_cost=self.task_config.backup_budget_cost,
            cascade_penalty=self.task_config.cascade_penalty,
            escalated_early=escalated_early,
        )
        observation = self._observation(
            tool_result=tool_result,
            last_action_error=last_action_error,
            reward=reward_breakdown.total,
            done=done,
        )
        history_entry = (
            f"step={self._state.step_count} {action_type}->{observation.tool_result}"
            f" [{observation.error_type}]"
            f" tool={self._state.tool_state}"
            f" budget={observation.budget_remaining:.2f}"
        )
        self._state.history.append(history_entry)
        self._state.history = self._state.history[-3:]
        observation.history_tail = list(self._state.history)
        observation.status_summary = self._status_summary(observation)
        observation.decision_context = self._decision_context(observation)
        self._state.last_reward = reward_breakdown.total
        self._state.last_tool_result = tool_result
        self.episode_log.append(
            {
                "task_id": self.task_id,
                "task_name": self.task_config.name,
                "action": action_type,
                "reward": reward_breakdown.total,
                "tool_result": tool_result,
                "observed_error_type": observation.error_type,
                "hidden_error_type": hidden_error_type,
                "step_count": self._state.step_count,
                "max_steps": self.task_config.max_steps,
                "budget_remaining": round(self._state.budget_remaining, 2),
                "progress": round(self._state.progress, 2),
                "progress_hint": observation.progress_hint,
                "active_tool": self._state.tool_state,
                "cooldown_remaining": self._state.cooldown_remaining,
                "bad_retry": bad_retry,
                "resolved_ambiguity": resolved_ambiguity,
                "ambiguous_outcome": tool_result == "AMBIGUOUS",
                "done": done,
                "completed": completed,
            }
        )
        return observation

    def _observation(
        self,
        tool_result: str,
        last_action_error: bool,
        reward: float,
        done: bool = False,
    ) -> IRCEObservation:
        observed_error_type = self._state.current_error_type
        if (
            tool_result in {"ERROR", "AMBIGUOUS"}
            and self.task_config.noise_level > 0.0
            and self._rng.random() < self.task_config.noise_level
        ):
            alternatives = [
                error_type
                for error_type in self.task_config.initial_errors
                if error_type != self._state.current_error_type
            ]
            if alternatives:
                observed_error_type = self._rng.choice(alternatives)

        progress_hint = self._progress_hint()
        observation = IRCEObservation(
            goal=self._state.goal,
            tool_result=tool_result,
            error_type=observed_error_type,
            same_error_count=self._state.same_error_count,
            budget_remaining=round(self._state.budget_remaining, 2),
            step_count=self._state.step_count,
            last_action_error=last_action_error,
            active_tool=self._state.tool_state,
            cooldown_remaining=self._state.cooldown_remaining,
            progress_hint=progress_hint,
            history_tail=list(self._state.history),
            status_summary="",
            reward=round(reward, 3),
            done=done,
        )
        observation.status_summary = self._status_summary(observation)
        observation.decision_context = self._decision_context(observation)
        return self._apply_transform(observation)

    def _status_summary(self, observation: IRCEObservation) -> str:
        steps_left = self.task_config.max_steps - observation.step_count
        recent_history = " -> ".join(observation.history_tail[-2:]) if observation.history_tail else "none"

        return (
            f"result={observation.tool_result}; "
            f"error={observation.error_type}; "
            f"tool={observation.active_tool}; "
            f"budget={observation.budget_remaining:.2f} ({observation.budget_remaining:.0%}); "
            f"step={observation.step_count}/{self.task_config.max_steps} ({steps_left} left); "
            f"cooldown={observation.cooldown_remaining}; "
            f"repeat_errors={observation.same_error_count}; "
            f"progress={observation.progress_hint:.0%}; "
            f"recent={recent_history}"
        )

    def _decision_context(self, observation: IRCEObservation) -> str:
        """Factual signal summary — describes active constraints, not recommended actions."""
        signals: list[str] = []

        # Cooldown — state the constraint, not the implication for specific actions
        if observation.cooldown_remaining > 0:
            signals.append(
                f"cooldown active ({observation.cooldown_remaining} step(s) remaining) — "
                "same-path retry attempts blocked until cooldown clears"
            )

        # Ambiguous outcome — state what happened, not what to do next
        if observation.tool_result == "AMBIGUOUS":
            signals.append(
                "partial progress recorded but outcome unconfirmed — "
                "ambiguity carried over from previous step"
            )

        # Error type — describe the failure pattern, not action preference
        if observation.error_type == "RATE_LIMIT":
            signals.append("rate-limited — same-path retries continuing to fail")
        elif observation.error_type == "HARD":
            if observation.same_error_count >= 2:
                signals.append(
                    f"hard failure on {observation.same_error_count} consecutive steps — "
                    "same-error penalty accumulating"
                )
            elif observation.same_error_count == 1:
                signals.append("hard failure on consecutive steps — structural issue likely")
            else:
                signals.append("hard failure — structural issue detected on this step")
        elif observation.error_type == "TRANSIENT":
            if observation.same_error_count >= 1:
                signals.append(
                    f"transient failure repeated {observation.same_error_count} time(s) — "
                    "pattern persisting"
                )
            else:
                signals.append("transient failure — single occurrence this episode")

        # Budget and tool cost — state facts, not cost-avoidance advice
        steps_left = self.task_config.max_steps - observation.step_count
        if observation.budget_remaining < 0.2:
            signals.append(
                f"budget at {observation.budget_remaining:.0%} with {steps_left} step(s) remaining"
            )
        elif observation.active_tool == "backup":
            signals.append("on backup tool — elevated per-step budget cost active")

        if steps_left <= 1 and observation.progress_hint < 0.8:
            signals.append(
                f"final step available — progress at {observation.progress_hint:.0%}"
            )

        return " | ".join(signals) if signals else "no active constraints detected"

    def _progress_hint(self) -> float:
        if self.task_config.noise_level <= 0.0:
            return round(self._state.progress, 2)

        direction = -1 if self._state.step_count % 2 else 1
        return round(
            max(0.0, min(1.0, self._state.progress + (direction * self.task_config.noise_level * 0.2))),
            2,
        )

    def _build_episode_id(self, seed: int) -> str:
        return f"irce-{seed}-{self._episode_index}"