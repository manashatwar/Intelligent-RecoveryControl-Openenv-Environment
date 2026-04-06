from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field, field_validator

if os.getenv("IRCE_STANDALONE") == "1":
    class Action(BaseModel):
        """Lightweight Action model for standalone inference."""

    class Observation(BaseModel):
        """Lightweight Observation model for standalone inference."""

        done: bool = False
        reward: float = 0.0
        metadata: dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        """Lightweight State model for standalone inference."""

        episode_id: str = ""
else:
    try:
        from openenv.core.env_server import Action, Observation, State
    except ImportError:  # pragma: no cover
        from openenv_core.env_server import Action, Observation, State

SUPPORTED_ACTIONS = {"RETRY", "MODIFY", "SWITCH", "REPLAN", "ESCALATE"}


class IRCEAction(Action):
    """Minimal string-based recovery action."""

    action_type: str = "RETRY"

    @field_validator("action_type", mode="before")
    @classmethod
    def normalize_action_type(cls, value: Any) -> str:
        if value is None:
            return "RETRY"

        normalized = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        alias_map = {
            "MODIFY_INPUT": "MODIFY",
            "SWITCH_TOOL": "SWITCH",
        }
        return alias_map.get(normalized, normalized)

    @property
    def is_supported(self) -> bool:
        return self.action_type in SUPPORTED_ACTIONS


class IRCEObservation(Observation):
    """Readable observation for LLMs and simple agents."""

    goal: str = "Recover the task efficiently."
    tool_result: str = "ERROR"
    error_type: str = "TRANSIENT"
    same_error_count: int = Field(default=0, ge=0)
    budget_remaining: float = Field(default=1.0, ge=0.0, le=1.0)
    step_count: int = Field(default=0, ge=0)
    last_action_error: bool = False
    active_tool: str = "primary"
    cooldown_remaining: int = Field(default=0, ge=0)
    progress_hint: float = Field(default=0.0, ge=0.0, le=1.0)
    history_tail: list[str] = Field(default_factory=list)
    status_summary: str = ""
    decision_context: str = ""


class IRCEState(State):
    """Minimal hidden state used by the environment and grader."""

    goal: str = "Recover the task efficiently."
    task_name: str = "easy"
    current_error_type: str = "TRANSIENT"
    same_error_count: int = Field(default=0, ge=0)
    budget_remaining: float = Field(default=1.0, ge=0.0, le=1.0)
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    step_count: int = Field(default=0, ge=0)
    tool_state: str = "primary"
    replan_bonus: float = Field(default=0.0, ge=0.0, le=1.0)
    cooldown_remaining: int = Field(default=0, ge=0)
    ambiguous_count: int = Field(default=0, ge=0)
    last_tool_result: str = "ERROR"
    history: list[str] = Field(default_factory=list)
    last_action: str = "RESET"
    last_reward: float = 0.0