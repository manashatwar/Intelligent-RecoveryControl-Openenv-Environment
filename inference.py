from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from dotenv import load_dotenv

if TYPE_CHECKING:
    from openai import OpenAI

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("IRCE_STANDALONE", "1")

from irce.environment import IRCEEnv
from irce.grading import grade_episode
from irce.models import IRCEAction, IRCEObservation
from irce.tasks import build_task_registry

SUPPORTED_ACTIONS = ("RETRY", "MODIFY", "SWITCH", "REPLAN", "ESCALATE")
SYSTEM_PROMPT = (
    "You are controlling a recovery policy for IRCE. "
    "Choose exactly one action from RETRY, MODIFY, SWITCH, REPLAN, ESCALATE. "
    "Prefer stable recovery, low waste, and low repeated failure."
)
ObservationT = TypeVar("ObservationT")


@dataclass
class StepResult(Generic[ObservationT]):
    observation: ObservationT
    reward: float | None = None
    done: bool = False


@dataclass
class PolicyMemory:
    last_action: str | None = None
    consecutive_failures: int = 0
    success_count: int = 0
    ambiguous_streak: int = 0
    recent_results: deque[str] = field(default_factory=lambda: deque(maxlen=3))
    recent_errors: deque[str] = field(default_factory=lambda: deque(maxlen=3))

    def update(self, action_type: str, observation: IRCEObservation) -> None:
        self.last_action = action_type
        self.recent_results.append(observation.tool_result)
        self.recent_errors.append(observation.error_type)

        if observation.tool_result == "SUCCESS":
            self.success_count += 1
            self.consecutive_failures = 0
            self.ambiguous_streak = 0
        elif observation.tool_result == "AMBIGUOUS":
            self.consecutive_failures = 0
            self.ambiguous_streak += 1
        else:
            self.consecutive_failures += 1
            self.ambiguous_streak = 0


class LocalEnvRunner:
    """Small StepResult adapter so inference matches the required client pattern."""

    def __init__(self, task_id: int, seed: int) -> None:
        self._env = IRCEEnv(task_id=task_id, seed=seed)
        self.task_id = task_id
        self.seed = seed

    @property
    def episode_log(self) -> list[dict[str, Any]]:
        return self._env.episode_log

    def reset(self) -> StepResult[IRCEObservation]:
        observation = self._env.reset(seed=self.seed, task_id=self.task_id)
        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    def step(self, action: IRCEAction) -> StepResult[IRCEObservation]:
        observation = self._env.step(action)
        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    def close(self) -> None:
        return None


def select_fallback_action(observation: IRCEObservation, memory: PolicyMemory) -> str:
    if (
        observation.budget_remaining <= 0.12
        and memory.consecutive_failures >= 3
        and observation.progress_hint < 0.5
    ):
        return "ESCALATE"

    if observation.tool_result == "AMBIGUOUS":
        if observation.progress_hint >= 0.7:
            return "REPLAN"
        if observation.error_type == "HARD":
            return "MODIFY"
        if observation.error_type == "RATE_LIMIT" or observation.cooldown_remaining > 0:
            return "REPLAN"
        return "REPLAN" if memory.ambiguous_streak >= 1 else "RETRY"

    if observation.cooldown_remaining > 0:
        return "SWITCH"

    if observation.error_type == "RATE_LIMIT":
        return "SWITCH" if observation.active_tool == "primary" else "REPLAN"

    if observation.same_error_count >= 2:
        return "SWITCH" if observation.active_tool == "primary" else "REPLAN"

    if observation.error_type == "HARD":
        if observation.active_tool == "primary" and memory.last_action != "MODIFY":
            return "MODIFY"
        return "SWITCH"

    if observation.error_type == "TRANSIENT":
        if observation.progress_hint >= 0.7:
            return "RETRY"
        if memory.consecutive_failures >= 1:
            return "SWITCH"
        return "RETRY" if observation.active_tool == "primary" else "SWITCH"

    return "REPLAN"


def build_user_prompt(step: int, observation: IRCEObservation, history: list[str]) -> str:
    recent_history = "\n".join(history[-3:]) if history else "none"
    return (
        f"Task step: {step}\n"
        f"Goal: {observation.goal}\n"
        f"Status: {observation.status_summary}\n"
        f"Recent history:\n{recent_history}\n"
        "Respond with exactly one action name: RETRY, MODIFY, SWITCH, REPLAN, or ESCALATE."
    )


def build_openai_client() -> tuple[Any | None, str, str]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or ""
    base_url = os.getenv("API_BASE_URL", "").strip()
    model_name = os.getenv("MODEL_NAME", "").strip()

    if not api_key or not model_name:
        return None, model_name, "OPENAI_API_KEY/HF_TOKEN or MODEL_NAME missing"

    try:
        from openai import OpenAI
    except Exception as exc:  # noqa: BLE001
        return None, model_name, f"OpenAI SDK import failed: {exc}"

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    return OpenAI(**client_kwargs), model_name, ""


def parse_model_action(response_text: str) -> str | None:
    if not response_text:
        return None

    text = response_text.strip()

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            candidate = str(payload.get("action_type", "")).strip().upper()
            if candidate in SUPPORTED_ACTIONS:
                return candidate
    except json.JSONDecodeError:
        pass

    match = re.search(r"\b(RETRY|MODIFY|SWITCH|REPLAN|ESCALATE)\b", text.upper())
    return match.group(1) if match else None


def request_action(
    *,
    client: Any | None,
    model_name: str,
    observation: IRCEObservation,
    memory: PolicyMemory,
    history: list[str],
    step: int,
) -> tuple[str, str]:
    fallback_action = select_fallback_action(observation, memory)
    if client is None or not model_name:
        return fallback_action, "fallback"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(step, observation, history)},
    ]

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=12,
            stream=False,
        )
        response_text = completion.choices[0].message.content or ""
        parsed_action = parse_model_action(response_text)
        if parsed_action:
            return parsed_action, "model"
    except Exception as exc:  # noqa: BLE001
        print(f"Model request failed ({exc}). Using fallback action.")

    return fallback_action, "fallback"


def run_task(task_id: int, seed: int, client: Any | None, model_name: str) -> float:
    task_config = build_task_registry()[task_id]
    env = LocalEnvRunner(task_id=task_id, seed=seed)
    memory = PolicyMemory()
    history: list[str] = []

    try:
        result = env.reset()
        observation = result.observation
        print(f"task_{task_id} reset:", observation.model_dump())

        for step in range(1, task_config.max_steps + 1):
            if result.done:
                break

            action_type, source = request_action(
                client=client,
                model_name=model_name,
                observation=observation,
                memory=memory,
                history=history,
                step=step,
            )
            print(f"task_{task_id} step_{step}: {source} -> {action_type}")

            result = env.step(IRCEAction(action_type=action_type))
            observation = result.observation
            reward = float(result.reward or 0.0)
            memory.update(action_type, observation)

            history_line = (
                f"step={step} action={action_type} result={observation.tool_result} "
                f"reward={reward:+.3f} done={result.done}"
            )
            history.append(history_line)
            print("  observation:", observation.model_dump())

            if result.done:
                break

        score = grade_episode(env.episode_log)
        print(f"task_{task_id} score: {score:.3f}")
        return score
    finally:
        env.close()


def main() -> None:
    load_dotenv()

    client, model_name, client_status = build_openai_client()
    if client is None:
        print(f"OpenAI client not configured ({client_status}). Using deterministic fallback policy.")
    else:
        print(f"OpenAI client configured for model '{model_name}'.")

    seed = 7
    task_registry = build_task_registry()
    scores = [
        run_task(task_id=task_id, seed=seed, client=client, model_name=model_name)
        for task_id in sorted(task_registry)
    ]
    average_score = sum(scores) / len(scores)
    print(f"average score: {average_score:.3f}")


if __name__ == "__main__":
    main()
