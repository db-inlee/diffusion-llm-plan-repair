"""Plans in and out of prompts.

Out: the plan and the task as JSON, so the model sees the same contract the validator enforces.
In: text back to an :class:`AgentPlan`, or a refusal.

The refusal matters more than the parsing. A model that returns broken JSON or a plan that does
not satisfy the schema has failed to repair, and that failure is part of what the model is being
measured on. Nothing here tries to rescue a malformed answer — the one accommodation is a
markdown code fence, which is a formatting habit rather than a defect in the answer. Anything
else raises :class:`PlanParseError`, and the caller records a failed repair.
"""

import json
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from plan_repair.schema.plan import AgentPlan
from plan_repair.schema.task import AgentTask


class PlanParseError(ValueError):
    """The model's answer was not a plan."""


def plan_to_json(plan: AgentPlan) -> str:
    """Serialize a plan for a prompt."""
    return json.dumps(plan.model_dump(), ensure_ascii=False, indent=2)


def task_to_json(task: AgentTask) -> str:
    """Serialize the parts of the task a repairer is allowed to see."""
    return json.dumps(
        {
            "task_id": task.task_id,
            "user_query": task.user_query,
            "available_tools": [tool.name for tool in task.available_tools],
            "required_evidence": task.required_evidence,
            "required_operations": task.required_operations,
        },
        ensure_ascii=False,
        indent=2,
    )


def parse_plan(text: str) -> AgentPlan:
    """Read a plan out of a model answer, or raise :class:`PlanParseError`."""
    payload = _strip_code_fence(text.strip())
    if not payload:
        raise PlanParseError("the answer was empty")
    try:
        data: Any = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise PlanParseError(f"the answer was not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PlanParseError(f"expected a JSON object, got {type(data).__name__}")
    try:
        return AgentPlan.model_validate(data)
    except PydanticValidationError as exc:
        raise PlanParseError(f"the answer did not satisfy the plan schema: {exc}") from exc


def _strip_code_fence(text: str) -> str:
    """Unwrap a single ```-delimited block, if the answer came in one."""
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 2 or not lines[-1].strip().startswith("```"):
        return text
    return "\n".join(lines[1:-1]).strip()


__all__ = ["PlanParseError", "parse_plan", "plan_to_json", "task_to_json"]
