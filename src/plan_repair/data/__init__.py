"""Reference plans.

Reference data is kept as JSON so it stays diffable against the scenario document, and is parsed
into the pydantic contract on load. Domain B (the 20-step data analysis pipeline) is the only
reference of this ticket; domain A is reserved for the generality check of a later ticket.
"""

import json
from importlib.resources import files
from typing import Any

from plan_repair.schema.plan import AgentPlan
from plan_repair.schema.task import AgentTask

DATA_PIPELINE_B = "data_pipeline_b"


def load_reference(name: str = DATA_PIPELINE_B) -> tuple[AgentTask, AgentPlan]:
    """Load a reference ``(task, plan)`` pair by file name."""
    payload = _load_json(name)
    return AgentTask.model_validate(payload["task"]), AgentPlan.model_validate(payload["plan"])


def load_task(name: str = DATA_PIPELINE_B) -> AgentTask:
    return AgentTask.model_validate(_load_json(name)["task"])


def load_reference_plan(name: str = DATA_PIPELINE_B) -> AgentPlan:
    return AgentPlan.model_validate(_load_json(name)["plan"])


def _load_json(name: str) -> dict[str, Any]:
    resource = files(__package__).joinpath(f"{name}.json")
    payload: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
    return payload


__all__ = ["DATA_PIPELINE_B", "load_reference", "load_reference_plan", "load_task"]
