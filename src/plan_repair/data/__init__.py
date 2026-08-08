"""Reference plans.

Reference data is kept as JSON so it stays diffable against the scenario document, and is parsed
into the pydantic contract on load. Domain B is the 20-step data analysis pipeline; domain A is
the 19-step research report pipeline, added so the validator and the injectors can be shown to
work without any domain-specific knowledge.
"""

import json
from importlib.resources import files
from typing import Any

from plan_repair.schema.plan import AgentPlan
from plan_repair.schema.task import AgentTask

DATA_PIPELINE_A = "data_pipeline_a"
DATA_PIPELINE_B = "data_pipeline_b"
REFERENCE_DOMAINS = (DATA_PIPELINE_A, DATA_PIPELINE_B)


def load_reference(name: str = DATA_PIPELINE_B) -> tuple[AgentTask, AgentPlan]:
    """Load a reference ``(task, plan)`` pair by file name."""
    payload = _load_json(name)
    return AgentTask.model_validate(payload["task"]), AgentPlan.model_validate(payload["plan"])


def load_task(name: str = DATA_PIPELINE_B) -> AgentTask:
    return AgentTask.model_validate(_load_json(name)["task"])


def load_reference_plan(name: str = DATA_PIPELINE_B) -> AgentPlan:
    return AgentPlan.model_validate(_load_json(name)["plan"])


def all_tool_names() -> set[str]:
    """Every tool name any reference task offers.

    A corruption that needs a plausible but *unavailable* tool draws from here and subtracts the
    task's own list: the names are real, chosen by a person for a real pipeline, and inventing
    strings to sit in a tool field would make the corruption a test of the inventor's taste.
    """
    return {name for domain in REFERENCE_DOMAINS for name in load_task(domain).tool_names()}


def _load_json(name: str) -> dict[str, Any]:
    resource = files(__package__).joinpath(f"{name}.json")
    payload: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
    return payload


__all__ = [
    "DATA_PIPELINE_A",
    "DATA_PIPELINE_B",
    "REFERENCE_DOMAINS",
    "all_tool_names",
    "load_reference",
    "load_reference_plan",
    "load_task",
]
