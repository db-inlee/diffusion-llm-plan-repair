"""Agent plan data contract.

The schema deliberately carries no free-text description fields: surface-level collateral
edits must be impossible to express, so that collateral edit measurement stays structural
(project plan section 7.5).
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Step(BaseModel):
    """A single tool call in a plan.

    ``input_from`` holds the ids of preceding steps whose output this step consumes; it is
    the dependency edge set of the plan graph.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    input_from: list[str] = Field(default_factory=list)
    # Names from the task's ``required_evidence`` / ``required_operations`` that this step
    # satisfies. Tagging is explicit rather than inferred from tool names: a naming convention
    # would only hold for the pipeline it was written against.
    produces: list[str] = Field(default_factory=list)


class AgentPlan(BaseModel):
    """An ordered list of steps meant to satisfy an :class:`~plan_repair.schema.task.AgentTask`.

    The order of ``steps`` is meaningful: a step may only consume steps that appear before it
    (checked by the ordering validator).
    """

    model_config = ConfigDict(extra="forbid")

    goal: str
    steps: list[Step]
    stop_condition: str | None = None
