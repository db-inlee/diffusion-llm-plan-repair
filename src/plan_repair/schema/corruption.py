"""Corruption metadata — the ground truth recorded at injection time.

``InjectedError`` is what detection recall is measured against: whatever the injector claims
to have damaged must be reported by the validator at step/path level.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from plan_repair.schema.plan import AgentPlan

BROKEN_DEPENDENCY = "broken_dependency"
STEP_DELETION = "step_deletion"
WRONG_TOOL = "wrong_tool"
WRONG_ORDERING = "wrong_ordering"
DUPLICATE_STEP = "duplicate_step"
MISSING_STOP_CONDITION = "missing_stop_condition"


class InjectedError(BaseModel):
    """Ground truth for one injected corruption.

    ``damaged_step_ids`` / ``damaged_paths`` only ever point at steps that exist in the broken
    plan. A deleted step has no location in the broken plan, so step deletion records the
    downstream steps whose references broke, and keeps the deleted step itself in ``detail``.
    """

    model_config = ConfigDict(extra="forbid")

    corruption_type: str
    damaged_step_ids: list[str]
    damaged_paths: list[str]
    detail: dict[str, Any] = Field(default_factory=dict)


class CorruptionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    broken_plan: AgentPlan
    injected: list[InjectedError]
    preserved_step_ids: list[str]
