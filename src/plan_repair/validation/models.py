"""Validator contract.

``step_ids`` and ``paths`` are always filled precisely: they are the input of the later
error-to-span mapping, and detection recall is measured against them.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SCHEMA = "schema"
UNKNOWN_TOOL = "unknown_tool"
UNKNOWN_DEPENDENCY = "unknown_dependency"
DEP_CYCLE = "dep_cycle"
ORDERING = "ordering"
DUPLICATE_STEP = "duplicate_step"
MISSING_STOP_CONDITION = "missing_stop_condition"
DANGLING_STEP = "dangling_step"

# Declared by the ticket's error type union but produced by none of the five checks in this
# ticket (it belongs to the coverage checks of a later ticket).
MISSING_DEPENDENCY = "missing_dependency"


class ValidationError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    step_ids: list[str]
    paths: list[str]
    message: str
    # Optional structured extras (for example whether a duplicate is an id clash or identical
    # content). Defaults to empty, so errors written against the Ticket 001 contract still match.
    detail: dict[str, Any] = Field(default_factory=dict)


class PlanValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    errors: list[ValidationError]

    def errors_of_type(self, error_type: str) -> list[ValidationError]:
        return [error for error in self.errors if error.type == error_type]

    def detected_step_ids(self) -> set[str]:
        return {step_id for error in self.errors for step_id in error.step_ids}

    def detected_paths(self) -> set[str]:
        return {path for error in self.errors for path in error.paths}
