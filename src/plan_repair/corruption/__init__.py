"""Corruption injection — structural errors with recorded ground truth."""

from plan_repair.corruption.injector import (
    CYCLE_MODE,
    UNKNOWN_MODE,
    inject_broken_dependency,
    inject_duplicate_step,
    inject_missing_stop_condition,
    inject_step_deletion,
    inject_wrong_ordering,
    inject_wrong_tool,
)
from plan_repair.corruption.multi import CorruptionSpec, inject_multi, touched_steps

__all__ = [
    "CYCLE_MODE",
    "UNKNOWN_MODE",
    "CorruptionSpec",
    "inject_broken_dependency",
    "inject_duplicate_step",
    "inject_missing_stop_condition",
    "inject_multi",
    "inject_step_deletion",
    "inject_wrong_ordering",
    "inject_wrong_tool",
    "touched_steps",
]
