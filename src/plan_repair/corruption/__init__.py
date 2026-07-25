"""Corruption injection — structural errors with recorded ground truth."""

from plan_repair.corruption.injector import (
    CYCLE_MODE,
    UNKNOWN_MODE,
    inject_broken_dependency,
    inject_duplicate_step,
    inject_step_deletion,
    inject_wrong_ordering,
    inject_wrong_tool,
)

__all__ = [
    "CYCLE_MODE",
    "UNKNOWN_MODE",
    "inject_broken_dependency",
    "inject_duplicate_step",
    "inject_step_deletion",
    "inject_wrong_ordering",
    "inject_wrong_tool",
]
