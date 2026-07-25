"""Corruption injection — structural errors with recorded ground truth."""

from plan_repair.corruption.injector import (
    CYCLE_MODE,
    UNKNOWN_MODE,
    inject_broken_dependency,
    inject_step_deletion,
)

__all__ = [
    "CYCLE_MODE",
    "UNKNOWN_MODE",
    "inject_broken_dependency",
    "inject_step_deletion",
]
