"""Corruption injection — structural errors with recorded ground truth."""

from plan_repair.corruption.injector import (
    CYCLE_MODE,
    LENGTH_MATCHED_MODE,
    SUFFIX_MODE,
    UNKNOWN_MODE,
    CorruptionNotApplicableError,
    TokenLength,
    VocabularyLength,
    inject_broken_dependency,
    inject_drop_required_step,
    inject_duplicate_step,
    inject_missing_stop_condition,
    inject_step_deletion,
    inject_wrong_ordering,
    inject_wrong_tool,
    inject_wrong_tool_length_matched,
    length_matched_tools,
)
from plan_repair.corruption.multi import CorruptionSpec, inject_multi, touched_steps

__all__ = [
    "CYCLE_MODE",
    "LENGTH_MATCHED_MODE",
    "SUFFIX_MODE",
    "UNKNOWN_MODE",
    "CorruptionNotApplicableError",
    "CorruptionSpec",
    "TokenLength",
    "VocabularyLength",
    "inject_broken_dependency",
    "inject_drop_required_step",
    "inject_duplicate_step",
    "inject_missing_stop_condition",
    "inject_multi",
    "inject_step_deletion",
    "inject_wrong_ordering",
    "inject_wrong_tool",
    "inject_wrong_tool_length_matched",
    "length_matched_tools",
    "touched_steps",
]
