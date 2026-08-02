"""Deterministic execution environment and its agreement with the static validator.

A package of its own because execution is a third axis of scoring next to structure and meaning:
the validator says whether a plan is well formed, the runtime says whether it would get anywhere.
"""

from plan_repair.runtime.agreement import (
    AGREE,
    VALIDATOR_MISSED,
    VALIDATOR_STRICTER,
    Agreement,
    agreement,
)
from plan_repair.runtime.execution import RunResult, StepOutcome, run_plan

__all__ = [
    "AGREE",
    "VALIDATOR_MISSED",
    "VALIDATOR_STRICTER",
    "Agreement",
    "RunResult",
    "StepOutcome",
    "agreement",
    "run_plan",
]
