"""Agreement between the static verdict and the executed one.

Two independently written judges of the same plan: if they disagree, one of them is wrong, and
which one is a question for a person. Nothing here treats either side as the reference — the
disagreement is classified and reported, not resolved.
"""

from pydantic import BaseModel, ConfigDict

from plan_repair.runtime.execution import RunResult
from plan_repair.validation.models import PlanValidationResult

AGREE = "agree"
# The validator passed a plan that could not run: a blind spot in the static checks.
VALIDATOR_MISSED = "validator_missed"
# The validator rejected a plan that ran fine: a check stricter than execution.
VALIDATOR_STRICTER = "validator_stricter"


class Agreement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_valid: bool
    run_succeeded: bool
    verdict: str

    @property
    def agrees(self) -> bool:
        return self.verdict == AGREE


def agreement(validation: PlanValidationResult, run: RunResult) -> Agreement:
    if validation.valid == run.succeeded:
        verdict = AGREE
    elif validation.valid:
        verdict = VALIDATOR_MISSED
    else:
        verdict = VALIDATOR_STRICTER
    return Agreement(plan_valid=validation.valid, run_succeeded=run.succeeded, verdict=verdict)


__all__ = ["AGREE", "VALIDATOR_MISSED", "VALIDATOR_STRICTER", "Agreement", "agreement"]
