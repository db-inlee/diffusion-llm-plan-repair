"""Detection metrics — how a validation result compares to an expected error set.

Tickets 001 and 002 measured recall by containment: every injected step/path had to appear
somewhere in the output. That is too loose once several corruptions overlap, because a validator
that over-reports would still score 1.0. This module compares *error sets*:

* **recall** — share of expected errors the validator reported (a miss is a plan that ships broken)
* **precision** — share of reported errors that were expected (a spurious error sends a repairer
  at healthy steps)

An expected set holds both the errors a corruption causes directly and the ones that follow from
it structurally; deciding which derived errors are legitimate is a human judgement recorded in
the golden, never read back out of the validator.

Signatures deliberately cover ``(type, step_ids, paths)`` only: ``detail`` carries explanatory
extras, and tests assert on it separately where it matters.
"""

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from plan_repair.validation.models import PlanValidationResult, ValidationError

ErrorSignature = tuple[str, tuple[str, ...], tuple[str, ...]]


class DetectionMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recall: float
    precision: float
    missed: list[ErrorSignature]
    spurious: list[ErrorSignature]

    @property
    def exact(self) -> bool:
        """Whether the detected set matches the expected set exactly."""
        return not self.missed and not self.spurious


def error_signature(error: ValidationError) -> ErrorSignature:
    return (error.type, tuple(error.step_ids), tuple(error.paths))


def detection_metrics(
    result: PlanValidationResult, expected: Iterable[ErrorSignature]
) -> DetectionMetrics:
    """Compare the errors of ``result`` against the expected set."""
    detected = {error_signature(error) for error in result.errors}
    wanted = set(expected)
    matched = detected & wanted
    return DetectionMetrics(
        recall=1.0 if not wanted else len(matched) / len(wanted),
        precision=1.0 if not detected else len(matched) / len(detected),
        missed=sorted(wanted - detected),
        spurious=sorted(detected - wanted),
    )


__all__ = ["DetectionMetrics", "ErrorSignature", "detection_metrics", "error_signature"]
