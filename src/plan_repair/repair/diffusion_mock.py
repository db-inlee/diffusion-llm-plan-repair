"""Mock diffusion repairers — the remask pipeline without a model in it.

The point of a mock here is not to imitate denoising. It is to hold the fill step still so that
the part this ticket owns can be judged on its own: does the mask land on the damaged steps, does
everything outside it survive byte for byte, and does the filled sequence read back as a plan.

* :class:`OracleDiffusion` fills each masked span with the step as it was before the corruption.
  If masking is correct this restores exactly the damaged steps and leaves the rest identical, so
  a collateral reading above zero here means the mask reached too far — the same use of a known
  answer that identity and oracle serve in Ticket B-1.
* :class:`NoisyDiffusion` fills the spans with something that is not a step, to check that a
  sequence which no longer parses is reported as a failed repair rather than crashing.

Both take the error region from the validator, which is what a real repairer sees — the same hint
the AR local mode is given, so the two remain comparable.

Neither can reach past the mask, and that has consequences worth stating plainly: a step the
corruption deleted has no span to fill, and the order of the steps is not inside any span. Those
repairs are out of reach of step-level remasking, whatever fills the mask.
"""

import json

from plan_repair.repair.ar import PARSE_FAILURE, RepairFailure
from plan_repair.repair.plan_io import PlanParseError
from plan_repair.repair.remask import (
    MaskSpec,
    PlanSequence,
    fill_masked,
    mask_spec,
    plan_to_sequence,
    sequence_to_plan,
)
from plan_repair.schema.plan import AgentPlan
from plan_repair.schema.task import AgentTask
from plan_repair.validation.models import PlanValidationResult


class _MaskingRepairer:
    """Mask what the validator flagged, fill it, read the plan back."""

    name = "diffusion"

    def __init__(self) -> None:
        self.failures: list[RepairFailure] = []
        self.last_mask: MaskSpec | None = None

    def repair(
        self,
        broken_plan: AgentPlan,
        validation: PlanValidationResult,
        task: AgentTask,
    ) -> AgentPlan:
        sequence = plan_to_sequence(broken_plan)
        spec = mask_spec(sequence, validation.detected_step_ids())
        self.last_mask = spec
        filled = fill_masked(sequence, spec, self._fill(sequence, spec))
        try:
            return sequence_to_plan(filled)
        except PlanParseError as exc:
            self.failures.append(
                RepairFailure(repairer=self.name, kind=PARSE_FAILURE, detail=str(exc))
            )
            return broken_plan.model_copy(deep=True)

    def _fill(
        self, sequence: PlanSequence, spec: MaskSpec
    ) -> dict[str, str | None]:  # pragma: no cover - overridden
        raise NotImplementedError


class OracleDiffusion(_MaskingRepairer):
    """Fills each masked span with the step as the reference plan has it.

    A masked step the reference does not contain is dropped: the answer to "what belongs here"
    is then "nothing", which is how a duplicated step disappears.
    """

    name = "diffusion_oracle"

    def __init__(self, reference_plan: AgentPlan) -> None:
        super().__init__()
        self._reference = {step.id: step for step in reference_plan.steps}

    def _fill(self, sequence: PlanSequence, spec: MaskSpec) -> dict[str, str | None]:
        filling: dict[str, str | None] = {}
        for step_id in spec.masked_step_ids:
            original = self._reference.get(step_id)
            filling[step_id] = (
                None if original is None else json.dumps(original.model_dump(), ensure_ascii=False)
            )
        return filling


class NoisyDiffusion(_MaskingRepairer):
    """Fills the mask with text that is not a step, to exercise the failure path."""

    name = "diffusion_noisy"

    def __init__(self, noise: str = '"a plausible looking step"') -> None:
        super().__init__()
        self._noise = noise

    def _fill(self, sequence: PlanSequence, spec: MaskSpec) -> dict[str, str | None]:
        return dict.fromkeys(spec.masked_step_ids, self._noise)


__all__ = ["NoisyDiffusion", "OracleDiffusion"]
