"""Mock repairers that pin the two ends of the scale.

Neither is a candidate in the real comparison. They exist so the scoring pipeline can be checked
against known answers: if the scale cannot tell the repairer that fixes nothing from the one that
fixes everything, the scale is broken and no result measured with it means anything.

* :class:`IdentityRepairer` — returns the broken plan untouched. The floor: every injected error
  survives, and collateral is zero because nothing was touched at all.
* :class:`OracleRepairer` — returns the reference plan it was built with. The ceiling: no error
  survives and collateral is zero, because the healthy steps are literally the original ones.

Identity and oracle share a collateral of zero for opposite reasons, which is exactly why
collateral is reported next to repair quality and never on its own.
"""

from plan_repair.schema.plan import AgentPlan
from plan_repair.schema.task import AgentTask
from plan_repair.validation.models import PlanValidationResult


class IdentityRepairer:
    """Repairs nothing."""

    name = "identity"

    def repair(
        self,
        broken_plan: AgentPlan,
        validation: PlanValidationResult,
        task: AgentTask,
    ) -> AgentPlan:
        return broken_plan.model_copy(deep=True)


class OracleRepairer:
    """Repairs perfectly, by holding the answer.

    Only legitimate as an upper bound: it is given the plan before corruption, which no real
    repairer sees.
    """

    name = "oracle"

    def __init__(self, reference_plan: AgentPlan) -> None:
        self._reference_plan = reference_plan

    def repair(
        self,
        broken_plan: AgentPlan,
        validation: PlanValidationResult,
        task: AgentTask,
    ) -> AgentPlan:
        return self._reference_plan.model_copy(deep=True)


__all__ = ["IdentityRepairer", "OracleRepairer"]
