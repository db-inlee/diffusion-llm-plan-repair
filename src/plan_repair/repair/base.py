"""The port every repairer plugs into.

One signature for all of them — rule-based, autoregressive, diffusion — because a comparison is
only fair if every candidate is handed exactly the same thing and asked for exactly the same
thing back.

What a repairer is given:

* the broken plan,
* the validator's verdict on it, whose errors carry ``step_ids`` and ``paths`` — this is how the
  error region reaches the repairer, and it is what a diffusion repairer will later turn into a
  remask region,
* the task, for the tools and requirements the plan has to satisfy.

What it returns is a plan. Nothing else: a repairer that also reported what it fixed would be
scored on its own testimony, and a model-based repairer cannot give that testimony anyway. What
was fixed is established by re-validating the returned plan, which works the same for every
repairer.

This module is deliberately free of any model or hardware concern; nothing under ``repair/``
imports torch.
"""

from typing import Protocol, runtime_checkable

from plan_repair.schema.plan import AgentPlan
from plan_repair.schema.task import AgentTask
from plan_repair.validation.models import PlanValidationResult


@runtime_checkable
class Repairer(Protocol):
    """A plan repairer. ``name`` identifies it in comparison reports."""

    name: str

    def repair(
        self,
        broken_plan: AgentPlan,
        validation: PlanValidationResult,
        task: AgentTask,
    ) -> AgentPlan:
        """Return a repaired copy of ``broken_plan``.

        Implementations must not mutate ``broken_plan``.
        """
        ...


__all__ = ["Repairer"]
