"""Scoring a repair on the three axes of stage A, plus collateral.

Repair quality is not one number. A repairer can make the errors go away and wreck the rest of
the plan doing it, so quality and collateral are always reported together:

* **structure and meaning** — does the validator still find errors?
* **execution** — does the repaired plan run to the end?
* **collateral** — how many healthy steps did the repair change?

Collateral follows the contract of the ticket exactly. A corruption records which steps it
damaged; every other step of the reference plan is healthy. A healthy step counts as collateral
when it is present in both the reference and the repaired plan and its **body** differs — body,
because what matters is whether the work a step does was altered, and the id-free hash from
Ticket 002 is what compares that. A damaged step coming back to its original form is a repair,
not collateral, which is why damaged steps are excluded from the count and reported separately as
``damaged_restored``.

Two things the collateral count deliberately does not absorb, kept as their own fields so they
cannot hide inside a zero: steps the repairer removed, and steps it added. Both are measured
against the broken plan — what the repairer was actually handed — so damage done by the
corruption is never charged to the repairer.
"""

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from plan_repair.canonical.canonicalize import step_body_hash
from plan_repair.repair.base import Repairer
from plan_repair.runtime.execution import run_plan
from plan_repair.schema.plan import AgentPlan
from plan_repair.schema.task import AgentTask
from plan_repair.validation.validator import validate_plan


class RepairScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repairer: str
    # structure and meaning
    valid: bool
    errors_remaining: int
    error_types_remaining: list[str]
    # execution
    runtime_succeeded: bool
    # collateral
    collateral: int
    collateral_step_ids: list[str]
    removed_step_ids: list[str]
    added_step_ids: list[str]
    # repair itself
    damaged_total: int
    damaged_restored: int

    @property
    def solved(self) -> bool:
        """Whether the plan came out clean on all three axes."""
        return self.valid and self.runtime_succeeded


def score_repair(
    *,
    repairer_name: str,
    reference_plan: AgentPlan,
    broken_plan: AgentPlan,
    repaired_plan: AgentPlan,
    task: AgentTask,
    damaged_step_ids: Iterable[str],
) -> RepairScore:
    """Score ``repaired_plan`` against the plan it should have been restored to."""
    damaged = set(damaged_step_ids)
    reference_bodies = _bodies(reference_plan)
    repaired_bodies = _bodies(repaired_plan)
    broken_ids = {step.id for step in broken_plan.steps}
    repaired_ids = set(repaired_bodies)

    healthy = [step_id for step_id in reference_bodies if step_id not in damaged]
    collateral = [
        step_id
        for step_id in healthy
        if step_id in repaired_bodies and repaired_bodies[step_id] != reference_bodies[step_id]
    ]
    restored = [
        step_id
        for step_id in damaged
        if step_id in repaired_bodies
        and step_id in reference_bodies
        and repaired_bodies[step_id] == reference_bodies[step_id]
    ]

    validation = validate_plan(repaired_plan, task)
    run = run_plan(repaired_plan, task)
    return RepairScore(
        repairer=repairer_name,
        valid=validation.valid,
        errors_remaining=len(validation.errors),
        error_types_remaining=sorted({error.type for error in validation.errors}),
        runtime_succeeded=run.succeeded,
        collateral=len(collateral),
        collateral_step_ids=collateral,
        removed_step_ids=sorted(broken_ids - repaired_ids),
        added_step_ids=sorted(repaired_ids - broken_ids),
        damaged_total=len(damaged),
        damaged_restored=len(restored),
    )


def repair_and_score(
    repairer: Repairer,
    *,
    reference_plan: AgentPlan,
    broken_plan: AgentPlan,
    task: AgentTask,
    damaged_step_ids: Iterable[str],
) -> tuple[AgentPlan, RepairScore]:
    """Run the whole loop: validate the broken plan, repair it, score the result."""
    validation = validate_plan(broken_plan, task)
    repaired = repairer.repair(broken_plan, validation, task)
    score = score_repair(
        repairer_name=repairer.name,
        reference_plan=reference_plan,
        broken_plan=broken_plan,
        repaired_plan=repaired,
        task=task,
        damaged_step_ids=damaged_step_ids,
    )
    return repaired, score


def _bodies(plan: AgentPlan) -> dict[str, str]:
    """Body hash per step id.

    An id carried by more than one step maps to a hash no single step can match, so a plan with
    duplicated ids never counts as unchanged by accident.
    """
    bodies: dict[str, str] = {}
    for step in plan.steps:
        bodies[step.id] = "<ambiguous>" if step.id in bodies else step_body_hash(step)
    return bodies


__all__ = ["RepairScore", "repair_and_score", "score_repair"]
