"""Scoring a repair on the three axes of stage A, plus collateral.

Repair quality is not one number. A repairer can make the errors go away and wreck the rest of
the plan doing it, so quality and collateral are always reported together:

* **structure and meaning** — does the validator still find errors?
* **execution** — does the repaired plan run to the end?
* **collateral** — how many healthy steps did the repair change?

Collateral is reported by **kind**, not as one number. A single count hid the worst case: a
repairer that rewrote the plan under fresh ids left nothing to compare id-for-id and scored zero,
so the most destructive answer looked like the cleanest one. Splitting the count also removes the
need to weigh one kind of damage against another, which nobody can do non-arbitrarily.

A corruption records which steps it damaged; every other step of the reference plan is healthy.
For each healthy step, what became of it:

* :attr:`collateral_modified` — same id, different body. The work changed under the same name.
* :attr:`collateral_renamed` — the body survives somewhere under a different id. The work is
  intact; only its name was rewritten, which still breaks every reference to it.
* :attr:`collateral_removed` — the body is nowhere in the repaired plan. The work is gone.

Telling *renamed* from *removed* is what closes the blind spot, and it is decided by the id-free
hash from Ticket 002: if a healthy step's body reappears under another id it was renamed, and if
it does not, it was lost.

:attr:`spurious_added` sits apart on purpose. A new step may be a legitimate repair rather than
damage, so it is never folded into the damage counts — it is read as a marker of wholesale
rewriting. Steps that merely carry a renamed body are not counted here, since that event is
already reported as a rename.

**What the repairer is answerable for.** Damage the corruption did is never charged to the
repairer: a healthy step is only in scope if the repairer actually received it intact, so a step
the corruption deleted does not count as removed by whoever failed to bring it back. A damaged
step returning to its original form is a repair rather than collateral, and is reported on its
own as ``damaged_restored``.
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
    # collateral, by kind — never summed into one number here
    collateral_modified: int
    collateral_renamed: int
    collateral_removed: int
    modified_step_ids: list[str]
    renamed_step_ids: list[str]
    removed_step_ids: list[str]
    # not damage, but the signature of a rewrite
    spurious_added: int
    added_step_ids: list[str]
    # repair itself
    damaged_total: int
    damaged_restored: int

    @property
    def solved(self) -> bool:
        """Whether the plan came out clean on all three axes."""
        return self.valid and self.runtime_succeeded

    @property
    def collateral_total(self) -> int:
        """Every healthy step the repair damaged, of whatever kind.

        A convenience for reading a table, not a verdict: the kinds are reported separately
        because how a step was damaged matters, and weighing them against each other is a
        decision for the comparison ticket.
        """
        return self.collateral_modified + self.collateral_renamed + self.collateral_removed


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
    broken_bodies = _bodies(broken_plan)
    repaired_bodies = _bodies(repaired_plan)
    surviving_bodies = set(repaired_bodies.values())

    # Only steps the repairer actually received intact are its responsibility: a step the
    # corruption already deleted cannot be removed again by whoever did not restore it.
    in_scope = [
        step_id
        for step_id, body in reference_bodies.items()
        if step_id not in damaged and broken_bodies.get(step_id) == body
    ]

    modified: list[str] = []
    renamed: list[str] = []
    removed: list[str] = []
    for step_id in in_scope:
        body = reference_bodies[step_id]
        if step_id in repaired_bodies:
            if repaired_bodies[step_id] != body:
                modified.append(step_id)
        elif body in surviving_bodies:
            renamed.append(step_id)
        else:
            removed.append(step_id)

    # A step carrying a healthy body under a new id is the rename already counted above, not an
    # invention; counting it here as well would report one event twice.
    renamed_bodies = {reference_bodies[step_id] for step_id in renamed}
    added = sorted(
        step_id
        for step_id, body in repaired_bodies.items()
        if step_id not in reference_bodies and body not in renamed_bodies
    )

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
        collateral_modified=len(modified),
        collateral_renamed=len(renamed),
        collateral_removed=len(removed),
        modified_step_ids=modified,
        renamed_step_ids=renamed,
        removed_step_ids=removed,
        spurious_added=len(added),
        added_step_ids=added,
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
