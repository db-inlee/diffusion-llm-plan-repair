"""Rule-based repairer — the floor a model repairer has to clear.

This is not meant to repair well. It is meant to establish how far mechanical rules get, so that
a model's contribution can be read as the distance above this line. A baseline that quietly
repaired everything would erase the reason for the models that follow.

The rules are confined to repairs that need no judgement about *intent*. The project plan draws
that line (section 7.5): correcting references to ids that do not exist, deleting exact
duplicates, restoring topological order and filling in missing containers are mechanical;
designing a step to cover a missing requirement, choosing a tool by meaning, or inventing
arguments are not — those belong to the model repairers.

Handled:

* ``unknown_dependency`` — drop the references that point at no step. Honestly partial: the step
  the plan meant to consume is usually one that was deleted, and no rule can bring back its tool
  or its arguments. The dependency stops dangling; the work it stood for stays gone.
* ``duplicate_step`` — delete the later copy of two steps doing identical work.
* ``ordering`` — reorder the list into a topological order, keeping the original relative order
  wherever the dependencies allow it.
* ``missing_stop_condition`` — fill it from the terminal step.

Declined, on purpose (see :data:`DECLINED_ERROR_TYPES`): unknown tools, dependency cycles,
dangling steps and uncovered requirements. Each of them needs a decision the rules cannot make —
which tool was meant, which edge of the cycle was the wrong one, which step should consume the
orphan, what a step covering the requirement would do. Those are the cases the model repairers
exist for, and leaving them untouched is what keeps the baseline honest.
"""

from plan_repair.canonical.canonicalize import step_body_hash
from plan_repair.schema.plan import AgentPlan, Step
from plan_repair.schema.task import AgentTask
from plan_repair.validation.models import (
    DANGLING_STEP,
    DEP_CYCLE,
    DUPLICATE_STEP,
    MISSING_EVIDENCE,
    MISSING_OPERATION,
    MISSING_STOP_CONDITION,
    ORDERING,
    SCHEMA,
    UNKNOWN_DEPENDENCY,
    UNKNOWN_TOOL,
    PlanValidationResult,
)

HANDLED_ERROR_TYPES = frozenset(
    {UNKNOWN_DEPENDENCY, DUPLICATE_STEP, ORDERING, MISSING_STOP_CONDITION}
)
DECLINED_ERROR_TYPES = frozenset(
    {SCHEMA, UNKNOWN_TOOL, DEP_CYCLE, DANGLING_STEP, MISSING_EVIDENCE, MISSING_OPERATION}
)


class DeterministicRepairer:
    """Applies the mechanical repairs listed in the module docstring, and nothing else."""

    name = "deterministic"

    def repair(
        self,
        broken_plan: AgentPlan,
        validation: PlanValidationResult,
        task: AgentTask,
    ) -> AgentPlan:
        plan = broken_plan.model_copy(deep=True)
        reported = {error.type for error in validation.errors}

        # Duplicates go first: a removed copy is one less step for the later rules to move around.
        if DUPLICATE_STEP in reported:
            _drop_duplicates(plan, validation)
        if UNKNOWN_DEPENDENCY in reported:
            _drop_unknown_dependencies(plan, validation)
        # A cyclic graph has no topological order to restore, and the cycle itself is declined.
        if ORDERING in reported and DEP_CYCLE not in reported:
            _reorder_topologically(plan)
        if MISSING_STOP_CONDITION in reported:
            _fill_stop_condition(plan)
        return plan


def _drop_duplicates(plan: AgentPlan, validation: PlanValidationResult) -> None:
    """Keep the first of each group of steps doing identical work and drop the rest.

    Only content duplicates are removed — a complete duplicate is redundant whatever it is called,
    so this covers a copy under a fresh id and a copy under the same id alike. An id clash between
    steps doing *different* work is left alone: choosing which of them keeps the name is a
    decision about intent, not a mechanical one.
    """
    doomed: set[int] = set()
    for error in validation.errors_of_type(DUPLICATE_STEP):
        if error.detail.get("kind") != "identical_content":
            continue
        group = set(error.step_ids)
        members = [index for index, step in enumerate(plan.steps) if step.id in group]
        if not members:
            continue
        # Match on the body too: an id in the group may also be worn by an unrelated step.
        body = step_body_hash(plan.steps[members[0]])
        identical = [index for index in members if step_body_hash(plan.steps[index]) == body]
        doomed.update(identical[1:])
    if not doomed:
        return
    plan.steps = [step for index, step in enumerate(plan.steps) if index not in doomed]


def _drop_unknown_dependencies(plan: AgentPlan, validation: PlanValidationResult) -> None:
    """Drop references that name no step of the plan."""
    known = {step.id for step in plan.steps}
    targets = {
        step_id
        for error in validation.errors_of_type(UNKNOWN_DEPENDENCY)
        for step_id in error.step_ids
    }
    for step in plan.steps:
        if step.id in targets:
            step.input_from = [dep for dep in step.input_from if dep in known]


def _reorder_topologically(plan: AgentPlan) -> None:
    """Sort the steps so every step follows what it consumes, disturbing the order minimally.

    At each position the earliest still-listed step whose dependencies are already satisfied is
    taken. Emitting one step at a time rather than a whole ready batch is what keeps the result
    close to the order the plan came in with: a plan with a single step out of place comes back
    with only that step moved, and a plan already in order comes back untouched.
    """
    known = {step.id for step in plan.steps}
    waiting = {
        step.id: {dep for dep in step.input_from if dep in known and dep != step.id}
        for step in plan.steps
    }
    ordered: list[Step] = []
    remaining = list(plan.steps)
    settled: set[str] = set()

    while remaining:
        for index, step in enumerate(remaining):
            if waiting[step.id] <= settled:
                ordered.append(step)
                settled.add(step.id)
                del remaining[index]
                break
        else:
            return  # not a DAG after all; leave the plan as it is rather than guess
    plan.steps = ordered


def _fill_stop_condition(plan: AgentPlan) -> None:
    """State the stop condition in terms of the terminal step.

    Derived from the plan's own structure rather than written: the terminal step is where the plan
    ends, so reaching it is what stopping means. No natural language is invented, and nothing in
    the project reads this text for meaning.
    """
    if not plan.steps:
        return
    plan.stop_condition = f"terminal step {plan.steps[-1].id} completed"


__all__ = ["DECLINED_ERROR_TYPES", "HANDLED_ERROR_TYPES", "DeterministicRepairer"]
