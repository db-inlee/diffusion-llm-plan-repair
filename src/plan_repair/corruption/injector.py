"""Corruption injectors — two structural operations.

Contracts held by both operations:

* the input plan is never mutated (the corruption is applied to a deep copy);
* the ids of surviving steps are never rewritten (step id preservation policy);
* neither operation is allowed to create an ordering violation — the validator reporting one
  is a signal that this module is buggy, not that the plan is wrong;
* ``InjectedError`` only points at steps that exist in the broken plan, because that is what a
  validator can possibly report. A deleted step is recorded in ``detail`` instead.
"""

from typing import Any

from plan_repair.schema.corruption import (
    BROKEN_DEPENDENCY,
    STEP_DELETION,
    CorruptionResult,
    InjectedError,
)
from plan_repair.schema.plan import AgentPlan, Step
from plan_repair.validation.paths import input_from_path

UNKNOWN_MODE = "unknown"
CYCLE_MODE = "cycle"


def inject_broken_dependency(
    plan: AgentPlan,
    *,
    step_id: str,
    mode: str = UNKNOWN_MODE,
    dep: str | None = None,
    cycle_with: str | None = None,
) -> CorruptionResult:
    """Break one dependency edge of ``step_id``.

    ``mode="unknown"`` rewrites one edge to an id that no step carries (detected as
    ``unknown_dependency``). ``mode="cycle"`` adds ``cycle_with`` — a step that transitively
    depends on ``step_id`` — as a new edge, closing a loop (detected as ``dep_cycle``).

    ``dep`` selects which edge to rewrite in unknown mode; it defaults to the last edge.
    """
    broken = plan.model_copy(deep=True)
    target = _find_step(broken, step_id)
    original_input_from = list(target.input_from)
    detail: dict[str, Any]

    if mode == UNKNOWN_MODE:
        if dep is None and not original_input_from:
            raise ValueError(f"step {step_id!r} has no dependency edge to break")
        removed = dep if dep is not None else original_input_from[-1]
        if removed not in original_input_from:
            raise ValueError(f"step {step_id!r} does not depend on {removed!r}")
        unknown_id = _unknown_id(broken, removed)
        target.input_from = [
            unknown_id if edge == removed else edge for edge in original_input_from
        ]
        detail = {
            "mode": UNKNOWN_MODE,
            "original_input_from": original_input_from,
            "removed_dep": removed,
            "unknown_dep": unknown_id,
        }
    elif mode == CYCLE_MODE:
        if cycle_with is None:
            raise ValueError("cycle mode requires cycle_with")
        _find_step(broken, cycle_with)
        if cycle_with in original_input_from:
            raise ValueError(f"step {step_id!r} already depends on {cycle_with!r}")
        if not _depends_on(broken, dependent=cycle_with, dependency=step_id):
            raise ValueError(
                f"adding {cycle_with!r} to {step_id!r} would not close a cycle: "
                f"{cycle_with!r} does not depend on {step_id!r}"
            )
        target.input_from = [*original_input_from, cycle_with]
        detail = {
            "mode": CYCLE_MODE,
            "original_input_from": original_input_from,
            "added_dep": cycle_with,
        }
    else:
        raise ValueError(f"unknown corruption mode: {mode!r}")

    injected = InjectedError(
        corruption_type=BROKEN_DEPENDENCY,
        damaged_step_ids=[step_id],
        damaged_paths=[input_from_path(step_id)],
        detail=detail,
    )
    return CorruptionResult(
        broken_plan=broken,
        injected=[injected],
        preserved_step_ids=_preserved(broken, damaged={step_id}),
    )


def inject_step_deletion(plan: AgentPlan, *, step_id: str) -> CorruptionResult:
    """Delete ``step_id`` from the plan.

    The steps that consumed it keep their now-dangling reference, which is what the validator
    detects. ``damaged_step_ids`` therefore lists those downstream steps only — the deleted step
    has no location in the broken plan — and the deleted step is kept in ``detail`` for later
    recoverability judgement.
    """
    broken = plan.model_copy(deep=True)
    deleted = _find_step(broken, step_id)
    broken.steps = [step for step in broken.steps if step.id != step_id]

    dependents = [step.id for step in broken.steps if step_id in step.input_from]
    injected = InjectedError(
        corruption_type=STEP_DELETION,
        damaged_step_ids=dependents,
        damaged_paths=[input_from_path(dependent) for dependent in dependents],
        detail={
            "deleted_step_id": step_id,
            "deleted_step": deleted.model_dump(),
            "deleted_index": next(
                index for index, step in enumerate(plan.steps) if step.id == step_id
            ),
        },
    )
    return CorruptionResult(
        broken_plan=broken,
        injected=[injected],
        preserved_step_ids=_preserved(broken, damaged=set(dependents)),
    )


def _find_step(plan: AgentPlan, step_id: str) -> Step:
    for step in plan.steps:
        if step.id == step_id:
            return step
    raise ValueError(f"no such step: {step_id!r}")


def _preserved(plan: AgentPlan, damaged: set[str]) -> list[str]:
    return [step.id for step in plan.steps if step.id not in damaged]


def _unknown_id(plan: AgentPlan, base: str) -> str:
    """Return an id derived from ``base`` that no step of ``plan`` carries."""
    known = {step.id for step in plan.steps}
    candidate = f"{base}_x"
    suffix = 2
    while candidate in known:
        candidate = f"{base}_x{suffix}"
        suffix += 1
    return candidate


def _depends_on(plan: AgentPlan, *, dependent: str, dependency: str) -> bool:
    """Whether ``dependent`` transitively consumes ``dependency``."""
    edges = {step.id: step.input_from for step in plan.steps}
    seen: set[str] = set()
    stack = list(edges.get(dependent, []))
    while stack:
        current = stack.pop()
        if current == dependency:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(edges.get(current, []))
    return False
