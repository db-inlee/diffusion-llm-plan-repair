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
    DUPLICATE_STEP,
    MISSING_STOP_CONDITION,
    STEP_DELETION,
    WRONG_ORDERING,
    WRONG_TOOL,
    CorruptionResult,
    InjectedError,
)
from plan_repair.schema.plan import AgentPlan, Step
from plan_repair.validation.paths import (
    input_from_path,
    step_path,
    stop_condition_path,
    tool_path,
)

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
        unknown_id = _free_id(broken, f"{removed}_x")
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


def inject_wrong_tool(
    plan: AgentPlan, *, step_id: str, new_tool: str | None = None
) -> CorruptionResult:
    """Replace the tool of ``step_id`` with a name the task does not offer.

    ``new_tool`` defaults to the original name with an ``_x`` suffix, the same convention the
    unknown dependency mode uses.
    """
    broken = plan.model_copy(deep=True)
    target = _find_step(broken, step_id)
    original_tool = target.tool
    target.tool = new_tool if new_tool is not None else f"{original_tool}_x"

    injected = InjectedError(
        corruption_type=WRONG_TOOL,
        damaged_step_ids=[step_id],
        damaged_paths=[tool_path(step_id)],
        detail={"original_tool": original_tool, "new_tool": target.tool},
    )
    return CorruptionResult(
        broken_plan=broken,
        injected=[injected],
        preserved_step_ids=_preserved(broken, damaged={step_id}),
    )


def inject_wrong_ordering(
    plan: AgentPlan, *, step_id: str, to_index: int | None = None
) -> CorruptionResult:
    """Move ``step_id`` earlier in the list so that it precedes a step it consumes.

    Only list positions change; ``input_from`` edges are untouched, so this can never introduce
    a dependency cycle (ordering is position-based, cycles are edge-based). The post-conditions
    below are contract defence, not an expected failure mode.

    ``to_index`` defaults to the position of the earliest step this one depends on.
    """
    broken = plan.model_copy(deep=True)
    positions = {step.id: index for index, step in enumerate(broken.steps)}
    if step_id not in positions:
        raise ValueError(f"no such step: {step_id!r}")

    target = _find_step(broken, step_id)
    dependencies = [dep for dep in target.input_from if dep in positions]
    if not dependencies:
        raise ValueError(f"step {step_id!r} has no dependency to be ordered against")

    from_index = positions[step_id]
    target_index = min(positions[dep] for dep in dependencies) if to_index is None else to_index
    if target_index >= from_index:
        raise ValueError(
            f"moving {step_id!r} to index {target_index} would not place it before its "
            f"dependencies (it currently sits at index {from_index})"
        )

    broken.steps.insert(target_index, broken.steps.pop(from_index))

    moved = {step.id: index for index, step in enumerate(broken.steps)}
    if not any(moved[dep] > moved[step_id] for dep in dependencies):
        raise ValueError(f"moving {step_id!r} did not create an ordering violation")

    injected = InjectedError(
        corruption_type=WRONG_ORDERING,
        damaged_step_ids=[step_id],
        damaged_paths=[step_path(step_id)],
        detail={"moved_from_index": from_index, "moved_to_index": target_index},
    )
    return CorruptionResult(
        broken_plan=broken,
        injected=[injected],
        preserved_step_ids=_preserved(broken, damaged={step_id}),
    )


def inject_duplicate_step(
    plan: AgentPlan, *, step_id: str, new_id: str | None = None
) -> CorruptionResult:
    """Insert a copy of ``step_id`` right after the original.

    The copy gets a fresh id (``<id>_dup``) by default, which the validator catches as identical
    content. Passing ``new_id=step_id`` produces an id clash instead; both are detectable.
    """
    broken = plan.model_copy(deep=True)
    original = _find_step(broken, step_id)
    duplicate = original.model_copy(deep=True)
    duplicate.id = new_id if new_id is not None else _free_id(broken, f"{step_id}_dup")

    index = next(i for i, step in enumerate(broken.steps) if step.id == step_id)
    broken.steps.insert(index + 1, duplicate)

    injected = InjectedError(
        corruption_type=DUPLICATE_STEP,
        damaged_step_ids=[duplicate.id],
        damaged_paths=[step_path(duplicate.id)],
        detail={"duplicate_of": step_id, "duplicate_id": duplicate.id},
    )
    return CorruptionResult(
        broken_plan=broken,
        injected=[injected],
        preserved_step_ids=_preserved(broken, damaged={duplicate.id}),
    )


def inject_missing_stop_condition(plan: AgentPlan) -> CorruptionResult:
    """Drop the plan's stop condition.

    The only plan-level corruption: it damages no step, so ``damaged_step_ids`` stays empty and
    the ground truth is a single path. Promoted from the Ticket 002 test helper because the
    multi-error combinations of Ticket 003 need it as a real injector.
    """
    broken = plan.model_copy(deep=True)
    original = broken.stop_condition
    broken.stop_condition = None

    injected = InjectedError(
        corruption_type=MISSING_STOP_CONDITION,
        damaged_step_ids=[],
        damaged_paths=[stop_condition_path()],
        detail={"original_stop_condition": original},
    )
    return CorruptionResult(
        broken_plan=broken,
        injected=[injected],
        preserved_step_ids=_preserved(broken, damaged=set()),
    )


def _find_step(plan: AgentPlan, step_id: str) -> Step:
    for step in plan.steps:
        if step.id == step_id:
            return step
    raise ValueError(f"no such step: {step_id!r}")


def _preserved(plan: AgentPlan, damaged: set[str]) -> list[str]:
    return [step.id for step in plan.steps if step.id not in damaged]


def _free_id(plan: AgentPlan, candidate: str) -> str:
    """Return ``candidate``, numbered if some step of ``plan`` already carries it."""
    known = {step.id for step in plan.steps}
    if candidate not in known:
        return candidate
    suffix = 2
    while f"{candidate}{suffix}" in known:
        suffix += 1
    return f"{candidate}{suffix}"


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
