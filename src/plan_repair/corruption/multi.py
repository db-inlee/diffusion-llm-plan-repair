"""Multi-error corruption — several injectors applied to one plan.

Each corruption keeps its own :class:`InjectedError`, so the ground truth stays attributable:
which error came from which injection is never lost in the combination.

**Interference guard.** Two corruptions that modify the same step produce a result nobody can
score — the second one rewrites what the first one damaged — so overlapping targets are rejected
before anything is applied. The guard looks at what each corruption *modifies*, not at what it
affects downstream: deleting two different steps whose consequences meet at a third step (the
deleted-join / deleted-co pair) is a legitimate interaction this ticket wants to measure, not
interference.
"""

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from plan_repair.corruption.injector import (
    inject_broken_dependency,
    inject_duplicate_step,
    inject_missing_stop_condition,
    inject_step_deletion,
    inject_wrong_ordering,
    inject_wrong_tool,
)
from plan_repair.schema.corruption import (
    BROKEN_DEPENDENCY,
    DUPLICATE_STEP,
    MISSING_STOP_CONDITION,
    STEP_DELETION,
    WRONG_ORDERING,
    WRONG_TOOL,
    CorruptionResult,
)
from plan_repair.schema.plan import AgentPlan


class CorruptionSpec(BaseModel):
    """One corruption of a combination, named declaratively so it can be checked up front."""

    model_config = ConfigDict(extra="forbid")

    corruption_type: str
    step_id: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class _Operation(NamedTuple):
    inject: Callable[..., CorruptionResult]
    # Option keys naming a step that must already exist; they count as touched by the corruption.
    step_options: tuple[str, ...] = ()
    plan_level: bool = False


_OPERATIONS: dict[str, _Operation] = {
    BROKEN_DEPENDENCY: _Operation(inject_broken_dependency, step_options=("dep", "cycle_with")),
    STEP_DELETION: _Operation(inject_step_deletion),
    WRONG_TOOL: _Operation(inject_wrong_tool),
    WRONG_ORDERING: _Operation(inject_wrong_ordering),
    # ``new_id`` names the copy, not an existing step, so it is not a touched step.
    DUPLICATE_STEP: _Operation(inject_duplicate_step),
    MISSING_STOP_CONDITION: _Operation(inject_missing_stop_condition, plan_level=True),
}


def inject_multi(plan: AgentPlan, specs: Sequence[CorruptionSpec]) -> CorruptionResult:
    """Apply every spec to ``plan`` and return the combined result.

    ``plan`` is left untouched: each injector works on its own copy, so the corruptions chain
    through the intermediate broken plans without ever reaching the caller's object.
    """
    if len(specs) < 2:
        raise ValueError("a multi-error corruption needs at least two corruptions")

    _guard_interference(plan, specs)

    broken = plan
    injected = []
    for spec in specs:
        result = _apply(broken, spec)
        broken = result.broken_plan
        injected.extend(result.injected)

    damaged = {step_id for error in injected for step_id in error.damaged_step_ids}
    return CorruptionResult(
        broken_plan=broken,
        injected=injected,
        preserved_step_ids=[step.id for step in broken.steps if step.id not in damaged],
    )


def touched_steps(spec: CorruptionSpec) -> set[str]:
    """Steps this corruption modifies or needs to exist unchanged."""
    operation = _operation(spec)
    touched = set() if spec.step_id is None else {spec.step_id}
    for key in operation.step_options:
        value = spec.options.get(key)
        if isinstance(value, str):
            touched.add(value)
    return touched


def _guard_interference(plan: AgentPlan, specs: Sequence[CorruptionSpec]) -> None:
    known = {step.id for step in plan.steps}
    claimed: dict[str, int] = {}
    for position, spec in enumerate(specs):
        for step_id in touched_steps(spec):
            if step_id not in known:
                raise ValueError(f"no such step: {step_id!r}")
            if step_id in claimed:
                raise ValueError(
                    f"corruptions {claimed[step_id]} and {position} both touch step "
                    f"{step_id!r}; combined corruptions must target distinct steps"
                )
            claimed[step_id] = position


def _apply(plan: AgentPlan, spec: CorruptionSpec) -> CorruptionResult:
    operation = _operation(spec)
    if operation.plan_level:
        if spec.step_id is not None:
            raise ValueError(f"{spec.corruption_type!r} is plan level and takes no step_id")
        return operation.inject(plan, **spec.options)
    if spec.step_id is None:
        raise ValueError(f"{spec.corruption_type!r} needs a step_id")
    return operation.inject(plan, step_id=spec.step_id, **spec.options)


def _operation(spec: CorruptionSpec) -> _Operation:
    try:
        return _OPERATIONS[spec.corruption_type]
    except KeyError:
        raise ValueError(f"unknown corruption type: {spec.corruption_type!r}") from None


__all__ = ["CorruptionSpec", "inject_multi", "touched_steps"]
