"""Core plan validator — five structural checks.

1. schema validity      — the payload parses as an :class:`AgentPlan`
2. tool existence       — every ``step.tool`` is offered by the task
3. dependency existence — every ``input_from`` id refers to a step of the plan
4. DAG cycle            — the dependency graph is acyclic
5. ordering             — a step never consumes a step that appears later in the list

The checks are static only: nothing here judges whether a plan would *execute* successfully.

Ordering is skipped once a cycle is found. A topological order is undefined on a cyclic graph, so
an ordering violation there is a derived symptom of the cycle rather than an independent finding
— reporting it would make "an ordering error means the injector is buggy" (ticket risk section)
untrue for the cycle mode of the dependency corruption.
"""

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from plan_repair.schema.plan import AgentPlan
from plan_repair.schema.task import AgentTask
from plan_repair.validation.models import (
    DEP_CYCLE,
    ORDERING,
    SCHEMA,
    UNKNOWN_DEPENDENCY,
    UNKNOWN_TOOL,
    PlanValidationResult,
    ValidationError,
)
from plan_repair.validation.paths import input_from_path, path_from_loc, tool_path


def validate_plan(plan: AgentPlan | Mapping[str, Any], task: AgentTask) -> PlanValidationResult:
    """Validate ``plan`` against ``task``.

    A raw mapping is accepted so that parse failures can be reported as ``schema`` errors
    instead of raising.
    """
    if isinstance(plan, AgentPlan):
        parsed = plan
    else:
        try:
            parsed = AgentPlan.model_validate(plan)
        except PydanticValidationError as exc:
            return PlanValidationResult(valid=False, errors=_schema_errors(exc, plan))

    errors: list[ValidationError] = []
    errors.extend(_unknown_tools(parsed, task))
    errors.extend(_unknown_dependencies(parsed))
    cycles = _cycle_errors(parsed)
    errors.extend(cycles)
    if not cycles:
        errors.extend(_ordering_errors(parsed))
    return PlanValidationResult(valid=not errors, errors=errors)


def _schema_errors(exc: PydanticValidationError, raw_plan: Any) -> list[ValidationError]:
    errors: list[ValidationError] = []
    for detail in exc.errors():
        path, step_ids = path_from_loc(tuple(detail["loc"]), raw_plan)
        errors.append(
            ValidationError(
                type=SCHEMA,
                step_ids=step_ids,
                paths=[path],
                message=f"{path}: {detail['msg']}",
            )
        )
    return errors


def _unknown_tools(plan: AgentPlan, task: AgentTask) -> list[ValidationError]:
    available = task.tool_names()
    return [
        ValidationError(
            type=UNKNOWN_TOOL,
            step_ids=[step.id],
            paths=[tool_path(step.id)],
            message=f"step {step.id!r} uses tool {step.tool!r} which is not in available_tools",
        )
        for step in plan.steps
        if step.tool not in available
    ]


def _unknown_dependencies(plan: AgentPlan) -> list[ValidationError]:
    known = {step.id for step in plan.steps}
    errors: list[ValidationError] = []
    for step in plan.steps:
        missing = [dep for dep in step.input_from if dep not in known]
        if not missing:
            continue
        errors.append(
            ValidationError(
                type=UNKNOWN_DEPENDENCY,
                step_ids=[step.id],
                paths=[input_from_path(step.id)],
                message=(
                    f"step {step.id!r} depends on unknown step id(s): "
                    f"{', '.join(repr(dep) for dep in missing)}"
                ),
            )
        )
    return errors


def _cycle_errors(plan: AgentPlan) -> list[ValidationError]:
    """Report one error per cycle, carrying every step of that cycle.

    Cycles are reported as strongly connected components, so the member set is well defined even
    when several cyclic paths overlap (for example a branch that re-converges).
    """
    known = {step.id for step in plan.steps}
    # Edge direction: dependent -> dependency. Unknown ids are left out; they are already
    # reported by the dependency existence check and cannot close a cycle.
    edges = {step.id: [dep for dep in step.input_from if dep in known] for step in plan.steps}
    order = [step.id for step in plan.steps]

    errors: list[ValidationError] = []
    for component in _cyclic_components(order, edges):
        members = [step_id for step_id in order if step_id in component]
        errors.append(
            ValidationError(
                type=DEP_CYCLE,
                step_ids=members,
                paths=[input_from_path(step_id) for step_id in members],
                message=f"dependency cycle among steps: {', '.join(members)}",
            )
        )
    return errors


def _cyclic_components(
    order: Iterable[str], edges: Mapping[str, list[str]]
) -> list[frozenset[str]]:
    """Return the strongly connected components that contain a cycle (Tarjan, iterative)."""
    index_of: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    counter = 0
    components: list[frozenset[str]] = []

    for root in order:
        if root in index_of:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            node, child_index = work.pop()
            if child_index == 0:
                index_of[node] = low[node] = counter
                counter += 1
                stack.append(node)
                on_stack.add(node)
            neighbours = edges.get(node, [])
            recursed = False
            while child_index < len(neighbours):
                child = neighbours[child_index]
                child_index += 1
                if child not in index_of:
                    work.append((node, child_index))
                    work.append((child, 0))
                    recursed = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], index_of[child])
            if recursed:
                continue
            if low[node] == index_of[node]:
                component: set[str] = set()
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.add(member)
                    if member == node:
                        break
                if len(component) > 1 or node in edges.get(node, []):
                    components.append(frozenset(component))
            if work:
                parent, _ = work[-1]
                low[parent] = min(low[parent], low[node])
    return components


def _ordering_errors(plan: AgentPlan) -> list[ValidationError]:
    position = {step.id: index for index, step in enumerate(plan.steps)}
    errors: list[ValidationError] = []
    for index, step in enumerate(plan.steps):
        late = [dep for dep in step.input_from if position.get(dep, -1) > index]
        if not late:
            continue
        errors.append(
            ValidationError(
                type=ORDERING,
                step_ids=[step.id, *late],
                paths=[input_from_path(step.id)],
                message=(
                    f"step {step.id!r} consumes step(s) listed after it: "
                    f"{', '.join(repr(dep) for dep in late)}"
                ),
            )
        )
    return errors


__all__ = ["validate_plan"]
