"""Core plan validator — eight structural checks.

1. schema validity      — the payload parses as an :class:`AgentPlan`
2. tool existence       — every ``step.tool`` is offered by the task
3. dependency existence — every ``input_from`` id refers to a step of the plan
4. DAG cycle            — the dependency graph is acyclic
5. ordering             — a step never consumes a step that appears later in the list
6. duplicate step       — no id clash and no two steps doing identical work
7. dangling step        — no step whose output nobody consumes, the plan terminal aside
8. stop condition       — the plan states when it is done
9. coverage             — every required evidence and operation is claimed by some step

Checks 1-8 are structural; check 9 is the one semantic check: it asks whether the plan does what
the task asked for, not whether it is well formed. Nothing here judges whether a plan would
*execute* successfully — that is what the runtime package is for.
Every check fires independently; the cycle/ordering hierarchy below is the single exception.

Ordering is skipped once a cycle is found. A topological order is undefined on a cyclic graph, so
an ordering violation there is a derived symptom of the cycle rather than an independent finding
— reporting it would make "an ordering error means the injector is buggy" (ticket risk section)
untrue for the cycle mode of the dependency corruption.
"""

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from plan_repair.canonical.canonicalize import step_body_hash
from plan_repair.schema.plan import AgentPlan
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
    ValidationError,
)
from plan_repair.validation.paths import (
    input_from_path,
    path_from_loc,
    required_evidence_path,
    required_operation_path,
    step_path,
    stop_condition_path,
    tool_path,
)


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
    errors.extend(_duplicate_errors(parsed))
    errors.extend(_dangling_errors(parsed))
    errors.extend(_missing_stop_condition(parsed))
    errors.extend(_coverage_errors(parsed, task))
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
                # What is wrong is where the step sits, not the edge itself — so the path points
                # at the step. The ordering corruption records the same location.
                paths=[step_path(step.id)],
                message=(
                    f"step {step.id!r} consumes step(s) listed after it: "
                    f"{', '.join(repr(dep) for dep in late)}"
                ),
            )
        )
    return errors


def _duplicate_errors(plan: AgentPlan) -> list[ValidationError]:
    """Report id clashes and steps doing identical work.

    Identical work is compared with the id stripped out (:func:`step_body_hash`), because a
    duplicate is usually inserted under a fresh id. Sharing a *tool* is not duplication — a
    pipeline may legitimately call the same tool on different branches — so the whole step body
    (tool, arguments, dependencies) has to match.
    """
    errors: list[ValidationError] = []

    positions: dict[str, list[int]] = {}
    for index, step in enumerate(plan.steps):
        positions.setdefault(step.id, []).append(index)
    for step_id, indices in positions.items():
        if len(indices) > 1:
            errors.append(
                ValidationError(
                    type=DUPLICATE_STEP,
                    step_ids=[step_id],
                    paths=[step_path(step_id)],
                    message=f"step id {step_id!r} is used by {len(indices)} steps",
                    detail={"kind": "duplicate_id", "group": [step_id], "count": len(indices)},
                )
            )

    bodies: dict[str, list[str]] = {}
    for step in plan.steps:
        bodies.setdefault(step_body_hash(step), []).append(step.id)
    for group in bodies.values():
        if len(group) > 1:
            errors.append(
                ValidationError(
                    type=DUPLICATE_STEP,
                    step_ids=group,
                    paths=[step_path(step_id) for step_id in group],
                    message=f"steps are identical apart from their id: {', '.join(group)}",
                    detail={"kind": "identical_content", "group": group},
                )
            )
    return errors


def _dangling_errors(plan: AgentPlan) -> list[ValidationError]:
    """Report steps whose output nobody consumes.

    The plan terminal is defined structurally as the last step of the list — the plan's final
    output — and never counts as dangling. Naming terminals per domain (``report``, ``fmt``, ...)
    would hardcode domain knowledge and break on the next pipeline.
    """
    if not plan.steps:
        return []
    terminal = plan.steps[-1].id
    consumed = {dep for step in plan.steps for dep in step.input_from}
    return [
        ValidationError(
            type=DANGLING_STEP,
            step_ids=[step.id],
            paths=[step_path(step.id)],
            message=f"step {step.id!r} produces output that no step consumes",
        )
        for step in plan.steps
        if step.id != terminal and step.id not in consumed
    ]


def _missing_stop_condition(plan: AgentPlan) -> list[ValidationError]:
    if plan.stop_condition is not None and plan.stop_condition.strip():
        return []
    return [
        ValidationError(
            type=MISSING_STOP_CONDITION,
            step_ids=[],
            paths=[stop_condition_path()],
            message="plan has no stop_condition",
        )
    ]


def _coverage_errors(plan: AgentPlan, task: AgentTask) -> list[ValidationError]:
    """Report requirements of the task that no step claims to satisfy.

    The only operation here is a set difference between what the steps declare in ``produces``
    and what the task requires: no requirement name, tool name or step id is known to this code,
    which is what lets the same check score an unrelated pipeline.
    """
    produced = {tag for step in plan.steps for tag in step.produces}
    errors: list[ValidationError] = []
    for name in task.required_evidence:
        if name not in produced:
            errors.append(
                ValidationError(
                    type=MISSING_EVIDENCE,
                    step_ids=[],
                    paths=[required_evidence_path(name)],
                    message=f"no step produces required evidence {name!r}",
                )
            )
    for name in task.required_operations:
        if name not in produced:
            errors.append(
                ValidationError(
                    type=MISSING_OPERATION,
                    step_ids=[],
                    paths=[required_operation_path(name)],
                    message=f"no step performs required operation {name!r}",
                )
            )
    return errors


__all__ = ["validate_plan"]
