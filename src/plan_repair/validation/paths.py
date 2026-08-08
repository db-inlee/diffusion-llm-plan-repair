"""JSONPath strings — single source of truth shared by the validator and the injector.

Paths are **id-based** (``$.steps[?enrich].input_from``) rather than index-based. Step deletion
shifts list indices, so an index-based path would name a different step depending on when it was
produced; the id-based form stays valid on the broken plan and is what downstream error-to-span
mapping needs.
"""

from typing import Any, NamedTuple

PLAN_PATH = "$"
STEPS_PATH = "$.steps"
STOP_CONDITION_PATH = "$.stop_condition"


def step_path(step_id: str) -> str:
    """Path of the step with ``step_id`` inside the (broken) plan."""
    return f"$.steps[?{step_id}]"


def input_from_path(step_id: str) -> str:
    """Path of the dependency edge list of ``step_id``."""
    return f"{step_path(step_id)}.input_from"


def tool_path(step_id: str) -> str:
    """Path of the tool field of ``step_id``."""
    return f"{step_path(step_id)}.tool"


def stop_condition_path() -> str:
    """Path of the plan-level stop condition."""
    return STOP_CONDITION_PATH


def required_evidence_path(name: str) -> str:
    """Path of a required evidence entry — a location in the **task**, not the plan.

    Coverage errors report an absence: no step claims the requirement, so there is nothing in the
    plan to point at. Naming the unmet requirement is what a repairer needs to know.
    """
    return f"$.required_evidence[?{name}]"


def required_operation_path(name: str) -> str:
    """Path of a required operation entry — a location in the task (see above)."""
    return f"$.required_operations[?{name}]"


class ParsedPath(NamedTuple):
    """A path read back: which step it names, and which field of it, if any.

    ``field`` is ``None`` when the path points at a whole step (``$.steps[?join]``) and
    ``step_id`` is ``None`` when it points somewhere else entirely — the plan's stop condition,
    or a requirement of the task. Both are locations no field mask can narrow.
    """

    step_id: str | None
    field: str | None


# The fields a path can name. Kept explicit rather than "whatever follows the dot" so that an
# unexpected suffix is reported as unparsed instead of silently masking something unintended.
MASKABLE_FIELDS = ("tool", "input_from")


def parse_path(path: str) -> ParsedPath:
    """Read a step path back into the step and field it names.

    The inverse of :func:`step_path` and friends. Paths are generated in one place and, until
    now, only ever generated — field-level masking is what makes reading them back necessary.
    """
    if not path.startswith("$.steps[?"):
        return ParsedPath(None, None)
    closing = path.find("]", len("$.steps[?"))
    if closing < 0:
        return ParsedPath(None, None)
    step_id = path[len("$.steps[?") : closing]
    remainder = path[closing + 1 :]
    if not remainder:
        return ParsedPath(step_id, None)
    field = remainder.removeprefix(".")
    return ParsedPath(step_id, field if field in MASKABLE_FIELDS else None)


def path_from_loc(loc: tuple[Any, ...], raw_plan: Any) -> tuple[str, list[str]]:
    """Translate a pydantic error location into ``(path, step_ids)``.

    Schema errors are raised before the plan is a model, so the step id is recovered from the raw
    payload when possible; otherwise the path falls back to the positional form.
    """
    step_id = _raw_step_id(loc, raw_plan)
    if step_id is None:
        rendered = PLAN_PATH + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}" for part in loc
        )
        return rendered, []
    tail = "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in loc[2:])
    return step_path(step_id) + tail, [step_id]


def _raw_step_id(loc: tuple[Any, ...], raw_plan: Any) -> str | None:
    if len(loc) < 2 or loc[0] != "steps" or not isinstance(loc[1], int):
        return None
    if not isinstance(raw_plan, dict):
        return None
    steps = raw_plan.get("steps")
    if not isinstance(steps, list) or not 0 <= loc[1] < len(steps):
        return None
    step = steps[loc[1]]
    if not isinstance(step, dict):
        return None
    step_id = step.get("id")
    return step_id if isinstance(step_id, str) else None
