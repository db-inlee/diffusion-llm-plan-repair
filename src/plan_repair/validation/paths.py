"""JSONPath strings — single source of truth shared by the validator and the injector.

Paths are **id-based** (``$.steps[?enrich].input_from``) rather than index-based. Step deletion
shifts list indices, so an index-based path would name a different step depending on when it was
produced; the id-based form stays valid on the broken plan and is what downstream error-to-span
mapping needs.
"""

from typing import Any

PLAN_PATH = "$"
STEPS_PATH = "$.steps"


def step_path(step_id: str) -> str:
    """Path of the step with ``step_id`` inside the (broken) plan."""
    return f"$.steps[?{step_id}]"


def input_from_path(step_id: str) -> str:
    """Path of the dependency edge list of ``step_id``."""
    return f"{step_path(step_id)}.input_from"


def tool_path(step_id: str) -> str:
    """Path of the tool field of ``step_id``."""
    return f"{step_path(step_id)}.tool"


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
