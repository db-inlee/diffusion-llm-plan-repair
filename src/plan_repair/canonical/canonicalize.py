"""Deterministic structural normalization of a plan.

Scope is strictly structural: no semantic similarity, no argument value normalization.

Two rules that are easy to confuse:

* ``input_from`` is a *set* of dependency edges — sorted, because the order carries no meaning.
* ``steps`` is a *sequence* — never sorted, because the list order is what the ordering
  validator checks.
"""

import hashlib
import json
from typing import Any

from plan_repair.schema.plan import AgentPlan, Step

_JSON_ARGS: dict[str, Any] = {
    "sort_keys": True,  # recursively sorts every dict key, including nested arguments
    "ensure_ascii": False,
    "separators": (",", ":"),
}


def canonical_step(step: Step) -> dict[str, Any]:
    """Return the canonical mapping of a single step."""
    return {
        "id": step.id,
        "tool": step.tool,
        "arguments": step.arguments,
        "input_from": sorted(step.input_from),
    }


def step_hash(step: Step) -> str:
    """Return the canonical hash of a single step."""
    return _sha256(json.dumps(canonical_step(step), **_JSON_ARGS))


def canonicalize(plan: AgentPlan) -> tuple[str, dict[str, str]]:
    """Return ``(canonical_json, step_hashes)`` for ``plan``.

    The output is byte-stable across runs: plans differing only in argument key order or
    ``input_from`` order canonicalize identically, while a different step order does not.
    """
    steps = [canonical_step(step) for step in plan.steps]
    document = {
        "goal": plan.goal,
        "steps": steps,
        "stop_condition": plan.stop_condition,
    }
    canonical_json = json.dumps(document, **_JSON_ARGS)
    step_hashes = {step["id"]: _sha256(json.dumps(step, **_JSON_ARGS)) for step in steps}
    return canonical_json, step_hashes


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
