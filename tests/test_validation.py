"""Unit coverage for all five checks.

The two corruption injectors only produce dependency and cycle errors, so the remaining checks
are exercised here by hand-editing the reference pipeline. Every implemented check is verified.
"""

from plan_repair.data import load_reference
from plan_repair.validation import (
    DEP_CYCLE,
    ORDERING,
    SCHEMA,
    UNKNOWN_DEPENDENCY,
    UNKNOWN_TOOL,
    input_from_path,
    step_path,
    tool_path,
    validate_plan,
)


def step(plan, step_id):
    return next(s for s in plan.steps if s.id == step_id)


def test_path_format_is_pinned_to_the_scenario_document():
    """Injector and validator share the path helper, so the literal form is pinned here."""
    assert step_path("enrich") == "$.steps[?enrich]"
    assert input_from_path("enrich") == "$.steps[?enrich].input_from"
    assert tool_path("join") == "$.steps[?join].tool"


def test_reference_plan_is_valid():
    task, plan = load_reference()
    result = validate_plan(plan, task)
    assert result.valid
    assert result.errors == []


def test_unknown_tool_is_detected():
    task, plan = load_reference()
    step(plan, "join").tool = "join_x"

    result = validate_plan(plan, task)
    errors = result.errors_of_type(UNKNOWN_TOOL)

    assert not result.valid
    assert len(errors) == 1
    assert errors[0].step_ids == ["join"]
    assert errors[0].paths == [tool_path("join")]
    assert result.errors_of_type(ORDERING) == []


def test_unknown_dependency_is_detected():
    task, plan = load_reference()
    step(plan, "enrich").input_from = ["join", "l_api_x"]

    result = validate_plan(plan, task)
    errors = result.errors_of_type(UNKNOWN_DEPENDENCY)

    assert not result.valid
    assert len(errors) == 1
    assert errors[0].step_ids == ["enrich"]
    assert errors[0].paths == [input_from_path("enrich")]
    assert "l_api_x" in errors[0].message


def test_dependency_cycle_is_detected_with_full_component():
    task, plan = load_reference()
    step(plan, "l_csv").input_from = ["report"]

    result = validate_plan(plan, task)
    errors = result.errors_of_type(DEP_CYCLE)

    assert not result.valid
    assert len(errors) == 1
    members = errors[0].step_ids
    assert {"l_csv", "report"} <= set(members)
    # The branch that re-converges is part of the cycle...
    assert {"stat", "corr", "pivot", "viz", "interp"} <= set(members)
    # ...while steps that cannot reach l_csv again are not.
    assert {"l_db", "l_api", "n_db"}.isdisjoint(members)
    assert errors[0].paths == [input_from_path(member) for member in members]


def test_self_loop_is_detected():
    task, plan = load_reference()
    step(plan, "agg").input_from = ["enrich", "agg"]

    errors = validate_plan(plan, task).errors_of_type(DEP_CYCLE)

    assert len(errors) == 1
    assert errors[0].step_ids == ["agg"]


def test_ordering_violation_is_detected():
    task, plan = load_reference()
    join = step(plan, "join")
    plan.steps.remove(join)
    plan.steps.insert([s.id for s in plan.steps].index("n_csv"), join)

    result = validate_plan(plan, task)
    errors = result.errors_of_type(ORDERING)

    assert not result.valid
    assert len(errors) == 1
    assert errors[0].step_ids == ["join", "n_csv", "n_db"]
    assert errors[0].paths == [input_from_path("join")]
    assert result.errors_of_type(DEP_CYCLE) == []


def test_ordering_is_skipped_when_a_cycle_exists():
    """A cyclic graph has no topological order, so an ordering error there is noise."""
    task, plan = load_reference()
    step(plan, "l_csv").input_from = ["report"]

    result = validate_plan(plan, task)

    assert result.errors_of_type(DEP_CYCLE)
    assert result.errors_of_type(ORDERING) == []


def test_schema_error_is_reported_for_raw_payload():
    task, plan = load_reference()
    raw = plan.model_dump()
    del raw["steps"][11]["tool"]

    result = validate_plan(raw, task)
    errors = result.errors_of_type(SCHEMA)

    assert not result.valid
    assert len(errors) == 1
    assert errors[0].step_ids == ["join"]
    assert errors[0].paths == [f"{step_path('join')}.tool"]


def test_schema_error_falls_back_to_positional_path():
    task, _ = load_reference()

    result = validate_plan({"steps": []}, task)
    errors = result.errors_of_type(SCHEMA)

    assert not result.valid
    assert errors[0].paths == ["$.goal"]
    assert errors[0].step_ids == []


def test_unknown_dependency_and_unknown_tool_are_reported_together():
    task, plan = load_reference()
    step(plan, "join").tool = "join_x"
    step(plan, "enrich").input_from = ["join", "l_api_x"]

    result = validate_plan(plan, task)

    assert {error.type for error in result.errors} == {UNKNOWN_TOOL, UNKNOWN_DEPENDENCY}
