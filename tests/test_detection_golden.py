"""Closed loop on the domain B pipeline: inject -> validate -> compare against ground truth.

These are the B-1..B-4 (Ticket 001) and B-5..B-8 (Ticket 002) corruption golden cases. Detection
recall is measured as containment (every injected step/path is reported by the validator), not as
set equality: a cycle is legitimately reported with its whole component while the injector only
claims the step it modified.
"""

import pytest

from plan_repair.corruption import (
    CYCLE_MODE,
    UNKNOWN_MODE,
    inject_broken_dependency,
    inject_duplicate_step,
    inject_missing_stop_condition,
    inject_step_deletion,
    inject_wrong_ordering,
    inject_wrong_tool,
)
from plan_repair.data import load_reference
from plan_repair.validation import (
    DANGLING_STEP,
    DEP_CYCLE,
    DUPLICATE_STEP,
    MISSING_STOP_CONDITION,
    ORDERING,
    UNKNOWN_DEPENDENCY,
    UNKNOWN_TOOL,
    input_from_path,
    step_path,
    stop_condition_path,
    validate_plan,
)


def detection_recall(injected, result):
    """Share of injected damaged steps/paths that the validator reports."""
    claimed_steps = {step_id for error in injected for step_id in error.damaged_step_ids}
    claimed_paths = {path for error in injected for path in error.damaged_paths}
    claimed = {("step", value) for value in claimed_steps} | {
        ("path", value) for value in claimed_paths
    }
    detected = {("step", value) for value in result.detected_step_ids()} | {
        ("path", value) for value in result.detected_paths()
    }
    assert claimed, "an injected error must claim at least one step or path"
    return len(claimed & detected) / len(claimed)


def corrupt(case, plan):
    if case == "B-1":
        return inject_broken_dependency(plan, step_id="enrich", mode=UNKNOWN_MODE)
    if case == "B-2":
        return inject_broken_dependency(plan, step_id="l_csv", mode=CYCLE_MODE, cycle_with="report")
    if case == "B-3":
        return inject_step_deletion(plan, step_id="join")
    if case == "B-4":
        return inject_step_deletion(plan, step_id="co")
    if case == "B-5":
        return inject_wrong_tool(plan, step_id="join")
    if case == "B-6":
        return inject_wrong_ordering(plan, step_id="join")
    if case == "B-7":
        return inject_duplicate_step(plan, step_id="agg")
    if case == "B-8":
        return inject_missing_stop_condition(plan)
    raise AssertionError(f"unknown golden case: {case}")


GOLDEN_CASES = ["B-1", "B-2", "B-3", "B-4", "B-5", "B-6", "B-7", "B-8"]
# B-6 is the ordering corruption itself; every other case must leave ordering silent.
NON_ORDERING_CASES = [case for case in GOLDEN_CASES if case != "B-6"]


def test_reference_pipeline_has_no_false_positive():
    task, plan = load_reference()
    result = validate_plan(plan, task)

    assert result.valid
    assert result.errors == []


@pytest.mark.parametrize("case", GOLDEN_CASES)
def test_detection_recall_is_one(case):
    task, plan = load_reference()
    corruption = corrupt(case, plan)

    result = validate_plan(corruption.broken_plan, task)

    assert not result.valid
    assert detection_recall(corruption.injected, result) == 1.0


@pytest.mark.parametrize("case", NON_ORDERING_CASES)
def test_corruptions_never_produce_ordering_errors(case):
    """Apart from the ordering corruption itself, an ordering error means an injector bug."""
    task, plan = load_reference()
    corruption = corrupt(case, plan)

    result = validate_plan(corruption.broken_plan, task)

    assert result.errors_of_type(ORDERING) == []


def test_b1_broken_dependency_unknown_mode():
    task, plan = load_reference()
    corruption = corrupt("B-1", plan)
    broken = corruption.broken_plan

    assert next(s for s in broken.steps if s.id == "enrich").input_from == ["join", "l_api_x"]

    errors = validate_plan(broken, task).errors_of_type(UNKNOWN_DEPENDENCY)
    assert len(errors) == 1
    assert errors[0].step_ids == ["enrich"]
    # Literal form of the scenario golden — both sides share the path helper, so the string
    # itself has to be asserted somewhere or a wrong notation would go unnoticed.
    assert errors[0].paths == ["$.steps[?enrich].input_from"]
    assert errors[0].paths == [input_from_path("enrich")]
    assert corruption.injected[0].damaged_step_ids == errors[0].step_ids
    assert corruption.injected[0].damaged_paths == errors[0].paths


def test_b2_broken_dependency_cycle_mode():
    task, plan = load_reference()
    corruption = corrupt("B-2", plan)

    errors = validate_plan(corruption.broken_plan, task).errors_of_type(DEP_CYCLE)

    assert len(errors) == 1
    assert {"l_csv", "report"} <= set(errors[0].step_ids)
    assert set(corruption.injected[0].damaged_step_ids) <= set(errors[0].step_ids)
    assert set(corruption.injected[0].damaged_paths) <= set(errors[0].paths)
    # Literal form of the scenario golden, asserted independently of the shared path helper.
    assert corruption.injected[0].damaged_paths == ["$.steps[?l_csv].input_from"]
    assert "$.steps[?l_csv].input_from" in errors[0].paths


def test_b3_step_deletion_at_a_fan_in():
    task, plan = load_reference()
    corruption = corrupt("B-3", plan)
    injected = corruption.injected[0]

    errors = validate_plan(corruption.broken_plan, task).errors_of_type(UNKNOWN_DEPENDENCY)

    assert len(errors) == 1
    assert errors[0].step_ids == ["enrich"]
    assert errors[0].paths == ["$.steps[?enrich].input_from"]
    assert injected.damaged_step_ids == ["enrich"]
    assert injected.damaged_paths == ["$.steps[?enrich].input_from"]
    assert injected.detail["deleted_step"]["id"] == "join"
    assert injected.detail["deleted_step"]["input_from"] == ["n_csv", "n_db"]
    # The collateral argument of this ticket: only one step of nineteen is damaged.
    assert len(corruption.preserved_step_ids) == 18


def test_b3_step_deletion_also_strands_the_steps_that_fed_it():
    """Deleting join leaves n_csv and n_db without a consumer (Ticket 002 dangling check)."""
    task, plan = load_reference()
    corruption = corrupt("B-3", plan)

    errors = validate_plan(corruption.broken_plan, task).errors_of_type(DANGLING_STEP)

    assert [error.step_ids[0] for error in errors] == ["n_csv", "n_db"]
    assert [error.paths[0] for error in errors] == ["$.steps[?n_csv]", "$.steps[?n_db]"]
    assert [error.paths[0] for error in errors] == [step_path("n_csv"), step_path("n_db")]


def test_b4_step_deletion_in_the_middle_of_a_chain():
    task, plan = load_reference()
    corruption = corrupt("B-4", plan)
    injected = corruption.injected[0]

    errors = validate_plan(corruption.broken_plan, task).errors_of_type(UNKNOWN_DEPENDENCY)

    assert len(errors) == 1
    assert errors[0].step_ids == ["n_csv"]
    assert errors[0].paths == ["$.steps[?n_csv].input_from"]
    assert injected.damaged_paths == [input_from_path("n_csv")]
    assert injected.damaged_step_ids == ["n_csv"]
    assert injected.detail["deleted_step"] == {
        "id": "co",
        "tool": "clean_outlier",
        "arguments": {},
        "input_from": ["cm"],
    }
    # cm survives but nothing consumes it any more — now caught by the dangling check.
    assert "cm" in corruption.preserved_step_ids
    dangling = validate_plan(corruption.broken_plan, task).errors_of_type(DANGLING_STEP)
    assert [error.step_ids[0] for error in dangling] == ["cm"]
    assert dangling[0].paths == ["$.steps[?cm]"]


def test_b5_wrong_tool():
    task, plan = load_reference()
    corruption = corrupt("B-5", plan)
    injected = corruption.injected[0]

    errors = validate_plan(corruption.broken_plan, task).errors_of_type(UNKNOWN_TOOL)

    assert len(errors) == 1
    assert errors[0].step_ids == ["join"]
    assert errors[0].paths == ["$.steps[?join].tool"]
    assert injected.damaged_step_ids == ["join"]
    assert injected.damaged_paths == ["$.steps[?join].tool"]
    assert injected.detail["original_tool"] == "join"


def test_b6_wrong_ordering():
    task, plan = load_reference()
    corruption = corrupt("B-6", plan)
    injected = corruption.injected[0]

    result = validate_plan(corruption.broken_plan, task)
    errors = result.errors_of_type(ORDERING)

    assert len(errors) == 1
    assert "join" in errors[0].step_ids
    assert errors[0].paths == ["$.steps[?join]"]
    assert injected.damaged_step_ids == ["join"]
    assert injected.damaged_paths == ["$.steps[?join]"]
    assert injected.detail["moved_from_index"] == 11
    # Reordering touches no edge, so the graph stays acyclic; a cycle here would be a bug.
    assert result.errors_of_type(DEP_CYCLE) == []


def test_b7_duplicate_step():
    task, plan = load_reference()
    corruption = corrupt("B-7", plan)
    injected = corruption.injected[0]

    result = validate_plan(corruption.broken_plan, task)
    errors = result.errors_of_type(DUPLICATE_STEP)

    assert len(errors) == 1
    assert errors[0].step_ids == ["agg", "agg_dup"]
    assert errors[0].paths == ["$.steps[?agg]", "$.steps[?agg_dup]"]
    assert errors[0].detail["kind"] == "identical_content"
    assert injected.damaged_step_ids == ["agg_dup"]
    assert injected.damaged_paths == ["$.steps[?agg_dup]"]
    assert injected.detail["duplicate_of"] == "agg"
    # The copy feeds nothing, so it is dangling as well; both checks fire independently.
    assert [error.step_ids[0] for error in result.errors_of_type(DANGLING_STEP)] == ["agg_dup"]


def test_b8_missing_stop_condition():
    task, plan = load_reference()
    corruption = corrupt("B-8", plan)

    errors = validate_plan(corruption.broken_plan, task).errors_of_type(MISSING_STOP_CONDITION)

    assert len(errors) == 1
    assert errors[0].step_ids == []
    assert errors[0].paths == ["$.stop_condition"]
    assert corruption.injected[0].damaged_paths == [stop_condition_path()]
