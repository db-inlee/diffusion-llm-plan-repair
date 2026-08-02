"""Coverage — the one semantic check: does the plan do what the task asked for?

The goldens for drop_required_step were computed by hand before the validator was run, the same
discipline as Ticket 003: a dropped step damages the graph *and* uncovers a requirement, and both
belong in the expected set.
"""

import pytest

from plan_repair.corruption import inject_drop_required_step
from plan_repair.data import load_reference
from plan_repair.validation import (
    MISSING_EVIDENCE,
    MISSING_OPERATION,
    detection_metrics,
    required_evidence_path,
    required_operation_path,
    validate_plan,
)


def test_the_reference_plan_covers_every_requirement():
    task, plan = load_reference()
    produced = {tag for step in plan.steps for tag in step.produces}

    assert set(task.required_evidence) <= produced
    assert set(task.required_operations) <= produced
    assert validate_plan(plan, task).valid


def test_an_untagged_plan_reports_every_requirement_as_missing():
    task, plan = load_reference()
    for step in plan.steps:
        step.produces = []

    result = validate_plan(plan, task)

    assert [error.paths[0] for error in result.errors_of_type(MISSING_EVIDENCE)] == [
        "$.required_evidence[?csv_dataset]",
        "$.required_evidence[?db_dataset]",
        "$.required_evidence[?market_index]",
    ]
    assert [error.paths[0] for error in result.errors_of_type(MISSING_OPERATION)] == [
        "$.required_operations[?cleaning]",
        "$.required_operations[?normalization]",
        "$.required_operations[?join]",
        "$.required_operations[?aggregation]",
        "$.required_operations[?statistical_analysis]",
    ]


def test_coverage_errors_name_no_step():
    """A requirement nothing covers has no location in the plan, only in the task."""
    task, plan = load_reference()
    next(step for step in plan.steps if step.id == "l_csv").produces = []

    errors = validate_plan(plan, task).errors_of_type(MISSING_EVIDENCE)

    assert len(errors) == 1
    assert errors[0].step_ids == []
    assert errors[0].paths == [required_evidence_path("csv_dataset")]


def test_a_requirement_kept_alive_by_a_second_step_is_not_missing():
    """cleaning is claimed by both cm and co; losing one leaves the requirement covered."""
    task, plan = load_reference()
    next(step for step in plan.steps if step.id == "co").produces = []

    assert validate_plan(plan, task).errors_of_type(MISSING_OPERATION) == []


def test_coverage_ignores_tool_names():
    """Renaming a tool breaks tool existence, never coverage — the tag is what is claimed."""
    task, plan = load_reference()
    next(step for step in plan.steps if step.id == "join").tool = "join_x"

    result = validate_plan(plan, task)

    assert result.errors_of_type(MISSING_OPERATION) == []
    assert result.errors_of_type("unknown_tool")


# --- drop_required_step ---------------------------------------------------------------------
#
# D-1  csv_dataset is produced by l_csv alone. Dropping it breaks pr_csv's reference (l_csv is
#      gone) and uncovers the evidence. Nothing is stranded: l_csv consumed nothing, so no step
#      loses its consumer.
D1_GOLDEN = {
    ("unknown_dependency", ("pr_csv",), ("$.steps[?pr_csv].input_from",)),
    ("missing_evidence", (), ("$.required_evidence[?csv_dataset]",)),
}

# D-2  the join operation is claimed by the join step alone. Dropping it repeats the structural
#      damage of B-3 (enrich's reference breaks, n_csv and n_db lose their consumer) and adds the
#      uncovered operation.
D2_GOLDEN = {
    ("unknown_dependency", ("enrich",), ("$.steps[?enrich].input_from",)),
    ("dangling_step", ("n_csv",), ("$.steps[?n_csv]",)),
    ("dangling_step", ("n_db",), ("$.steps[?n_db]",)),
    ("missing_operation", (), ("$.required_operations[?join]",)),
}

DROP_CASES = {"D-1": ("csv_dataset", D1_GOLDEN), "D-2": ("join", D2_GOLDEN)}


@pytest.mark.parametrize("case", sorted(DROP_CASES))
def test_drop_required_step_detection_matches_the_golden_exactly(case):
    requirement, golden = DROP_CASES[case]
    task, plan = load_reference()

    corruption = inject_drop_required_step(plan, task, requirement=requirement)
    metrics = detection_metrics(validate_plan(corruption.broken_plan, task), golden)

    assert metrics.missed == []
    assert metrics.spurious == []
    assert metrics.recall == 1.0
    assert metrics.precision == 1.0


def test_drop_required_step_records_both_kinds_of_damage():
    task, plan = load_reference()

    corruption = inject_drop_required_step(plan, task, requirement="csv_dataset")
    injected = corruption.injected[0]

    assert injected.corruption_type == "drop_required_step"
    assert injected.damaged_step_ids == ["pr_csv"]
    assert injected.damaged_paths == [
        "$.steps[?pr_csv].input_from",
        "$.required_evidence[?csv_dataset]",
    ]
    assert injected.detail["requirement"] == "csv_dataset"
    assert injected.detail["deleted_step"]["id"] == "l_csv"
    assert plan.steps[0].id == "l_csv"  # the original plan is untouched


def test_drop_required_step_can_target_an_operation():
    task, plan = load_reference()

    corruption = inject_drop_required_step(plan, task, requirement="join")

    assert corruption.injected[0].damaged_paths[-1] == required_operation_path("join")


def test_drop_required_step_refuses_a_requirement_with_several_producers():
    """Dropping one of two producers would leave coverage intact — not this corruption."""
    task, plan = load_reference()

    with pytest.raises(ValueError, match="would not uncover it"):
        inject_drop_required_step(plan, task, requirement="cleaning")


def test_drop_required_step_refuses_something_the_task_does_not_require():
    task, plan = load_reference()

    with pytest.raises(ValueError, match="does not require"):
        inject_drop_required_step(plan, task, requirement="pivoting")


def test_drop_required_step_refuses_an_unproduced_requirement():
    task, plan = load_reference()
    next(step for step in plan.steps if step.id == "l_csv").produces = []

    with pytest.raises(ValueError, match="no step produces"):
        inject_drop_required_step(plan, task, requirement="csv_dataset")
