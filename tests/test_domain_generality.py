"""Domain generality: the same validator and injectors on two unrelated pipelines.

Domain B (data analysis, 20 steps) is what the code was written against; domain A (research
report, 19 steps) is added here as data only. If any domain knowledge — a step id, a tool name,
a terminal step — had leaked into ``validation/`` or ``corruption/``, the A cases below would
fail. That is the point of this file.

A-1..A-8 are the golden cases of the scenario document and Ticket 002; the shared battery at the
bottom runs the full corruption set over both domains.
"""

import pytest

from plan_repair.corruption import (
    CYCLE_MODE,
    UNKNOWN_MODE,
    inject_broken_dependency,
    inject_duplicate_step,
    inject_step_deletion,
    inject_wrong_ordering,
    inject_wrong_tool,
)
from plan_repair.data import DATA_PIPELINE_A, DATA_PIPELINE_B, load_reference
from plan_repair.validation import (
    DANGLING_STEP,
    DEP_CYCLE,
    DUPLICATE_STEP,
    MISSING_STOP_CONDITION,
    ORDERING,
    UNKNOWN_DEPENDENCY,
    UNKNOWN_TOOL,
    validate_plan,
)
from tests.test_detection_golden import detection_recall, inject_missing_stop_condition


def domain_a():
    return load_reference(DATA_PIPELINE_A)


# (case, injector) — A-5..A-8 follow the B-5..B-8 rules on the A pipeline.
A_CASES = {
    "A-1": lambda plan: inject_broken_dependency(plan, step_id="xcheck", mode=UNKNOWN_MODE),
    "A-2": lambda plan: inject_broken_dependency(
        plan, step_id="s_web", mode=CYCLE_MODE, cycle_with="fmt"
    ),
    "A-3": lambda plan: inject_step_deletion(plan, step_id="dedupe"),
    "A-4": lambda plan: inject_step_deletion(plan, step_id="e_paper"),
    "A-5": lambda plan: inject_wrong_tool(plan, step_id="dedupe"),
    "A-6": lambda plan: inject_wrong_ordering(plan, step_id="dedupe"),
    "A-7": lambda plan: inject_duplicate_step(plan, step_id="xcheck"),
    "A-8": inject_missing_stop_condition,
}
NON_ORDERING_A_CASES = [case for case in A_CASES if case != "A-6"]


def test_reference_pipeline_a_has_no_false_positive():
    task, plan = domain_a()
    result = validate_plan(plan, task)

    assert len(plan.steps) == 19
    assert len(task.available_tools) == 14
    assert result.valid
    assert result.errors == []


def test_same_tool_on_parallel_branches_is_not_duplication():
    """A calls parse_html twice and fetch three times — none of that is a duplicate."""
    task, plan = domain_a()

    assert [s.id for s in plan.steps if s.tool == "parse_html"] == ["p_web", "p_news"]
    assert [s.id for s in plan.steps if s.tool == "fetch"] == ["f_web", "f_paper", "f_news"]
    assert validate_plan(plan, task).errors_of_type(DUPLICATE_STEP) == []


@pytest.mark.parametrize("case", sorted(A_CASES))
def test_domain_a_detection_recall_is_one(case):
    task, plan = domain_a()
    corruption = A_CASES[case](plan)

    result = validate_plan(corruption.broken_plan, task)

    assert not result.valid
    assert detection_recall(corruption.injected, result) == 1.0


@pytest.mark.parametrize("case", sorted(NON_ORDERING_A_CASES))
def test_domain_a_corruptions_never_produce_ordering_errors(case):
    task, plan = domain_a()
    corruption = A_CASES[case](plan)

    assert validate_plan(corruption.broken_plan, task).errors_of_type(ORDERING) == []


def test_a1_broken_dependency_unknown_mode():
    task, plan = domain_a()
    corruption = A_CASES["A-1"](plan)

    errors = validate_plan(corruption.broken_plan, task).errors_of_type(UNKNOWN_DEPENDENCY)

    assert errors[0].step_ids == ["xcheck"]
    assert errors[0].paths == ["$.steps[?xcheck].input_from"]
    assert corruption.injected[0].damaged_paths == ["$.steps[?xcheck].input_from"]


def test_a2_broken_dependency_cycle_mode():
    task, plan = domain_a()
    corruption = A_CASES["A-2"](plan)

    errors = validate_plan(corruption.broken_plan, task).errors_of_type(DEP_CYCLE)

    assert len(errors) == 1
    assert {"s_web", "fmt"} <= set(errors[0].step_ids)
    # The other two collection branches never return to s_web, so they are not in the component.
    assert {"s_paper", "s_news"}.isdisjoint(errors[0].step_ids)
    assert corruption.injected[0].damaged_paths == ["$.steps[?s_web].input_from"]
    assert "$.steps[?s_web].input_from" in errors[0].paths


def test_a3_step_deletion_at_a_fan_in():
    task, plan = domain_a()
    corruption = A_CASES["A-3"](plan)
    injected = corruption.injected[0]

    errors = validate_plan(corruption.broken_plan, task).errors_of_type(UNKNOWN_DEPENDENCY)

    assert errors[0].step_ids == ["xcheck"]
    assert errors[0].paths == ["$.steps[?xcheck].input_from"]
    assert injected.damaged_step_ids == ["xcheck"]
    assert injected.damaged_paths == ["$.steps[?xcheck].input_from"]
    assert injected.detail["deleted_step"]["id"] == "dedupe"


def test_a4_step_deletion_on_a_parallel_branch():
    task, plan = domain_a()
    corruption = A_CASES["A-4"](plan)
    result = validate_plan(corruption.broken_plan, task)

    unknown = result.errors_of_type(UNKNOWN_DEPENDENCY)
    dangling = result.errors_of_type(DANGLING_STEP)

    assert unknown[0].step_ids == ["dedupe"]
    assert unknown[0].paths == ["$.steps[?dedupe].input_from"]
    assert corruption.injected[0].damaged_step_ids == ["dedupe"]
    # Only p_paper is stranded: f_paper and s_paper are still consumed by their branch.
    assert [error.step_ids[0] for error in dangling] == ["p_paper"]
    assert dangling[0].paths == ["$.steps[?p_paper]"]


def test_a5_wrong_tool():
    task, plan = domain_a()
    corruption = A_CASES["A-5"](plan)

    errors = validate_plan(corruption.broken_plan, task).errors_of_type(UNKNOWN_TOOL)

    assert errors[0].step_ids == ["dedupe"]
    assert errors[0].paths == ["$.steps[?dedupe].tool"]
    assert corruption.injected[0].damaged_paths == ["$.steps[?dedupe].tool"]
    assert corruption.injected[0].detail["original_tool"] == "dedupe"


def test_a6_wrong_ordering():
    task, plan = domain_a()
    corruption = A_CASES["A-6"](plan)

    result = validate_plan(corruption.broken_plan, task)
    errors = result.errors_of_type(ORDERING)

    assert "dedupe" in errors[0].step_ids
    assert errors[0].paths == ["$.steps[?dedupe]"]
    assert corruption.injected[0].damaged_paths == ["$.steps[?dedupe]"]
    assert result.errors_of_type(DEP_CYCLE) == []


def test_a7_duplicate_step():
    task, plan = domain_a()
    corruption = A_CASES["A-7"](plan)

    errors = validate_plan(corruption.broken_plan, task).errors_of_type(DUPLICATE_STEP)

    assert errors[0].step_ids == ["xcheck", "xcheck_dup"]
    assert errors[0].paths == ["$.steps[?xcheck]", "$.steps[?xcheck_dup]"]
    assert errors[0].detail["kind"] == "identical_content"
    assert corruption.injected[0].damaged_paths == ["$.steps[?xcheck_dup]"]


def test_a8_missing_stop_condition():
    task, plan = domain_a()
    corruption = A_CASES["A-8"](plan)

    errors = validate_plan(corruption.broken_plan, task).errors_of_type(MISSING_STOP_CONDITION)

    assert errors[0].step_ids == []
    assert errors[0].paths == ["$.stop_condition"]


# --- the same battery over both domains, driven only by structure ------------------------------

DOMAINS = [DATA_PIPELINE_A, DATA_PIPELINE_B]


def battery(task, plan):
    """Every corruption of tickets 001 and 002, targeted purely structurally.

    Targets are derived from the graph (a fan-in step, a chain step, the first source), never
    from hardcoded ids, so the same code drives both pipelines.
    """
    consumers = {dep for step in plan.steps for dep in step.input_from}
    fan_in = next(step for step in plan.steps if len(step.input_from) > 1)
    chained = next(step for step in plan.steps if len(step.input_from) == 1)
    source = next(step for step in plan.steps if not step.input_from)
    terminal = plan.steps[-1]
    consumed_source = next(
        step for step in plan.steps if not step.input_from and step.id in consumers
    )

    return {
        "broken_dependency/unknown": inject_broken_dependency(
            plan, step_id=fan_in.id, mode=UNKNOWN_MODE
        ),
        "broken_dependency/cycle": inject_broken_dependency(
            plan, step_id=consumed_source.id, mode=CYCLE_MODE, cycle_with=terminal.id
        ),
        "step_deletion": inject_step_deletion(plan, step_id=chained.id),
        "wrong_tool": inject_wrong_tool(plan, step_id=fan_in.id),
        "wrong_ordering": inject_wrong_ordering(plan, step_id=fan_in.id),
        "duplicate_step": inject_duplicate_step(plan, step_id=chained.id),
        "missing_stop_condition": inject_missing_stop_condition(plan),
        "duplicate_id": inject_duplicate_step(plan, step_id=source.id, new_id=source.id),
    }


@pytest.mark.parametrize("domain", DOMAINS)
def test_full_corruption_battery_is_detected_in_both_domains(domain):
    task, plan = load_reference(domain)

    for name, corruption in battery(task, plan).items():
        result = validate_plan(corruption.broken_plan, task)
        assert not result.valid, f"{domain}/{name} was not detected at all"
        assert detection_recall(corruption.injected, result) == 1.0, f"{domain}/{name}"
