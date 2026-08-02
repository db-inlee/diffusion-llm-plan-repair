"""What the rules repair, and — just as deliberately — what they refuse to.

The declined cases are pinned as tightly as the handled ones. A baseline that silently grew new
abilities would move the line the model repairers are measured against without anyone noticing.
"""

import pytest

from plan_repair.corruption import (
    CYCLE_MODE,
    UNKNOWN_MODE,
    inject_broken_dependency,
    inject_drop_required_step,
    inject_duplicate_step,
    inject_missing_stop_condition,
    inject_step_deletion,
    inject_wrong_ordering,
    inject_wrong_tool,
)
from plan_repair.data import DATA_PIPELINE_A, DATA_PIPELINE_B, load_reference
from plan_repair.repair import (
    DECLINED_ERROR_TYPES,
    HANDLED_ERROR_TYPES,
    DeterministicRepairer,
    Repairer,
    repair_and_score,
)
from plan_repair.validation import (
    DANGLING_STEP,
    DEP_CYCLE,
    DUPLICATE_STEP,
    MISSING_EVIDENCE,
    MISSING_STOP_CONDITION,
    ORDERING,
    UNKNOWN_DEPENDENCY,
    UNKNOWN_TOOL,
    validate_plan,
)

DOMAINS = [DATA_PIPELINE_A, DATA_PIPELINE_B]


def repaired_for(corruption, task, reference):
    return repair_and_score(
        DeterministicRepairer(),
        reference_plan=reference,
        broken_plan=corruption.broken_plan,
        task=task,
        damaged_step_ids=corruption.injected[0].damaged_step_ids,
    )


def test_the_repairer_satisfies_the_port():
    assert isinstance(DeterministicRepairer(), Repairer)


def test_the_handled_and_declined_sets_do_not_overlap():
    assert not HANDLED_ERROR_TYPES & DECLINED_ERROR_TYPES


def test_the_broken_plan_is_never_mutated():
    task, plan = load_reference()
    corruption = inject_missing_stop_condition(plan)
    broken = corruption.broken_plan
    before = broken.model_dump_json()

    DeterministicRepairer().repair(broken, validate_plan(broken, task), task)

    assert broken.model_dump_json() == before


# --- handled ------------------------------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_missing_stop_condition_is_filled_from_the_terminal(domain):
    task, plan = load_reference(domain)
    corruption = inject_missing_stop_condition(plan)

    repaired, score = repaired_for(corruption, task, plan)

    assert repaired.stop_condition == f"terminal step {plan.steps[-1].id} completed"
    assert score.solved
    assert score.collateral_total == 0


@pytest.mark.parametrize("domain", DOMAINS)
def test_an_unknown_dependency_is_dropped(domain):
    task, plan = load_reference(domain)
    fan_in = next(step for step in plan.steps if len(step.input_from) > 1)
    corruption = inject_broken_dependency(plan, step_id=fan_in.id, mode=UNKNOWN_MODE)

    repaired, score = repaired_for(corruption, task, plan)

    assert validate_plan(repaired, task).errors_of_type(UNKNOWN_DEPENDENCY) == []
    assert score.collateral_total == 0
    # The reference it was meant to consume is gone rather than restored: the rules cannot know
    # what the plan intended, so the step comes back valid but poorer than the original.
    assert score.damaged_restored == 0


def test_a_duplicated_step_is_removed():
    task, plan = load_reference()
    corruption = inject_duplicate_step(plan, step_id="agg")

    repaired, score = repaired_for(corruption, task, plan)

    assert [step.id for step in repaired.steps] == [step.id for step in plan.steps]
    assert score.solved
    # The copy was the damaged step, so deleting it is the repair — nothing healthy was touched.
    assert score.collateral_total == 0
    assert score.removed_step_ids == []


def test_a_complete_duplicate_is_removed_even_under_the_same_id():
    """Redundant whatever it is called: one of two identical steps goes."""
    task, plan = load_reference()
    corruption = inject_duplicate_step(plan, step_id="agg", new_id="agg")

    repaired, score = repaired_for(corruption, task, plan)

    assert [step.id for step in repaired.steps].count("agg") == 1
    assert score.solved


def test_an_id_clash_between_different_steps_is_left_alone():
    """Two different steps under one name: choosing which keeps the name is not mechanical."""
    task, plan = load_reference()
    corruption = inject_duplicate_step(plan, step_id="agg", new_id="agg")
    clash = corruption.broken_plan
    # Make the copy do different work, so only the id collides.
    [_, second] = [step for step in clash.steps if step.id == "agg"]
    second.arguments = {"window": "quarterly"}

    repaired, _ = repair_and_score(
        DeterministicRepairer(),
        reference_plan=plan,
        broken_plan=clash,
        task=task,
        damaged_step_ids=["agg"],
    )

    assert [step.id for step in repaired.steps].count("agg") == 2
    assert validate_plan(repaired, task).errors_of_type(DUPLICATE_STEP)


@pytest.mark.parametrize("domain", DOMAINS)
def test_ordering_is_restored_topologically(domain):
    task, plan = load_reference(domain)
    fan_in = next(step for step in plan.steps if len(step.input_from) > 1)
    corruption = inject_wrong_ordering(plan, step_id=fan_in.id)

    repaired, score = repaired_for(corruption, task, plan)

    assert validate_plan(repaired, task).errors_of_type(ORDERING) == []
    assert score.solved
    # Reordering moves steps without rewriting any of them, so nothing is charged as collateral.
    assert score.collateral_total == 0
    assert [step.id for step in repaired.steps] == [step.id for step in plan.steps]


# --- declined -----------------------------------------------------------------------------------


def test_an_unknown_tool_is_left_alone():
    """Which tool was meant is a reading of intent, not a lookup."""
    task, plan = load_reference()
    corruption = inject_wrong_tool(plan, step_id="join")

    repaired, score = repaired_for(corruption, task, plan)

    assert validate_plan(repaired, task).errors_of_type(UNKNOWN_TOOL)
    assert not score.solved
    assert next(s for s in repaired.steps if s.id == "join").tool == "join_x"


def test_a_cycle_is_left_alone():
    """Cutting the wrong edge of a cycle destroys the dependency the plan meant to keep."""
    task, plan = load_reference()
    corruption = inject_broken_dependency(
        plan, step_id="l_csv", mode=CYCLE_MODE, cycle_with="report"
    )

    repaired, score = repaired_for(corruption, task, plan)

    assert validate_plan(repaired, task).errors_of_type(DEP_CYCLE)
    assert not score.solved
    assert score.collateral_total == 0


def test_a_deleted_step_is_not_restored():
    """The rules can unbreak the reference but cannot bring back the work that was deleted."""
    task, plan = load_reference()
    corruption = inject_step_deletion(plan, step_id="join")

    repaired, score = repaired_for(corruption, task, plan)

    result = validate_plan(repaired, task)
    assert result.errors_of_type(UNKNOWN_DEPENDENCY) == []  # the dangling reference is gone
    assert "join" not in {step.id for step in repaired.steps}  # the step itself is not
    assert result.errors_of_type(DANGLING_STEP)  # and its inputs are now orphans
    assert not score.solved
    # The characteristic weakness of rule-based repair: dropping the reference makes the plan
    # runnable again while the work it stood for stays missing. It executes, and does less.
    assert score.runtime_succeeded
    assert not score.valid


def test_an_uncovered_requirement_is_left_alone():
    """Designing a step to cover a requirement is the model repairers' work."""
    task, plan = load_reference()
    corruption = inject_drop_required_step(plan, task, requirement="csv_dataset")

    repaired, score = repaired_for(corruption, task, plan)

    assert validate_plan(repaired, task).errors_of_type(MISSING_EVIDENCE)
    assert not score.solved


def test_the_declined_types_are_never_repaired_away():
    """One sweep asserting the baseline stays where it was drawn."""
    task, plan = load_reference()
    cases = {
        UNKNOWN_TOOL: inject_wrong_tool(plan, step_id="join"),
        DEP_CYCLE: inject_broken_dependency(
            plan, step_id="l_csv", mode=CYCLE_MODE, cycle_with="report"
        ),
        MISSING_EVIDENCE: inject_drop_required_step(plan, task, requirement="csv_dataset"),
    }
    for error_type, corruption in cases.items():
        repaired, _ = repaired_for(corruption, task, plan)
        assert validate_plan(repaired, task).errors_of_type(error_type), error_type


# --- a repair that mixes both -------------------------------------------------------------------


def test_a_plan_with_both_kinds_gets_the_handled_part_only():
    task, plan = load_reference()
    with_tool = inject_wrong_tool(plan, step_id="join").broken_plan
    both = inject_missing_stop_condition(with_tool).broken_plan

    repaired, score = repair_and_score(
        DeterministicRepairer(),
        reference_plan=plan,
        broken_plan=both,
        task=task,
        damaged_step_ids=["join"],
    )

    remaining = {error.type for error in validate_plan(repaired, task).errors}
    assert MISSING_STOP_CONDITION not in remaining  # handled
    assert UNKNOWN_TOOL in remaining  # declined
    assert not score.solved
    assert score.collateral_total == 0
    assert DUPLICATE_STEP not in remaining
