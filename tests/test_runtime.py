"""Deterministic execution, and how its verdict lines up with the static one.

Expected outcomes are derived from the three rules (tool offered, arguments usable, predecessors
succeeded) rather than read back out of the implementation.
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
from plan_repair.runtime import AGREE, VALIDATOR_MISSED, VALIDATOR_STRICTER, agreement, run_plan
from plan_repair.runtime.execution import (
    FAILED_DEPENDENCY,
    UNKNOWN_DEPENDENCY,
    UNKNOWN_TOOL,
    UNRESOLVABLE,
    UNUSABLE_ARGUMENT,
)
from plan_repair.validation import validate_plan


def step(plan, step_id):
    return next(s for s in plan.steps if s.id == step_id)


def test_the_reference_plan_runs_to_the_end():
    task, plan = load_reference()

    run = run_plan(plan, task)

    assert run.succeeded
    assert run.stop_condition_reached
    assert run.failed_step_ids() == []
    assert [outcome.step_id for outcome in run.outcomes] == [s.id for s in plan.steps]


def test_execution_is_deterministic():
    task, plan = load_reference()
    broken = inject_step_deletion(plan, step_id="join").broken_plan

    assert run_plan(plan, task) == run_plan(plan, task)
    assert run_plan(broken, task) == run_plan(broken, task)


def test_an_unknown_tool_fails_its_step_and_everything_downstream():
    task, plan = load_reference()
    broken = inject_wrong_tool(plan, step_id="join").broken_plan

    run = run_plan(broken, task)

    assert not run.succeeded
    assert run.failed_step_ids() == ["join", "enrich", "agg", "pivot", "stat", "corr", "viz",
                                     "interp", "report"]  # fmt: skip
    assert next(o for o in run.outcomes if o.step_id == "join").reason == UNKNOWN_TOOL
    assert next(o for o in run.outcomes if o.step_id == "enrich").reason == FAILED_DEPENDENCY
    # Everything above the break still ran.
    assert next(o for o in run.outcomes if o.step_id == "n_csv").succeeded


def test_failure_propagates_to_consumers_and_stops_there():
    """A deleted step takes its whole downstream with it and leaves the rest running."""
    task, plan = load_reference()
    broken = inject_step_deletion(plan, step_id="join").broken_plan

    run = run_plan(broken, task)

    assert run.failed_step_ids() == ["enrich", "agg", "pivot", "stat", "corr", "viz",
                                     "interp", "report"]  # fmt: skip
    assert [o.step_id for o in run.outcomes if o.succeeded] == ["l_csv", "l_db", "l_api",
                                                               "pr_csv", "pr_db", "vs_csv",
                                                               "vs_db", "cm", "co", "n_csv",
                                                               "n_db"]  # fmt: skip
    assert next(o for o in run.outcomes if o.step_id == "enrich").reason == UNKNOWN_DEPENDENCY
    assert {o.reason for o in run.outcomes if o.step_id in {"agg", "report"}} == {FAILED_DEPENDENCY}


def test_a_blank_argument_makes_a_step_unrunnable():
    task, plan = load_reference()
    step(plan, "l_csv").arguments = {"path": "   "}

    run = run_plan(plan, task)

    assert not run.succeeded
    assert next(o for o in run.outcomes if o.step_id == "l_csv").reason == UNUSABLE_ARGUMENT


def test_a_reference_to_a_missing_step_fails_the_consumer():
    task, plan = load_reference()
    broken = inject_broken_dependency(plan, step_id="enrich", mode=UNKNOWN_MODE).broken_plan

    run = run_plan(broken, task)

    assert not run.succeeded
    assert next(o for o in run.outcomes if o.step_id == "enrich").reason == UNKNOWN_DEPENDENCY


def test_a_cycle_never_resolves():
    task, plan = load_reference()
    broken = inject_broken_dependency(
        plan, step_id="l_csv", mode=CYCLE_MODE, cycle_with="report"
    ).broken_plan

    run = run_plan(broken, task)

    assert not run.succeeded
    assert not run.stop_condition_reached
    assert next(o for o in run.outcomes if o.step_id == "l_csv").reason == UNRESOLVABLE
    # Steps outside the cycle still run.
    assert next(o for o in run.outcomes if o.step_id == "l_db").succeeded


def test_a_failing_terminal_means_the_stop_condition_was_not_reached():
    task, plan = load_reference()
    step(plan, "report").tool = "report_x"

    run = run_plan(plan, task)

    assert not run.stop_condition_reached
    assert not run.succeeded


def test_an_empty_plan_reaches_nothing():
    task, plan = load_reference()
    plan.steps = []

    run = run_plan(plan, task)

    assert not run.succeeded
    assert not run.stop_condition_reached


# --- agreement between the two judges ----------------------------------------------------------
#
# Expected from the rules, not from the code: a corruption fails at runtime only when it stops a
# tool call from being made. Reordering, duplicating and dropping the stop condition leave every
# call intact, so those plans still run — the validator is stricter there, on purpose.
AGREEMENT_CASES = {
    "reference": (lambda t, p: p, AGREE),
    "broken_dependency/unknown": (
        lambda t, p: inject_broken_dependency(p, step_id="enrich", mode=UNKNOWN_MODE).broken_plan,
        AGREE,
    ),
    "broken_dependency/cycle": (
        lambda t, p: (
            inject_broken_dependency(
                p, step_id="l_csv", mode=CYCLE_MODE, cycle_with="report"
            ).broken_plan
        ),
        AGREE,
    ),
    "step_deletion": (
        lambda t, p: inject_step_deletion(p, step_id="join").broken_plan,
        AGREE,
    ),
    "wrong_tool": (
        lambda t, p: inject_wrong_tool(p, step_id="join").broken_plan,
        AGREE,
    ),
    "wrong_ordering": (
        lambda t, p: inject_wrong_ordering(p, step_id="join").broken_plan,
        VALIDATOR_STRICTER,
    ),
    "duplicate_step": (
        lambda t, p: inject_duplicate_step(p, step_id="agg").broken_plan,
        VALIDATOR_STRICTER,
    ),
    "missing_stop_condition": (
        lambda t, p: inject_missing_stop_condition(p).broken_plan,
        VALIDATOR_STRICTER,
    ),
}


@pytest.mark.parametrize("case", sorted(AGREEMENT_CASES))
def test_validator_and_runtime_agree_where_expected(case):
    corrupt, expected = AGREEMENT_CASES[case]
    task, plan = load_reference()
    subject = corrupt(task, plan)

    verdict = agreement(validate_plan(subject, task), run_plan(subject, task))

    assert verdict.verdict == expected


def test_the_static_blind_spot_is_a_blank_argument():
    """The one shape where the runtime is stricter: no static check looks at argument values."""
    task, plan = load_reference()
    step(plan, "l_csv").arguments = {"path": ""}

    verdict = agreement(validate_plan(plan, task), run_plan(plan, task))

    assert verdict.plan_valid
    assert not verdict.run_succeeded
    assert verdict.verdict == VALIDATOR_MISSED
