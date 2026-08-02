"""The scale before the things being weighed.

Identity and oracle have known answers, so they are what proves the scoring pipeline can tell a
repair from a non-repair. A sloppy repairer is added here — test-only, never a candidate — to
prove the same for collateral: a metric that reads zero for everything measures nothing.
"""

import pytest

from plan_repair.corruption import (
    UNKNOWN_MODE,
    inject_broken_dependency,
    inject_missing_stop_condition,
    inject_step_deletion,
)
from plan_repair.data import DATA_PIPELINE_A, DATA_PIPELINE_B, load_reference
from plan_repair.repair import (
    IdentityRepairer,
    OracleRepairer,
    Repairer,
    repair_and_score,
    score_repair,
)
from plan_repair.validation import validate_plan

DOMAINS = [DATA_PIPELINE_A, DATA_PIPELINE_B]


class SloppyRepairer:
    """Fixes the stop condition and rewrites a healthy step on the way past.

    Exists only to give the collateral metric something to catch.
    """

    name = "sloppy"

    def __init__(self, victim_id: str) -> None:
        self._victim_id = victim_id

    def repair(self, broken_plan, validation, task):
        plan = broken_plan.model_copy(deep=True)
        plan.stop_condition = "done"
        for step in plan.steps:
            if step.id == self._victim_id:
                step.arguments = {**step.arguments, "note": "touched"}
        return plan


def corrupted(domain=DATA_PIPELINE_B):
    """A plan with one unknown dependency, and the ground truth of what it damaged."""
    task, plan = load_reference(domain)
    fan_in = next(step for step in plan.steps if len(step.input_from) > 1)
    result = inject_broken_dependency(plan, step_id=fan_in.id, mode=UNKNOWN_MODE)
    return task, plan, result


def test_the_mocks_satisfy_the_repairer_port():
    _, plan = load_reference()

    assert isinstance(IdentityRepairer(), Repairer)
    assert isinstance(OracleRepairer(plan), Repairer)


@pytest.mark.parametrize("domain", DOMAINS)
def test_identity_is_the_floor(domain):
    task, plan, corruption = corrupted(domain)

    _, score = repair_and_score(
        IdentityRepairer(),
        reference_plan=plan,
        broken_plan=corruption.broken_plan,
        task=task,
        damaged_step_ids=corruption.injected[0].damaged_step_ids,
    )

    assert not score.solved
    assert not score.valid
    assert score.errors_remaining > 0
    assert score.damaged_restored == 0
    # Nothing was touched, so nothing healthy was harmed either.
    assert score.collateral == 0


@pytest.mark.parametrize("domain", DOMAINS)
def test_oracle_is_the_ceiling(domain):
    task, plan, corruption = corrupted(domain)

    repaired, score = repair_and_score(
        OracleRepairer(plan),
        reference_plan=plan,
        broken_plan=corruption.broken_plan,
        task=task,
        damaged_step_ids=corruption.injected[0].damaged_step_ids,
    )

    assert score.solved
    assert score.valid
    assert score.errors_remaining == 0
    assert score.runtime_succeeded
    assert score.collateral == 0
    assert score.damaged_restored == score.damaged_total == 1
    assert repaired == plan


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_floor_and_the_ceiling_are_distinguishable(domain):
    task, plan, corruption = corrupted(domain)
    damaged = corruption.injected[0].damaged_step_ids
    args = dict(
        reference_plan=plan,
        broken_plan=corruption.broken_plan,
        task=task,
        damaged_step_ids=damaged,
    )

    _, floor = repair_and_score(IdentityRepairer(), **args)
    _, ceiling = repair_and_score(OracleRepairer(plan), **args)

    assert (floor.solved, ceiling.solved) == (False, True)
    assert floor.errors_remaining > ceiling.errors_remaining
    assert floor.damaged_restored < ceiling.damaged_restored


def test_collateral_catches_a_touched_healthy_step():
    """Without this the zeros above would prove nothing."""
    task, plan = load_reference()
    corruption = inject_missing_stop_condition(plan)
    victim = plan.steps[0].id

    _, score = repair_and_score(
        SloppyRepairer(victim),
        reference_plan=plan,
        broken_plan=corruption.broken_plan,
        task=task,
        damaged_step_ids=[],
    )

    assert score.collateral == 1
    assert score.collateral_step_ids == [victim]
    # It did fix the reported error — quality and collateral are independent readings.
    assert score.valid


def test_a_restored_damaged_step_is_not_collateral():
    """The distinction the metric exists for: repairing damage is not damage."""
    task, plan, corruption = corrupted()
    damaged = corruption.injected[0].damaged_step_ids

    _, score = repair_and_score(
        OracleRepairer(plan),
        reference_plan=plan,
        broken_plan=corruption.broken_plan,
        task=task,
        damaged_step_ids=damaged,
    )

    assert damaged == ["join"]  # the first fan-in of the domain B pipeline
    assert score.damaged_restored == 1
    assert "join" not in score.collateral_step_ids


def test_removed_and_added_steps_are_reported_separately():
    """A step the corruption deleted is not charged to the repairer that did not restore it."""
    task, plan = load_reference()
    corruption = inject_step_deletion(plan, step_id="join")

    _, score = repair_and_score(
        IdentityRepairer(),
        reference_plan=plan,
        broken_plan=corruption.broken_plan,
        task=task,
        damaged_step_ids=corruption.injected[0].damaged_step_ids,
    )

    assert score.removed_step_ids == []
    assert score.added_step_ids == []
    assert score.collateral == 0
    assert not score.valid  # the loss shows up as errors, not as collateral


def test_scoring_reads_the_repaired_plan_not_the_broken_one():
    task, plan, corruption = corrupted()

    broken_errors = len(validate_plan(corruption.broken_plan, task).errors)
    score = score_repair(
        repairer_name="oracle",
        reference_plan=plan,
        broken_plan=corruption.broken_plan,
        repaired_plan=plan,
        task=task,
        damaged_step_ids=corruption.injected[0].damaged_step_ids,
    )

    assert broken_errors > 0
    assert score.errors_remaining == 0
