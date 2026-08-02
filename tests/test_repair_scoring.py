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
    # Nothing was touched, so nothing healthy was harmed either — in any of the ways.
    assert score.collateral_total == 0
    assert (score.collateral_modified, score.collateral_renamed, score.collateral_removed) == (
        0,
        0,
        0,
    )
    assert score.spurious_added == 0


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
    # The floor of the scale: perfect repair damages nothing, in any category.
    assert score.collateral_total == 0
    assert (score.collateral_modified, score.collateral_renamed, score.collateral_removed) == (
        0,
        0,
        0,
    )
    assert score.spurious_added == 0
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

    assert score.collateral_modified == 1
    assert score.modified_step_ids == [victim]
    assert score.collateral_total == 1
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
    assert "join" not in score.modified_step_ids
    assert score.collateral_total == 0


def test_a_step_the_corruption_deleted_is_not_charged_to_the_repairer():
    """`join` is healthy by the ground truth, but identity never had it to lose."""
    task, plan = load_reference()
    corruption = inject_step_deletion(plan, step_id="join")

    _, score = repair_and_score(
        IdentityRepairer(),
        reference_plan=plan,
        broken_plan=corruption.broken_plan,
        task=task,
        damaged_step_ids=corruption.injected[0].damaged_step_ids,
    )

    assert "join" not in corruption.injected[0].damaged_step_ids
    assert score.removed_step_ids == []
    assert score.added_step_ids == []
    assert score.collateral_total == 0
    assert not score.valid  # the loss shows up as errors, not as collateral


# --- the taxonomy: telling one kind of damage from another ---------------------------------------


class RewritingRepairer:
    """Returns a plan built by a transform of the reference, to stage a specific kind of damage."""

    name = "rewriting"

    def __init__(self, produce) -> None:
        self._produce = produce

    def repair(self, broken_plan, validation, task):
        return self._produce(broken_plan)


def rewritten(plan, transform):
    copy = plan.model_copy(deep=True)
    transform(copy)
    return RewritingRepairer(lambda _broken: copy)


def score_of(repairer, task, plan, corruption):
    _, score = repair_and_score(
        repairer,
        reference_plan=plan,
        broken_plan=corruption.broken_plan,
        task=task,
        damaged_step_ids=corruption.injected[0].damaged_step_ids,
    )
    return score


def test_renaming_a_healthy_step_is_a_rename_not_a_removal():
    """Its work survives under another name — damage to the reference, not to the work."""
    task, plan, corruption = corrupted()

    def rename_first_three(copy):
        for step in copy.steps[:3]:
            step.id = f"{step.id}_renamed"

    score = score_of(rewritten(plan, rename_first_three), task, plan, corruption)

    assert score.collateral_renamed == 3
    assert score.renamed_step_ids == [step.id for step in plan.steps[:3]]
    assert score.collateral_removed == 0
    assert score.collateral_modified == 0
    # The new ids carry renamed bodies, so they are not also counted as inventions.
    assert score.spurious_added == 0


def test_deleting_a_healthy_step_is_a_removal():
    """Nothing in the repaired plan does that work any more."""
    task, plan, corruption = corrupted()
    victims = [step.id for step in plan.steps[:3]]

    def delete_first_three(copy):
        copy.steps = [step for step in copy.steps if step.id not in victims]

    score = score_of(rewritten(plan, delete_first_three), task, plan, corruption)

    assert score.collateral_removed == 3
    assert score.removed_step_ids == victims
    assert score.collateral_renamed == 0
    assert score.collateral_modified == 0


def test_renaming_and_rewriting_a_step_is_a_removal_not_a_rename():
    """The id changed *and* the work changed: the original step is simply gone."""
    task, plan, corruption = corrupted()
    victim = plan.steps[0].id

    def rename_and_rewrite(copy):
        step = copy.steps[0]
        step.id = f"{victim}_v2"
        step.arguments = {**step.arguments, "rewritten": True}

    score = score_of(rewritten(plan, rename_and_rewrite), task, plan, corruption)

    assert score.removed_step_ids == [victim]
    assert score.collateral_renamed == 0
    # The replacement carries no healthy body, so it does count as an invention.
    assert score.added_step_ids == [f"{victim}_v2"]


def test_the_three_kinds_are_counted_side_by_side():
    task, plan, corruption = corrupted()
    modified, renamed, removed = (step.id for step in plan.steps[:3])

    def damage_one_of_each(copy):
        by_id = {step.id: step for step in copy.steps}
        by_id[modified].arguments = {**by_id[modified].arguments, "rewritten": True}
        by_id[renamed].id = f"{renamed}_renamed"
        copy.steps = [step for step in copy.steps if step.id != removed]

    score = score_of(rewritten(plan, damage_one_of_each), task, plan, corruption)

    assert (score.collateral_modified, score.collateral_renamed, score.collateral_removed) == (
        1,
        1,
        1,
    )
    assert score.modified_step_ids == [modified]
    assert score.renamed_step_ids == [renamed]
    assert score.removed_step_ids == [removed]
    assert score.collateral_total == 3


def test_a_wholesale_rewrite_under_fresh_ids_no_longer_scores_zero():
    """The blind spot this taxonomy exists for: renaming everything used to read as clean."""
    task, plan, corruption = corrupted()

    def rename_everything(copy):
        for index, step in enumerate(copy.steps):
            step.input_from = [
                f"s{copy.steps.index(other)}" for other in copy.steps if other.id in step.input_from
            ]
            step.id = f"s{index}"

    score = score_of(rewritten(plan, rename_everything), task, plan, corruption)
    healthy = len(plan.steps) - len(corruption.injected[0].damaged_step_ids)

    # Every healthy step is accounted for as damaged, one way or the other.
    assert score.collateral_total == healthy
    assert (
        score.collateral_modified + score.collateral_renamed + score.collateral_removed == healthy
    )


def test_a_genuinely_new_step_is_added_not_renamed():
    """Adding work is reported apart from damage: it may well be a legitimate repair."""
    task, plan, corruption = corrupted()

    def append_a_step(copy):
        extra = copy.steps[-1].model_copy(deep=True)
        extra.id = "audit"
        extra.arguments = {"scope": "final"}
        extra.input_from = [copy.steps[-1].id]
        copy.steps.append(extra)

    score = score_of(rewritten(plan, append_a_step), task, plan, corruption)

    assert score.spurious_added == 1
    assert score.added_step_ids == ["audit"]
    assert score.collateral_total == 0  # nothing healthy was harmed by adding


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
