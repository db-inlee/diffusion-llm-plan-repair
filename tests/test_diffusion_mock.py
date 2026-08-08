"""Mock diffusion against the scale.

Two things are being established. That a correct mask filled with the right answer restores the
damage and leaves everything else identical — and that this zero is earned rather than structural,
which is what the widened-mask repairer at the bottom is for. A collateral metric that reads zero
no matter how far the mask reaches would be measuring nothing.
"""

import json

import pytest

from plan_repair.corruption import (
    CYCLE_MODE,
    UNKNOWN_MODE,
    CorruptionSpec,
    inject_broken_dependency,
    inject_duplicate_step,
    inject_missing_stop_condition,
    inject_multi,
    inject_step_deletion,
    inject_wrong_ordering,
    inject_wrong_tool,
)
from plan_repair.data import DATA_PIPELINE_A, DATA_PIPELINE_B, load_reference
from plan_repair.repair import (
    PARSE_FAILURE,
    NoisyDiffusion,
    OracleDiffusion,
    Repairer,
    fill_masked,
    mask_spec,
    plan_to_sequence,
    repair_and_score,
    sequence_to_plan,
)
from plan_repair.schema import STEP_DELETION, WRONG_TOOL
from plan_repair.validation import validate_plan

DOMAINS = [DATA_PIPELINE_A, DATA_PIPELINE_B]


def scored(repairer, task, reference, corruption):
    return repair_and_score(
        repairer,
        reference_plan=reference,
        broken_plan=corruption.broken_plan,
        task=task,
        damaged_step_ids=corruption.injected[0].damaged_step_ids,
    )


def zero_collateral(score):
    return (
        score.collateral_modified,
        score.collateral_renamed,
        score.collateral_removed,
    ) == (0, 0, 0)


def test_the_mock_satisfies_the_repairer_port():
    _, plan = load_reference()

    assert isinstance(OracleDiffusion(plan), Repairer)
    assert isinstance(NoisyDiffusion(), Repairer)


# --- the ceiling: a correct mask, filled with the right answer -----------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_a_broken_dependency_is_regenerated_and_nothing_else_moves(domain):
    task, plan = load_reference(domain)
    fan_in = next(step for step in plan.steps if len(step.input_from) > 1)
    corruption = inject_broken_dependency(plan, step_id=fan_in.id, mode=UNKNOWN_MODE)

    repaired, score = scored(OracleDiffusion(plan), task, plan, corruption)

    assert score.solved
    assert score.damaged_restored == score.damaged_total == 1
    assert zero_collateral(score)
    assert score.spurious_added == 0
    assert repaired == plan


@pytest.mark.parametrize("domain", DOMAINS)
def test_a_wrong_tool_is_regenerated_and_nothing_else_moves(domain):
    task, plan = load_reference(domain)
    fan_in = next(step for step in plan.steps if len(step.input_from) > 1)
    corruption = inject_wrong_tool(plan, step_id=fan_in.id)

    _, score = scored(OracleDiffusion(plan), task, plan, corruption)

    assert score.solved
    assert zero_collateral(score)


def test_a_duplicated_step_is_regenerated_as_nothing():
    """The reference has no such step, so the honest fill is an empty one."""
    task, plan = load_reference()
    corruption = inject_duplicate_step(plan, step_id="agg")

    repaired, score = scored(OracleDiffusion(plan), task, plan, corruption)

    assert score.solved
    assert zero_collateral(score)
    assert [step.id for step in repaired.steps] == [step.id for step in plan.steps]


def test_a_cycle_masks_its_whole_component():
    """The mask is only as narrow as the validator's localisation, and a cycle is not narrow."""
    task, plan = load_reference()
    corruption = inject_broken_dependency(
        plan, step_id="l_csv", mode=CYCLE_MODE, cycle_with="report"
    )
    repairer = OracleDiffusion(plan)

    _, score = scored(repairer, task, plan, corruption)

    assert repairer.last_mask is not None
    assert len(repairer.last_mask.masked_step_ids) == 15  # the strongly connected component
    assert score.solved
    # Filling fifteen spans with the original steps still disturbs nothing.
    assert zero_collateral(score)


def test_multi_error_regenerates_every_damaged_region():
    task, plan = load_reference()
    corruption = inject_multi(
        plan,
        [
            CorruptionSpec(corruption_type=WRONG_TOOL, step_id="join"),
            CorruptionSpec(corruption_type=STEP_DELETION, step_id="co"),
        ],
    )
    damaged = [step_id for error in corruption.injected for step_id in error.damaged_step_ids]
    repairer = OracleDiffusion(plan)

    repaired, score = repair_and_score(
        repairer,
        reference_plan=plan,
        broken_plan=corruption.broken_plan,
        task=task,
        damaged_step_ids=damaged,
    )

    assert repairer.last_mask is not None
    assert {"join", "n_csv"} <= set(repairer.last_mask.masked_step_ids)
    assert zero_collateral(score)
    assert next(step for step in repaired.steps if step.id == "join").tool == "join"


# --- what step-level masking cannot reach --------------------------------------------------------


def test_a_deleted_step_cannot_be_brought_back():
    """No span, no regeneration: the answer exists but there is nowhere to put it."""
    task, plan = load_reference()
    corruption = inject_step_deletion(plan, step_id="join")
    repairer = OracleDiffusion(plan)

    repaired, score = scored(repairer, task, plan, corruption)

    assert "join" not in {step.id for step in repaired.steps}
    assert not score.solved
    # Still no damage to the healthy steps — it fails without making things worse.
    assert zero_collateral(score)


def test_an_ordering_violation_cannot_be_repaired_by_filling_spans():
    """Order lives in the layout, not inside any span."""
    task, plan = load_reference()
    corruption = inject_wrong_ordering(plan, step_id="join")

    repaired, score = scored(OracleDiffusion(plan), task, plan, corruption)

    assert [step.id for step in repaired.steps] != [step.id for step in plan.steps]
    assert not score.solved
    assert zero_collateral(score)


def test_a_missing_stop_condition_is_outside_every_span():
    """A plan-level field belongs to no step, so step-level masking never touches it."""
    task, plan = load_reference()
    corruption = inject_missing_stop_condition(plan)
    repairer = OracleDiffusion(plan)

    repaired, score = scored(repairer, task, plan, corruption)

    assert repairer.last_mask is not None
    assert repairer.last_mask.masked_step_ids == []
    assert repaired.stop_condition is None
    assert not score.solved
    assert zero_collateral(score)


# --- the failure path ----------------------------------------------------------------------------


def test_a_fill_that_is_not_a_step_is_a_failed_repair_not_a_crash():
    task, plan = load_reference()
    corruption = inject_wrong_tool(plan, step_id="join")
    repairer = NoisyDiffusion()

    repaired, score = scored(repairer, task, plan, corruption)

    assert repaired == corruption.broken_plan
    assert not score.solved
    assert [failure.kind for failure in repairer.failures] == [PARSE_FAILURE]
    assert zero_collateral(score)


# --- the mask mutation: is the zero earned? -------------------------------------------------------


class WideningDiffusion:
    """Masks the flagged steps plus a few healthy ones, and regenerates all of them.

    A stand-in for what any real fill does: the text inside the mask comes back rewritten rather
    than reproduced. Only the width of the mask differs from :class:`OracleDiffusion`.
    """

    name = "diffusion_widened"

    def __init__(self, extra_step_ids):
        self._extra = list(extra_step_ids)

    def repair(self, broken_plan, validation, task):
        sequence = plan_to_sequence(broken_plan)
        spec = mask_spec(sequence, set(validation.detected_step_ids()) | set(self._extra))
        filling = {}
        for step_id in (span.key for span in spec.spans):
            original = next(s for s in broken_plan.steps if s.id == step_id)
            payload = original.model_dump()
            payload["arguments"] = {**payload["arguments"], "regenerated": True}
            filling[step_id] = json.dumps(payload, ensure_ascii=False)
        return sequence_to_plan(fill_masked(sequence, spec, filling))


def test_widening_the_mask_onto_healthy_steps_produces_collateral():
    """The zeros above are a property of the mask, not of the metric."""
    task, plan = load_reference()
    fan_in = next(step for step in plan.steps if len(step.input_from) > 1)
    corruption = inject_broken_dependency(plan, step_id=fan_in.id, mode=UNKNOWN_MODE)
    healthy = ["l_csv", "l_db", "pr_csv"]

    _, narrow = scored(OracleDiffusion(plan), task, plan, corruption)
    _, unwidened = scored(WideningDiffusion([]), task, plan, corruption)
    _, widened = scored(WideningDiffusion(healthy), task, plan, corruption)

    assert narrow.collateral_modified == 0
    assert widened.collateral_modified == unwidened.collateral_modified + 3
    assert set(healthy) <= set(widened.modified_step_ids)


@pytest.mark.parametrize("extra", [1, 2, 5, 10])
def test_collateral_tracks_how_far_the_mask_reaches(extra):
    """Each further step brought inside the mask costs exactly one more healthy step."""
    task, plan = load_reference()
    fan_in = next(step for step in plan.steps if len(step.input_from) > 1)
    corruption = inject_broken_dependency(plan, step_id=fan_in.id, mode=UNKNOWN_MODE)
    validation = validate_plan(corruption.broken_plan, task)
    damaged = set(corruption.injected[0].damaged_step_ids)
    already_masked = validation.detected_step_ids()
    healthy = [
        step.id for step in plan.steps if step.id not in damaged and step.id not in already_masked
    ][:extra]

    _, baseline = scored(WideningDiffusion([]), task, plan, corruption)
    _, score = scored(WideningDiffusion(healthy), task, plan, corruption)

    assert score.collateral_modified == baseline.collateral_modified + extra


def test_the_error_region_can_already_contain_healthy_steps():
    """The mask is as precise as the validator, and the validator flags the dangling step too.

    Breaking a dependency leaves the step it pointed at without a consumer, so the validator
    reports that healthy step as well — and remasking therefore regenerates it. Collateral is
    bounded by the precision of the error region, not by the ground truth of what was damaged.
    """
    task, plan = load_reference()
    corruption = inject_broken_dependency(plan, step_id="join", mode=UNKNOWN_MODE)
    validation = validate_plan(corruption.broken_plan, task)
    damaged = set(corruption.injected[0].damaged_step_ids)

    flagged_but_healthy = validation.detected_step_ids() - damaged

    assert damaged == {"join"}
    assert flagged_but_healthy == {"n_db"}  # it lost its consumer when the edge was rewritten
    _, score = scored(WideningDiffusion([]), task, plan, corruption)
    assert score.modified_step_ids == ["n_db"]
