"""A dangling step left behind by a broken reference is not damage, and is not masked.

Breaking one dependency edge makes the validator report two things: the reference that now points
nowhere, and the step whose output nobody consumes any more. Only the first is damage. The second
step was never touched — it goes back to being consumed the moment the reference is repaired — yet
the mask covered it whole, which is 72-81% of everything the model was asked to rewrite. In domain
B the model duly rewrote it and changed an argument that was never broken.

So the mask stops acting on a dangling step whose cause is a broken reference. The word doing the
work is *cause*: a dangling step that no broken reference could explain is still masked, because
then nothing else is going to bring it back. That distinction is what these tests pin, in both
directions, for every corruption type that produces a dangling step.

The precedent is the validator's own: it skips the ordering check once it finds a cycle, because
an ordering violation there is a derived symptom rather than an independent finding. The judgement
here is the same shape, made on the mask side — the validator is the instrument and keeps
reporting everything it sees.
"""

import sys
from pathlib import Path

import pytest

from plan_repair.corruption import (
    inject_broken_dependency,
    inject_duplicate_step,
    inject_step_deletion,
)
from plan_repair.data import DATA_PIPELINE_A, DATA_PIPELINE_B, load_reference
from plan_repair.repair import (
    FillRequest,
    OracleBackend,
    fill_masked,
    mask_spec_from_paths,
    plan_to_sequence,
    score_repair,
    sequence_to_plan,
)
from plan_repair.repair.diffusion import LLaDARepairer
from plan_repair.repair.remask import derived_dangling_step_ids, paths_to_mask
from plan_repair.schema.plan import AgentPlan, Step
from plan_repair.schema.task import AgentTask, ToolSpec
from plan_repair.validation import validate_plan

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import run_diffusion_experiment as runner

DOMAINS = [DATA_PIPELINE_A, DATA_PIPELINE_B]


class CharTokenizer:
    """One token per character — the mask shape is what is under test, not the vocabulary."""

    name = "char"

    def encode_with_offsets(self, text):
        return [ord(character) for character in text], [(i, i + 1) for i in range(len(text))]


def fan_in_id(plan):
    return next(step.id for step in plan.steps if len(step.input_from) > 1)


def broken(domain, corrupt):
    """Corrupt a reference plan and return everything a mask is built from."""
    task, plan = load_reference(domain)
    result = corrupt(task, plan)
    validation = validate_plan(result.broken_plan, task)
    return task, plan, result, validation


def wrong_dependency(task, plan):
    return inject_broken_dependency(plan, step_id=fan_in_id(plan))


def spans_of(sequence, plan, paths):
    spec = mask_spec_from_paths(sequence, plan, paths)
    return [f"{span.step_id}.{span.field or 'WHOLE'}" for span in spec.spans], spec


# --- the causal judgement ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("domain", "dangling", "target"),
    [(DATA_PIPELINE_A, "e_news", "dedupe"), (DATA_PIPELINE_B, "n_db", "join")],
)
def test_a_dangling_step_left_by_a_broken_reference_is_derived(domain, dangling, target):
    _, _, result, validation = broken(domain, wrong_dependency)

    assert derived_dangling_step_ids(validation, result.broken_plan) == {dangling}
    # The reference itself is damage and stays: only the consequence is dropped.
    assert f"$.steps[?{target}].input_from" in paths_to_mask(validation, result.broken_plan)


def test_a_dangling_step_no_broken_reference_explains_is_kept():
    """A duplicate leaves a step nobody consumes, and nothing else will bring it back."""
    _, _, result, validation = broken(
        DATA_PIPELINE_B, lambda task, plan: inject_duplicate_step(plan, step_id="pr_csv")
    )

    assert validation.errors_of_type("dangling_step")
    assert not validation.errors_of_type("unknown_dependency")
    assert derived_dangling_step_ids(validation, result.broken_plan) == set()
    assert paths_to_mask(validation, result.broken_plan) == validation.detected_paths()


def test_a_step_the_broken_reference_could_not_have_consumed_is_kept():
    """Causation runs one way: a step may only consume steps listed before it.

    ``late`` sits after the step with the broken reference, so repairing that reference cannot
    make anything consume ``late`` — it is dangling on its own account and stays in the mask.
    ``early`` sits before it and is exactly the step the reference lost.
    """
    plan = AgentPlan(
        goal="g",
        steps=[
            Step(id="early", tool="t", produces=["e"]),
            Step(id="consumer", tool="t", input_from=["early_x"]),
            Step(id="late", tool="t"),
            Step(id="terminal", tool="t", input_from=["consumer"]),
        ],
        stop_condition="done",
    )
    task = AgentTask(task_id="t", user_query="q", available_tools=[ToolSpec(name="t")])
    validation = validate_plan(plan, task)

    assert {error.step_ids[0] for error in validation.errors_of_type("dangling_step")} == {
        "early",
        "late",
    }
    assert derived_dangling_step_ids(validation, plan) == {"early"}
    assert "$.steps[?late]" in paths_to_mask(validation, plan)
    assert "$.steps[?early]" not in paths_to_mask(validation, plan)


# --- what it does to the mask ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("domain", "target", "before", "after"),
    [(DATA_PIPELINE_A, "dedupe", 148, 32), (DATA_PIPELINE_B, "join", 145, 19)],
)
def test_the_mask_narrows_to_the_broken_field_alone(domain, target, before, after):
    _, _, result, validation = broken(domain, wrong_dependency)
    sequence = plan_to_sequence(result.broken_plan)

    wide, wide_spec = spans_of(sequence, result.broken_plan, validation.detected_paths())
    narrow, narrow_spec = spans_of(
        sequence, result.broken_plan, paths_to_mask(validation, result.broken_plan)
    )

    assert narrow == [f"{target}.input_from"]
    assert len(wide) == 2  # the field and the whole healthy step
    assert (wide_spec.masked_characters, narrow_spec.masked_characters) == (before, after)


def test_the_step_dropped_from_the_mask_comes_back_untouched():
    """It is not regenerated at all, so there is nothing for a model to change."""
    task, _, result, validation = broken(DATA_PIPELINE_B, wrong_dependency)
    repairer = LLaDARepairer(OracleBackend(result.broken_plan), CharTokenizer())

    repaired = repairer.repair(result.broken_plan, validation, task)

    assert repairer.last_mask is not None
    assert "n_db" in repairer.last_mask.preserved_step_ids
    before = next(step for step in result.broken_plan.steps if step.id == "n_db")
    after = next(step for step in repaired.steps if step.id == "n_db")
    assert after == before


def test_the_repairer_masks_what_the_narrowing_leaves():
    task, _, result, validation = broken(DATA_PIPELINE_A, wrong_dependency)
    repairer = LLaDARepairer(OracleBackend(result.broken_plan), CharTokenizer())

    repairer.repair(result.broken_plan, validation, task)

    assert repairer.last_mask is not None
    assert [span.key for span in repairer.last_mask.spans] == ["dedupe.input_from"]


# --- the other corruption types ----------------------------------------------------------------


def ceiling(domain, corrupt, choose_paths):
    """What the pipeline scores when the mask is filled with the pre-corruption answer.

    The steps of ``DiffusionRepairer.repair`` run here directly, so the mask can be built from
    either set of paths without the repairer needing a switch it would never use in production.
    """
    task, plan = load_reference(domain)
    result = corrupt(task, plan)
    damaged = [step for error in result.injected for step in error.damaged_step_ids]
    validation = validate_plan(result.broken_plan, task)

    sequence = plan_to_sequence(result.broken_plan)
    spec = mask_spec_from_paths(
        sequence, result.broken_plan, choose_paths(validation, result.broken_plan)
    )
    request = FillRequest(sequence=sequence, mask=spec, alignment=None, task=task)
    filling = OracleBackend(plan).fill(request)
    repaired = sequence_to_plan(fill_masked(sequence, spec, filling, result.broken_plan))

    return score_repair(
        repairer_name="oracle",
        reference_plan=plan,
        broken_plan=result.broken_plan,
        repaired_plan=repaired,
        task=task,
        damaged_step_ids=damaged,
    )


def every_path(validation, plan):
    """What the mask was built from before this ticket."""
    return validation.detected_paths()


@pytest.mark.parametrize("name", sorted(runner.CORRUPTIONS))
@pytest.mark.parametrize("domain", DOMAINS)
def test_no_corruption_type_loses_its_ceiling_to_the_narrowing(domain, name):
    """The narrowing may only ever remove damage from the mask, never the answer from reach."""
    corrupt = runner.CORRUPTIONS[name]

    before = ceiling(domain, corrupt, every_path)
    after = ceiling(domain, corrupt, paths_to_mask)

    assert after.solved == before.solved, name
    assert after.collateral_total <= before.collateral_total, name
    assert after.damaged_restored >= before.damaged_restored, name


@pytest.mark.parametrize(
    ("domain", "step_id"), [(DATA_PIPELINE_A, "f_web"), (DATA_PIPELINE_B, "pr_csv")]
)
def test_a_duplicate_still_masks_both_copies(domain, step_id):
    """The rule must not reach a corruption it has nothing to say about."""
    _, _, result, validation = broken(
        domain, lambda task, plan: inject_duplicate_step(plan, step_id=step_id)
    )
    sequence = plan_to_sequence(result.broken_plan)

    spans, _ = spans_of(sequence, result.broken_plan, paths_to_mask(validation, result.broken_plan))

    assert spans == [f"{step_id}.WHOLE", f"{step_id}_dup.WHOLE"]


@pytest.mark.parametrize(
    ("domain", "kept"),
    [(DATA_PIPELINE_A, "xcheck.input_from"), (DATA_PIPELINE_B, "enrich.input_from")],
)
def test_a_deletion_keeps_the_broken_reference_and_drops_the_orphans(domain, kept):
    """The steps the deleted one fed are dangling because of it — masking them cannot help."""
    _, _, result, validation = broken(
        domain, lambda task, plan: inject_step_deletion(plan, step_id=fan_in_id(plan))
    )
    sequence = plan_to_sequence(result.broken_plan)

    spans, _ = spans_of(sequence, result.broken_plan, paths_to_mask(validation, result.broken_plan))

    assert spans == [kept]
