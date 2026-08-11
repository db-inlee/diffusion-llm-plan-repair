"""Cleaning a regenerated ``input_from``: drop what is not a reference, resolve what names one.

Ticket BD-1 narrowed the mask to the broken field and the collateral went to zero, but the field
still came back wrong, and wrong in two different ways. Domain B wrote the right step ids and put
an empty string in front of them, which is the surplus mask cell of Ticket C-3 showing up in a
list instead of a name. Domain A wrote ``["web_findings", "paper_findings", "news_findings"]`` —
the ``produces`` tags of exactly the three steps it should have referenced. The meaning was right
and the vocabulary was wrong.

Both are read back here rather than repaired by the model. An empty string is never a step id, so
it goes. A tag is turned into the step that produces it — **only when one step does**, because
domain B has three tags that two steps each claim and picking one would be inventing an answer
rather than reading one. A refusal is recorded, so what conservatism costs stays countable.

The producing step also has to be one this step could legally depend on: earlier in the plan.
That rules out a step resolving a tag to itself, which would swap a broken reference for a cycle
— the snap is not allowed to manufacture a new error while fixing one.
"""

import json

import pytest

from plan_repair.corruption import inject_broken_dependency
from plan_repair.data import DATA_PIPELINE_A, DATA_PIPELINE_B, load_reference
from plan_repair.repair import (
    mask_spec_from_paths,
    paths_to_mask,
    plan_to_sequence,
    repair_and_score,
)
from plan_repair.repair.diffusion import LLaDARepairer
from plan_repair.repair.snap import (
    AMBIGUOUS_TAG,
    NOT_A_LIST,
    REFUSED,
    UNCHANGED,
    UNRESOLVED,
    DependencySnap,
    snap_dependency_fillings,
    snap_dependency_value,
)
from plan_repair.validation import validate_plan

# What LLaDA returned once the mask was narrowed, from results/diffusion_bd1/bd1_remeasure.json.
# Replaying them says what the cleaning would have done to those answers — not what the model
# would write next time.
MEASURED = {
    DATA_PIPELINE_A: (' ["web_findings", "paper_findings", "news_findings"],', "dedupe"),
    DATA_PIPELINE_B: (' ["", "n_csv", "n_db"],', "join"),
}


def fan_in_id(plan):
    return next(step.id for step in plan.steps if len(step.input_from) > 1)


def masked(domain):
    """Corrupt a reference plan the way the matrix does and build the mask a repairer gets."""
    task, plan = load_reference(domain)
    result = inject_broken_dependency(plan, step_id=fan_in_id(plan))
    broken = result.broken_plan
    validation = validate_plan(broken, task)
    sequence = plan_to_sequence(broken)
    spec = mask_spec_from_paths(sequence, broken, paths_to_mask(validation, broken))
    return task, plan, broken, validation, spec


def cleaned(domain, text):
    _, _, broken, _, _ = masked(domain)
    step_id = fan_in_id(broken)
    return snap_dependency_value(text, broken, step_id)


# --- the cleaning itself -------------------------------------------------------------------


def test_an_empty_reference_is_dropped():
    """The surplus mask cell arrives as an element that names nothing."""
    decision = cleaned(DATA_PIPELINE_B, ' ["", "n_csv", "n_db"],')

    assert decision.snapped == ["n_csv", "n_db"]
    assert decision.dropped == [""]
    assert decision.resolved == {}


def test_a_produces_tag_becomes_the_step_that_produces_it():
    decision = cleaned(DATA_PIPELINE_A, ' ["web_findings", "paper_findings", "news_findings"],')

    assert decision.snapped == ["e_web", "e_paper", "e_news"]
    assert decision.resolved == {
        "web_findings": "e_web",
        "paper_findings": "e_paper",
        "news_findings": "e_news",
    }
    assert decision.refused == {}


def test_a_tag_two_steps_produce_is_refused():
    """``normalization`` is claimed by ``n_csv`` and ``n_db``; choosing would be inventing."""
    decision = cleaned(DATA_PIPELINE_B, '["n_csv", "normalization"]')

    assert decision.snapped is None
    assert decision.refused == {"normalization": AMBIGUOUS_TAG}
    assert decision.resolved == {}


def test_a_tag_the_step_produces_itself_is_refused():
    """Resolving it would swap a broken reference for a step that depends on itself."""
    decision = cleaned(DATA_PIPELINE_A, '["e_web", "dedup"]')

    assert decision.snapped is None
    assert decision.refused == {"dedup": UNRESOLVED}


def test_a_valid_step_id_is_left_alone():
    decision = cleaned(DATA_PIPELINE_B, '["n_csv", "n_db"]')

    assert decision.snapped is None  # nothing to change, so nothing is written back
    assert decision.refused == {}
    assert decision.original == ["n_csv", "n_db"]
    assert decision.reason == UNCHANGED


def test_a_record_that_changed_nothing_says_whether_it_gave_up():
    """ "Nothing to do" and "nothing I was willing to do" are different outcomes."""
    unchanged = cleaned(DATA_PIPELINE_B, '["n_csv", "n_db"]')
    gave_up = cleaned(DATA_PIPELINE_B, '["n_csv", "normalization"]')

    assert unchanged.snapped is gave_up.snapped is None
    assert unchanged.reason == UNCHANGED
    assert gave_up.reason == REFUSED


def test_a_string_that_names_nothing_is_kept_and_recorded():
    """It stays broken, and the validator goes on saying so — but the refusal is countable."""
    decision = cleaned(DATA_PIPELINE_B, '["n_csv", "n_db_x"]')

    assert decision.snapped is None
    assert decision.refused == {"n_db_x": UNRESOLVED}


def test_a_filling_that_is_not_a_list_is_left_alone():
    decision = cleaned(DATA_PIPELINE_B, ' "n_csv",')

    assert decision.snapped is None
    assert decision.reason == NOT_A_LIST


def test_the_replacement_is_rendered_as_json():
    decision = cleaned(DATA_PIPELINE_B, ' ["", "n_csv", "n_db"],')

    assert decision.replacement == json.dumps(["n_csv", "n_db"])


# --- applying it to a mask's fillings -------------------------------------------------------


def test_only_input_from_fields_are_cleaned():
    """A tool field belongs to the other post-processing and must not be touched here."""
    task, plan = load_reference(DATA_PIPELINE_B)
    from plan_repair.corruption import inject_wrong_tool

    broken = inject_wrong_tool(plan, step_id=fan_in_id(plan)).broken_plan
    sequence = plan_to_sequence(broken)
    validation = validate_plan(broken, task)
    spec = mask_spec_from_paths(sequence, broken, paths_to_mask(validation, broken))
    filling = {spec.spans[0].key: ' "join_db",'}

    snapped, record = snap_dependency_fillings(filling, spec, broken)

    assert spec.spans[0].field == "tool"
    assert snapped == filling
    assert record == {}


def test_the_record_names_the_span_that_was_cleaned():
    _, _, broken, _, spec = masked(DATA_PIPELINE_B)
    key = spec.spans[0].key

    snapped, record = snap_dependency_fillings({key: ' ["", "n_csv", "n_db"],'}, spec, broken)

    assert key == "join.input_from"
    assert snapped[key] == '["n_csv", "n_db"]'
    assert record[key].fired


def test_a_dropped_step_is_not_cleaned():
    _, _, broken, _, spec = masked(DATA_PIPELINE_B)

    snapped, record = snap_dependency_fillings({spec.spans[0].key: None}, spec, broken)

    assert snapped == {spec.spans[0].key: None}
    assert record == {}


# --- the repairer ----------------------------------------------------------------------------


class RecordedBackend:
    """A backend that returns a filling that was measured, so a run needs no model."""

    name = "recorded"

    def __init__(self, filling):
        self._filling = filling

    def fill(self, request):
        return {span.key: self._filling for span in request.mask.spans}


class CharTokenizer:
    name = "char"

    def encode_with_offsets(self, text):
        return [ord(character) for character in text], [(i, i + 1) for i in range(len(text))]


def repaired_with(domain, filling, *, snap_dependencies):
    task, plan = load_reference(domain)
    result = inject_broken_dependency(plan, step_id=fan_in_id(plan))
    repairer = LLaDARepairer(
        RecordedBackend(filling), CharTokenizer(), snap_dependencies=snap_dependencies
    )
    repaired, score = repair_and_score(
        repairer,
        reference_plan=plan,
        broken_plan=result.broken_plan,
        task=task,
        damaged_step_ids=[fan_in_id(plan)],
    )
    return repairer, repaired, score


@pytest.mark.parametrize("domain", sorted(MEASURED))
def test_the_cleaning_is_off_unless_it_is_asked_for(domain):
    """Ticket BD-1's measurement has to stay reproducible."""
    filling, _ = MEASURED[domain]

    _, _, score = repaired_with(domain, filling, snap_dependencies=False)

    assert not score.solved
    assert score.collateral_total == 0  # what BD-1 achieved, and BD-2 must not spend


@pytest.mark.parametrize("domain", sorted(MEASURED))
def test_the_measured_fillings_are_repaired_by_the_cleaning(domain):
    filling, target = MEASURED[domain]

    _, repaired, score = repaired_with(domain, filling, snap_dependencies=True)

    assert score.solved
    assert score.collateral_total == 0
    assert score.damaged_restored == 1
    _, reference = load_reference(domain)
    expected = next(step for step in reference.steps if step.id == target)
    assert next(step for step in repaired.steps if step.id == target).input_from == (
        expected.input_from
    )


def test_the_diagnostics_keep_the_model_output_that_the_cleaning_replaced():
    filling, _ = MEASURED[DATA_PIPELINE_A]

    repairer, _, _ = repaired_with(DATA_PIPELINE_A, filling, snap_dependencies=True)

    diagnostics = repairer.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.fillings["dedupe.input_from"] == filling
    assert diagnostics.dependency_snaps["dedupe.input_from"].resolved == {
        "web_findings": "e_web",
        "paper_findings": "e_paper",
        "news_findings": "e_news",
    }


def test_a_refusal_reaches_the_diagnostics():
    """Otherwise the cases conservatism gave up on cannot be counted."""
    repairer, _, score = repaired_with(
        DATA_PIPELINE_B, ' ["n_csv", "normalization"],', snap_dependencies=True
    )

    assert not score.solved
    assert repairer.last_diagnostics is not None
    record = repairer.last_diagnostics.dependency_snaps["join.input_from"]
    assert not record.fired
    assert record.refused == {"normalization": AMBIGUOUS_TAG}


def test_the_two_post_processings_are_independent_switches():
    """A tool snap must not turn the dependency cleaning on, or a control could not be built."""
    filling, _ = MEASURED[DATA_PIPELINE_B]
    task, plan = load_reference(DATA_PIPELINE_B)
    result = inject_broken_dependency(plan, step_id=fan_in_id(plan))
    repairer = LLaDARepairer(RecordedBackend(filling), CharTokenizer(), snap_tools=True)

    repairer.repair(result.broken_plan, validate_plan(result.broken_plan, task), task)

    assert repairer.last_diagnostics is not None
    assert repairer.last_diagnostics.dependency_snaps == {}


def test_a_snap_model_round_trips_through_json():
    """The record travels to a result file, so it has to survive being written down."""
    decision = cleaned(DATA_PIPELINE_B, ' ["", "n_csv", "n_db"],')

    assert DependencySnap.model_validate(json.loads(decision.model_dump_json())) == decision
