"""Selective remask: does the mask land exactly on the damage?

The contract has two halves and both need holding. The mask must cover the steps the validator
flagged, and it must cover nothing else — the second half is what makes a diffusion repairer
structurally unable to disturb healthy work, and it is the half that would fail silently.
"""

import json

import pytest

from plan_repair.corruption import (
    UNKNOWN_MODE,
    CorruptionSpec,
    inject_broken_dependency,
    inject_duplicate_step,
    inject_multi,
    inject_step_deletion,
    inject_wrong_tool,
)
from plan_repair.data import DATA_PIPELINE_A, DATA_PIPELINE_B, load_reference
from plan_repair.repair import (
    DEFAULT_PLACEHOLDER,
    PlanParseError,
    fill_masked,
    mask_spec,
    plan_to_sequence,
    render_masked,
    sequence_to_plan,
)
from plan_repair.repair.remask import _rebuild, normalise_filling
from plan_repair.schema import MISSING_STOP_CONDITION, STEP_DELETION, WRONG_TOOL
from plan_repair.validation import validate_plan

DOMAINS = [DATA_PIPELINE_A, DATA_PIPELINE_B]


# --- the representation -------------------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_a_plan_survives_the_round_trip(domain):
    _, plan = load_reference(domain)

    assert sequence_to_plan(plan_to_sequence(plan).text) == plan


@pytest.mark.parametrize("domain", DOMAINS)
def test_every_step_is_exactly_one_span(domain):
    _, plan = load_reference(domain)
    sequence = plan_to_sequence(plan)

    assert sequence.step_ids() == [step.id for step in plan.steps]
    for span, step in zip(sequence.spans, plan.steps, strict=True):
        body = sequence.text[span.start : span.end]
        assert json.loads(body)["id"] == step.id
        assert json.loads(body) == step.model_dump()


@pytest.mark.parametrize("domain", DOMAINS)
def test_spans_do_not_overlap_and_follow_the_plan_order(domain):
    _, plan = load_reference(domain)
    sequence = plan_to_sequence(plan)

    for earlier, later in zip(sequence.spans, sequence.spans[1:], strict=False):
        assert earlier.end < later.start


def test_a_step_occupies_a_single_line():
    """Boundaries that need no parsing to find cannot drift."""
    _, plan = load_reference()
    sequence = plan_to_sequence(plan)

    for span in sequence.spans:
        assert "\n" not in sequence.text[span.start : span.end]


# --- the mask -----------------------------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_mask_covers_the_flagged_steps_and_only_those(domain):
    task, plan = load_reference(domain)
    fan_in = next(step for step in plan.steps if len(step.input_from) > 1)
    broken = inject_broken_dependency(plan, step_id=fan_in.id, mode=UNKNOWN_MODE).broken_plan
    validation = validate_plan(broken, task)
    sequence = plan_to_sequence(broken)

    spec = mask_spec(sequence, validation.detected_step_ids())

    assert set(spec.masked_step_ids) == validation.detected_step_ids()
    assert set(spec.preserved_step_ids) == set(sequence.step_ids()) - set(spec.masked_step_ids)
    assert not set(spec.masked_step_ids) & set(spec.preserved_step_ids)
    assert len(spec.masked_step_ids) + len(spec.preserved_step_ids) == len(broken.steps)


def test_rendering_replaces_only_the_masked_spans():
    _, plan = load_reference()
    broken = inject_wrong_tool(plan, step_id="join").broken_plan
    sequence = plan_to_sequence(broken)
    spec = mask_spec(sequence, {"join"})

    rendered = render_masked(sequence, spec)

    assert rendered.count(DEFAULT_PLACEHOLDER) == 1
    assert '"id": "join"' not in rendered
    for step_id in spec.preserved_step_ids:
        assert f'"id": "{step_id}"' in rendered


def test_the_preserved_text_is_copied_verbatim():
    """The guarantee the approach rests on, checked character by character."""
    _, plan = load_reference()
    broken = inject_wrong_tool(plan, step_id="join").broken_plan
    sequence = plan_to_sequence(broken)
    spec = mask_spec(sequence, {"join"})

    filled = fill_masked(sequence, spec, {"join": json.dumps({"id": "join", "tool": "join",
                                                             "arguments": {}, "input_from":
                                                             ["n_csv", "n_db"], "produces":
                                                             ["join"]})})  # fmt: skip

    for span in sequence.spans:
        if span.step_id == "join":
            continue
        assert sequence.text[span.start : span.end] in filled


def test_a_masked_step_can_be_filled_with_nothing():
    _, plan = load_reference()
    broken = inject_duplicate_step(plan, step_id="agg").broken_plan
    sequence = plan_to_sequence(broken)
    spec = mask_spec(sequence, {"agg_dup"})

    filled = sequence_to_plan(fill_masked(sequence, spec, {"agg_dup": None}))

    assert [step.id for step in filled.steps] == [step.id for step in plan.steps]


def test_filling_an_unmasked_step_is_refused():
    """Anything outside the mask is out of reach by construction, not by convention."""
    _, plan = load_reference()
    sequence = plan_to_sequence(plan)
    spec = mask_spec(sequence, {"join"})

    with pytest.raises(ValueError, match="not masked"):
        fill_masked(sequence, spec, {"agg": "{}"})


def test_multi_error_masks_every_damaged_region():
    task, plan = load_reference()
    corruption = inject_multi(
        plan,
        [
            CorruptionSpec(corruption_type=WRONG_TOOL, step_id="join"),
            CorruptionSpec(corruption_type=STEP_DELETION, step_id="co"),
            CorruptionSpec(corruption_type=MISSING_STOP_CONDITION),
        ],
    )
    broken = corruption.broken_plan
    validation = validate_plan(broken, task)
    sequence = plan_to_sequence(broken)

    spec = mask_spec(sequence, validation.detected_step_ids())

    # join lost its tool, n_csv lost the step it consumed, cm lost its consumer.
    assert {"join", "n_csv", "cm"} <= set(spec.masked_step_ids)
    assert "agg" in spec.preserved_step_ids
    assert len(spec.spans) == len(spec.masked_step_ids)


def test_a_deleted_step_has_no_span_to_mask():
    """Named honestly rather than silently dropped: remasking cannot regenerate absent text."""
    _, plan = load_reference()
    broken = inject_step_deletion(plan, step_id="join").broken_plan
    sequence = plan_to_sequence(broken)

    spec = mask_spec(sequence, {"join", "enrich"})

    assert spec.masked_step_ids == ["enrich"]
    assert spec.unmaskable_step_ids == ["join"]


def test_an_empty_error_region_masks_nothing():
    """Coverage errors name no step, so there is nothing for a mask to cover."""
    _, plan = load_reference()
    sequence = plan_to_sequence(plan)

    spec = mask_spec(sequence, set())

    assert spec.masked_step_ids == []
    assert spec.preserved_step_ids == sequence.step_ids()
    assert spec.masked_characters == 0


def test_a_filled_sequence_that_is_not_a_plan_is_refused():
    _, plan = load_reference()
    sequence = plan_to_sequence(plan)
    spec = mask_spec(sequence, {"join"})

    filled = fill_masked(sequence, spec, {"join": "not a step at all"})

    with pytest.raises(PlanParseError):
        sequence_to_plan(filled)


# --- where steps meet -----------------------------------------------------------------------------
#
# A real run failed every case on this. Two adjacent steps were masked together, so the comma
# between them was inside the mask and came back as part of the regenerated text; reassembly then
# added its own and the sequence read "},,". The plans the model produced were fine — the parse
# was not. These pin the division of labour that settles it: a filling contributes one step's
# body, and the separators belong to the sequence it is put back into.


def kaggle_shaped(plan, step_id):
    """A filling as the LLaDA run actually produced them: leading space, trailing comma."""
    body = next(step for step in plan.steps if step.id == step_id).model_dump()
    return " " + json.dumps(body, ensure_ascii=False) + ","


def masked_pair():
    """domain A with a broken dependency — the validator flags two steps that sit side by side."""
    task, plan = load_reference(DATA_PIPELINE_A)
    broken = inject_broken_dependency(plan, step_id="dedupe", mode=UNKNOWN_MODE).broken_plan
    sequence = plan_to_sequence(broken)
    spec = mask_spec(sequence, validate_plan(broken, task).detected_step_ids())
    return plan, sequence, spec


def test_the_run_that_failed_on_a_doubled_comma_now_parses():
    """The regression this fix exists for, reproduced from the fillings the run recorded."""
    plan, sequence, spec = masked_pair()
    assert spec.masked_step_ids == ["e_news", "dedupe"]  # adjacent, so they share a separator
    fillings = {step_id: kaggle_shaped(plan, step_id) for step_id in spec.masked_step_ids}

    filled = fill_masked(sequence, spec, fillings)

    assert "},," not in filled
    assert sequence_to_plan(filled) == plan


def test_without_normalisation_the_same_fillings_produce_the_reported_error():
    """The failure is in the reassembly, not in what the model wrote — shown by doing it raw."""
    plan, sequence, spec = masked_pair()
    fillings = {step_id: kaggle_shaped(plan, step_id) for step_id in spec.masked_step_ids}

    unnormalised = _rebuild(sequence, fillings)

    assert "},," in unnormalised
    with pytest.raises(PlanParseError, match="line 15 column 123"):
        sequence_to_plan(unnormalised)


@pytest.mark.parametrize(
    ("decoration", "why"),
    [
        ("{body},", "a trailing comma"),
        ("{body}", "nothing at all"),
        (" {body} ", "surrounding spaces"),
        ("\n    {body},\n", "a newline and the next line's indent"),
        (", {body}", "a leading comma"),
        (" ,\n{body},\n ", "separators on both sides"),
    ],
)
def test_a_filling_parses_however_it_is_decorated(decoration, why):
    plan, sequence, spec = masked_pair()
    fillings = {
        step_id: decoration.replace(
            "{body}",
            json.dumps(
                next(s for s in plan.steps if s.id == step_id).model_dump(), ensure_ascii=False
            ),
        )
        for step_id in spec.masked_step_ids
    }

    filled = fill_masked(sequence, spec, fillings)

    assert sequence_to_plan(filled) == plan, why


def test_the_last_step_gets_no_trailing_comma():
    """A JSON array ends without one, whatever the filling arrived with."""
    _, plan = load_reference()
    sequence = plan_to_sequence(plan)
    terminal = plan.steps[-1]
    spec = mask_spec(sequence, {terminal.id})
    filling = json.dumps(terminal.model_dump(), ensure_ascii=False) + ","

    filled = fill_masked(sequence, spec, {terminal.id: filling})

    assert "}," not in filled.split('"steps": [')[1].rsplit("}", 1)[1]
    assert sequence_to_plan(filled) == plan


def test_a_middle_step_still_gets_its_comma():
    _, plan = load_reference()
    sequence = plan_to_sequence(plan)
    middle = plan.steps[5]
    spec = mask_spec(sequence, {middle.id})
    filling = json.dumps(middle.model_dump(), ensure_ascii=False)  # no comma of its own

    filled = fill_masked(sequence, spec, {middle.id: filling})

    assert sequence_to_plan(filled) == plan
    assert filled.count('"id"') == len(plan.steps)


def test_commas_inside_a_step_are_left_alone():
    """Only the outermost characters are touched — a step is full of commas that mean something."""
    _, plan = load_reference()
    sequence = plan_to_sequence(plan)
    spec = mask_spec(sequence, {"join"})
    join = next(step for step in plan.steps if step.id == "join")
    assert len(join.input_from) == 2  # the body genuinely contains a comma

    filled = fill_masked(
        sequence, spec, {"join": " " + json.dumps(join.model_dump(), ensure_ascii=False) + ","}
    )

    assert '"input_from": ["n_csv", "n_db"]' in filled
    assert sequence_to_plan(filled) == plan


def test_normalising_touches_only_the_ends():
    body = '{"id": "x", "input_from": ["a", "b"], "arguments": {"k": 1}}'

    assert normalise_filling(f" {body},") == body
    assert normalise_filling(f",\n  {body}\n,") == body
    assert normalise_filling(body) == body
    assert normalise_filling("") == ""
    assert normalise_filling(" , ") == ""


def test_normalising_does_not_repair_a_broken_filling():
    """Nothing is inserted or re-punctuated: text that is not a step still fails to parse."""
    _, plan = load_reference()
    sequence = plan_to_sequence(plan)
    spec = mask_spec(sequence, {"join"})

    filled = fill_masked(sequence, spec, {"join": ' {"id": "join", "tool": ,'})

    with pytest.raises(PlanParseError):
        sequence_to_plan(filled)


def test_healthy_steps_are_still_copied_verbatim_through_normalisation():
    """The guarantee of Ticket B-3a is untouched by any of this."""
    plan, sequence, spec = masked_pair()
    fillings = {step_id: kaggle_shaped(plan, step_id) for step_id in spec.masked_step_ids}

    filled = fill_masked(sequence, spec, fillings)

    for span in sequence.spans:
        if span.step_id in spec.masked_step_ids:
            continue
        assert sequence.text[span.start : span.end] in filled
