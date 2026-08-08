"""Field-level masking: the mask narrows to the broken field, and ``produces`` stays outside it.

This exists because of a measurement. In the B-stage run LLaDA returned valid plans and still
scored zero, because masking a whole step handed the model that step's ``produces`` tag, which it
rewrote into something plausible and different — ``dedup`` came back as ``deduplicated``. The tag
was never broken. Narrowing the mask to the field the validator actually pointed at is the fix,
and these tests pin the property that makes it a fix: the healthy fields are not regenerated.

What does *not* change is the fallback. Errors that name no field — ordering, duplicates, dangling
steps — still mask the whole step, because there is nothing narrower to mask.
"""

import json

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
from plan_repair.data import DATA_PIPELINE_A, DATA_PIPELINE_B, load_reference
from plan_repair.repair import (
    OracleBackend,
    OracleDiffusion,
    fill_masked,
    mask_spec_from_paths,
    plan_to_sequence,
    repair_and_score,
    sequence_to_plan,
)
from plan_repair.repair.diffusion import LLaDARepairer
from plan_repair.repair.remask import field_spans
from plan_repair.validation import validate_plan
from plan_repair.validation.paths import MASKABLE_FIELDS, ParsedPath, parse_path

DOMAINS = [DATA_PIPELINE_A, DATA_PIPELINE_B]


def fan_in_id(plan):
    """The step with more than one dependency — the one the B-stage failures were about."""
    return next(step.id for step in plan.steps if len(step.input_from) > 1)


def masked(domain, corrupt):
    """Corrupt a reference plan and build the mask the repairers would get."""
    task, plan = load_reference(domain)
    corruption = corrupt(plan)
    broken = corruption.broken_plan
    sequence = plan_to_sequence(broken)
    spec = mask_spec_from_paths(sequence, broken, validate_plan(broken, task).detected_paths())
    return task, plan, corruption, sequence, spec


def masked_characters(spec):
    return {position for span in spec.spans for position in range(span.start, span.end)}


def produces_span(plan, sequence, step_id):
    step = next(step for step in plan.steps if step.id == step_id)
    return field_spans(step, sequence.span_of(step_id).start)["produces"]


# --- reading a path back -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("$.steps[?join].tool", ParsedPath("join", "tool")),
        ("$.steps[?join].input_from", ParsedPath("join", "input_from")),
        ("$.steps[?join]", ParsedPath("join", None)),
        ("$.steps[?e_paper].produces", ParsedPath("e_paper", None)),
        ("$.steps[?join].arguments.limit", ParsedPath("join", None)),
        ("$.stop_condition", ParsedPath(None, None)),
        ("$.required_evidence[?csv_dataset]", ParsedPath(None, None)),
        ("$.goal", ParsedPath(None, None)),
        ("$.steps[?unterminated", ParsedPath(None, None)),
    ],
)
def test_a_path_reads_back_to_the_step_and_field_it_names(path, expected):
    assert parse_path(path) == expected


def test_an_unrecognised_suffix_widens_to_the_step_rather_than_masking_something_unintended():
    """``produces`` is deliberately not maskable: no path should ever narrow a mask onto it."""
    assert MASKABLE_FIELDS == ("tool", "input_from")
    assert parse_path("$.steps[?join].produces").field is None


def test_every_path_the_validator_emits_is_either_a_step_or_outside_the_plan():
    """An unreadable step path would silently widen a mask; an unreadable plan path is ignored."""
    task, plan = load_reference()
    corruptions = [
        inject_wrong_tool(plan, step_id="join"),
        inject_broken_dependency(plan, step_id="join", mode=UNKNOWN_MODE),
        inject_duplicate_step(plan, step_id="co"),
        inject_wrong_ordering(plan, step_id="join"),
        inject_missing_stop_condition(plan),
        inject_step_deletion(plan, step_id="n_csv"),
    ]

    for corruption in corruptions:
        broken = corruption.broken_plan
        ids = {step.id for step in broken.steps}
        for path in validate_plan(broken, task).detected_paths():
            parsed = parse_path(path)
            if path.startswith("$.steps["):
                assert parsed.step_id in ids or parsed.step_id, path
            else:
                assert parsed == ParsedPath(None, None), path


# --- where a field sits --------------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_field_boundaries_are_exact(domain):
    """Computed by re-rendering; checked against the characters actually in the sequence."""
    _, plan = load_reference(domain)
    sequence = plan_to_sequence(plan)

    for step, span in zip(plan.steps, sequence.spans, strict=True):
        boundaries = field_spans(step, span.start)

        assert set(boundaries) == {"id", "tool", "arguments", "input_from", "produces"}
        for name, (start, end) in boundaries.items():
            assert sequence.text[start:end] == json.dumps(
                getattr(step, name), ensure_ascii=False
            ), f"{domain}/{step.id}.{name}"
            assert span.start < start < end < span.end


# --- the mask narrows ----------------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_a_wrong_tool_masks_the_tool_and_nothing_else(domain):
    _, plan = load_reference(domain)
    target = fan_in_id(plan)

    _, _, _, sequence, spec = masked(domain, lambda p: inject_wrong_tool(p, step_id=target))

    assert {span.key for span in spec.spans} == {f"{target}.tool"}
    assert not masked_characters(spec) & set(range(*produces_span(plan, sequence, target)))


@pytest.mark.parametrize("domain", DOMAINS)
def test_a_broken_dependency_masks_input_from_and_no_produces(domain):
    """The dangling step the broken edge leaves behind has no field, so it is masked whole.

    Which is why the claim is stated per step: every step the mask *narrowed* keeps its tag.
    """
    _, plan = load_reference(domain)
    target = fan_in_id(plan)

    _, _, _, sequence, spec = masked(
        domain, lambda p: inject_broken_dependency(p, step_id=target, mode=UNKNOWN_MODE)
    )

    narrowed = {span.step_id for span in spec.spans if span.field is not None}
    covered = masked_characters(spec)

    assert f"{target}.input_from" in {span.key for span in spec.spans}
    assert narrowed == {target}
    for step_id in narrowed:
        assert not covered & set(range(*produces_span(plan, sequence, step_id))), step_id


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_mask_is_a_small_fraction_of_the_step_it_came_from(domain):
    """The narrowing this ticket is for, as a number rather than a claim."""
    _, plan = load_reference(domain)
    target = fan_in_id(plan)

    _, _, _, sequence, spec = masked(domain, lambda p: inject_wrong_tool(p, step_id=target))
    step_span = sequence.span_of(target)

    assert spec.masked_characters < (step_span.end - step_span.start) / 5


# --- the fallback stays --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("corrupt", "why"),
    [
        (lambda p: inject_duplicate_step(p, step_id="co"), "a duplicate names no field"),
        (lambda p: inject_wrong_ordering(p, step_id="join"), "order is not a field"),
    ],
)
def test_an_error_with_no_field_still_masks_the_whole_step(corrupt, why):
    _, _, _, _, spec = masked(DATA_PIPELINE_B, corrupt)

    assert spec.spans, why
    assert all(span.field is None for span in spec.spans), why
    assert {span.key for span in spec.spans} == set(spec.masked_step_ids), why


def test_a_plan_level_error_still_masks_nothing():
    _, _, _, _, spec = masked(DATA_PIPELINE_B, inject_missing_stop_condition)

    assert spec.spans == []
    assert spec.masked_step_ids == []


def test_a_deleted_step_is_still_out_of_reach():
    """Narrowing changes what is masked, not what can be masked at all."""
    _, _, _, _, spec = masked(DATA_PIPELINE_B, lambda p: inject_step_deletion(p, step_id="n_csv"))

    assert "n_csv" not in spec.masked_step_ids


def test_a_whole_step_request_wins_over_a_field_request_for_the_same_step():
    """Two paths can disagree about how much to take; a mask inside a mask means nothing."""
    _, plan = load_reference()
    sequence = plan_to_sequence(plan)

    spec = mask_spec_from_paths(sequence, plan, ["$.steps[?join].tool", "$.steps[?join]"])

    assert [span.field for span in spec.spans] == [None]
    assert spec.spans[0] == sequence.span_of("join")


# --- putting it back -----------------------------------------------------------------------


def test_replacing_a_field_leaves_every_other_field_byte_for_byte():
    """The whole point: ``produces`` comes back because it never left."""
    _, plan = load_reference()
    sequence = plan_to_sequence(plan)
    spec = mask_spec_from_paths(sequence, plan, ["$.steps[?join].tool"])
    before = next(line for line in sequence.text.splitlines() if '"id": "join"' in line)

    filled = fill_masked(sequence, spec, {"join.tool": '"joiner"'}, plan)
    after = next(line for line in filled.splitlines() if '"id": "join"' in line)

    assert before.replace('"join_merge"', '"joiner"') == after or '"tool": "joiner"' in after
    assert '"produces": ["join"]' in after
    assert '"input_from": ["n_csv", "n_db"]' in after
    assert sequence_to_plan(filled).steps[11].tool == "joiner"


@pytest.mark.parametrize(
    ("filling", "why"),
    [
        ('"joiner"', "no decoration"),
        (' "joiner",', "a leading space and the field separator"),
        (',"joiner"', "a leading comma"),
        ('\n  "joiner" ,\n', "newlines on both sides"),
    ],
)
def test_a_field_filling_parses_however_it_is_decorated(filling, why):
    """The separator between two fields belongs to the template, as it does between two steps."""
    _, plan = load_reference()
    sequence = plan_to_sequence(plan)
    spec = mask_spec_from_paths(sequence, plan, ["$.steps[?join].tool"])

    filled = fill_masked(sequence, spec, {"join.tool": filling}, plan)

    assert ",," not in filled and ", ," not in filled, why
    assert sequence_to_plan(filled).steps[11].tool == "joiner", why


def test_two_fields_of_one_step_are_replaced_together():
    _, plan = load_reference()
    sequence = plan_to_sequence(plan)
    spec = mask_spec_from_paths(
        sequence, plan, ["$.steps[?join].tool", "$.steps[?join].input_from"]
    )

    filled = fill_masked(
        sequence, spec, {"join.tool": '"joiner"', "join.input_from": '["n_csv"]'}, plan
    )
    repaired = sequence_to_plan(filled)

    assert (repaired.steps[11].tool, repaired.steps[11].input_from) == ("joiner", ["n_csv"])
    assert repaired.steps[11].produces == ["join"]


def test_a_field_mask_and_a_whole_step_mask_coexist_in_one_sequence():
    """What a broken dependency actually produces: one narrowed step, one dangling step."""
    task, plan, _, sequence, spec = masked(
        DATA_PIPELINE_B, lambda p: inject_broken_dependency(p, step_id="join", mode=UNKNOWN_MODE)
    )
    original = {step.id: step for step in plan.steps}

    filled = fill_masked(
        sequence,
        spec,
        {
            span.key: json.dumps(
                original[span.step_id].model_dump()
                if span.field is None
                else getattr(original[span.step_id], span.field),
                ensure_ascii=False,
            )
            for span in spec.spans
        },
        plan,
    )

    assert sequence_to_plan(filled) == plan
    assert not validate_plan(sequence_to_plan(filled), task).errors


def test_filling_a_field_that_was_not_masked_is_refused():
    _, plan = load_reference()
    sequence = plan_to_sequence(plan)
    spec = mask_spec_from_paths(sequence, plan, ["$.steps[?join].tool"])

    with pytest.raises(ValueError, match="not masked"):
        fill_masked(sequence, spec, {"join.produces": '["anything"]'}, plan)


# --- end to end ----------------------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize(
    ("corrupt", "label"),
    [
        (lambda p, target: inject_wrong_tool(p, step_id=target), "wrong_tool"),
        (
            lambda p, target: inject_broken_dependency(p, step_id=target, mode=UNKNOWN_MODE),
            "broken_dependency",
        ),
    ],
)
def test_the_oracle_repairs_through_a_field_mask_without_collateral(corrupt, label, domain):
    task, plan = load_reference(domain)
    corruption = corrupt(plan, fan_in_id(plan))

    repaired, score = repair_and_score(
        OracleDiffusion(plan),
        reference_plan=plan,
        broken_plan=corruption.broken_plan,
        task=task,
        damaged_step_ids=corruption.injected[0].damaged_step_ids,
    )

    assert score.solved, label
    assert (score.collateral_modified, score.collateral_renamed, score.collateral_removed) == (
        0,
        0,
        0,
    ), label
    assert repaired == plan, label


def test_the_backend_is_handed_one_field_not_a_step():
    """What a real model will see — the reason the ``produces`` tag can no longer be rewritten."""
    task, plan = load_reference()
    broken = inject_wrong_tool(plan, step_id="join").broken_plan
    seen: dict[str, list[str]] = {}

    class Recording(OracleBackend):
        name = "recording"

        def fill(self, request):
            seen["keys"] = [span.key for span in request.mask.spans]
            seen["text"] = [
                request.sequence.text[span.start : span.end] for span in request.mask.spans
            ]
            return super().fill(request)

    LLaDARepairer(Recording(plan)).repair(broken, validate_plan(broken, task), task)

    assert seen["keys"] == ["join.tool"]
    assert seen["text"] == ['"join_x"']


def test_a_cycle_narrows_every_member_to_its_edges_without_becoming_a_narrow_error():
    """Narrowing does not make a wide error small — it only spares the fields nobody flagged.

    A cycle implicates every step that can reach it, so fifteen steps are masked either way. What
    changes is that each contributes its ``input_from`` rather than its whole body.
    """
    _, plan, _, sequence, spec = masked(
        DATA_PIPELINE_B,
        lambda p: inject_broken_dependency(
            p, step_id="l_csv", mode=CYCLE_MODE, cycle_with="report"
        ),
    )
    covered = masked_characters(spec)

    assert len(spec.spans) == 15
    assert {span.field for span in spec.spans} == {"input_from"}
    for span in spec.spans:
        assert not covered & set(range(*produces_span(plan, sequence, span.step_id)))
