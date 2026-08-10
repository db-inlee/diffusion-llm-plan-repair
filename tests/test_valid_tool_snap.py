"""The valid-tool snap: a filled tool name is completed only when the value clearly completes it.

Ticket C-3 established what the failure was. The model writes the right tool and then has a cell
left over, so ``join`` comes back as ``join_db`` and ``dedupe`` as ``deduplicate`` — the answer is
there, at the front, with surplus attached. D-1 removes the surplus by snapping such a value to
the valid tool it reproduces, and the whole question is where to draw the line: a snap that always
picks the nearest name would let a model that wrote nonsense pass, which is cheating rather than
repairing.

The line drawn here is a **prefix completion**: the value has to reproduce a valid name from the
front, all of it or all but the last character, and no other valid name may do as well. That is
why ``merge_join`` is *not* snapped even though ``join`` is the right answer and sits inside it —
finding a name hidden in the middle is a search, not a completion, and it is the kind of licence
that turns the snap into a rubber stamp.

The threshold is not fitted to the four observations: no two valid tool names in either domain
reach it, which is pinned below so that a vocabulary change cannot quietly make the snap
ambiguous.
"""

import json

import pytest

from plan_repair.corruption import inject_wrong_ordering, inject_wrong_tool
from plan_repair.data import DATA_PIPELINE_A, DATA_PIPELINE_B, all_tool_names, load_reference
from plan_repair.repair import mask_spec_from_paths, plan_to_sequence, repair_and_score
from plan_repair.repair.diffusion import LLaDARepairer
from plan_repair.repair.snap import (
    ALREADY_VALID,
    AMBIGUOUS,
    BELOW_FLOOR,
    NOT_A_STRING,
    SNAP_RATIO_FLOOR,
    SNAPPED,
    prefix_ratio,
    snap_tool_fillings,
    snap_tool_value,
)
from plan_repair.validation import validate_plan

DOMAIN_TOOLS = {
    domain: load_reference(domain)[0].tool_names() for domain in (DATA_PIPELINE_A, DATA_PIPELINE_B)
}


def tools(domain):
    return DOMAIN_TOOLS[domain]


# --- the ratio itself ------------------------------------------------------------------------


def test_a_value_that_reproduces_the_whole_name_scores_one():
    assert prefix_ratio("join_db", "join") == 1.0


def test_a_value_one_character_short_scores_just_under_one():
    # 'deduplicate' shares 'dedup' with 'dedupe' — five of its six characters.
    assert prefix_ratio("deduplicate", "dedupe") == pytest.approx(5 / 6)


def test_a_name_hidden_after_the_start_scores_nothing():
    assert prefix_ratio("merge_join", "join") == 0.0


# --- the snap decision -----------------------------------------------------------------------


def test_a_surplus_suffix_is_snapped_away():
    decision = snap_tool_value(' "join_db",', tools(DATA_PIPELINE_B))
    assert decision.snapped == "join"
    assert decision.ratio == 1.0
    assert decision.reason == SNAPPED


def test_a_value_that_is_one_character_short_is_still_a_completion():
    decision = snap_tool_value(' "deduplicate",', tools(DATA_PIPELINE_A))
    assert decision.snapped == "dedupe"
    assert decision.ratio == pytest.approx(5 / 6)


def test_a_name_hidden_behind_another_word_is_not_snapped():
    """``merge_join`` should stay broken. Finding ``join`` inside it would be a search."""
    decision = snap_tool_value(' "merge_join",', tools(DATA_PIPELINE_B))
    assert decision.snapped is None
    assert decision.reason == BELOW_FLOOR


def test_a_value_that_is_already_a_valid_tool_is_left_alone():
    decision = snap_tool_value(' "join",', tools(DATA_PIPELINE_B))
    assert decision.snapped is None
    assert decision.reason == ALREADY_VALID


def test_a_value_that_is_not_a_json_string_is_left_alone():
    decision = snap_tool_value(' "jo', tools(DATA_PIPELINE_B))
    assert decision.snapped is None
    assert decision.reason == NOT_A_STRING


def test_two_names_completed_equally_well_snap_to_neither():
    """Uniqueness is half the rule: a tie is exactly the case the snap must not decide.

    ``load_csvx`` reproduces both ``load_csv`` and ``load_cs`` in full, so both score 1.0 and
    neither is the answer. No pair in the real vocabularies nests like this — which is what the
    collision tests below pin — but the rule has to refuse it rather than pick the first.
    """
    decision = snap_tool_value('"load_csvx"', {"load_csv", "load_cs", "join"})
    assert decision.snapped is None
    assert decision.reason == AMBIGUOUS


def test_the_runner_up_is_recorded_so_a_refusal_can_be_read():
    decision = snap_tool_value(' "merge_join",', tools(DATA_PIPELINE_B))
    assert decision.ratio == 0.0
    assert decision.runner_up == 0.0
    assert decision.original == "merge_join"


# --- the threshold is a property of the vocabulary, not of the four observations ---------------


@pytest.mark.parametrize("domain", [DATA_PIPELINE_A, DATA_PIPELINE_B])
def test_no_valid_tool_completes_another_valid_tool(domain):
    """The floor sits above the vocabulary's own confusability, so a valid name never snaps.

    Without this the threshold would be a number fitted to four measurements. With it, a tool
    added later that collides with an existing one fails here rather than silently making the
    snap wrong.
    """
    names = sorted(tools(domain))
    collisions = [
        (value, other)
        for value in names
        for other in names
        if value != other and prefix_ratio(value, other) >= SNAP_RATIO_FLOOR
    ]
    assert collisions == []


def test_no_tool_in_the_whole_pool_completes_another():
    names = sorted(all_tool_names())
    collisions = [
        (value, other)
        for value in names
        for other in names
        if value != other and prefix_ratio(value, other) >= SNAP_RATIO_FLOOR
    ]
    assert collisions == []


def test_the_replacement_is_written_back_as_json():
    """The filling goes back into a JSON line, so the snap has to hand over a rendered value."""
    decision = snap_tool_value(' "join_db",', tools(DATA_PIPELINE_B))
    assert decision.replacement == json.dumps("join")


# --- applying it to a mask's fillings ---------------------------------------------------------


def masked(domain, corrupt):
    """Corrupt a reference plan and build the mask a repairer would be given."""
    task, plan = load_reference(domain)
    broken = corrupt(plan).broken_plan
    sequence = plan_to_sequence(broken)
    spec = mask_spec_from_paths(sequence, broken, validate_plan(broken, task).detected_paths())
    return task, plan, broken, sequence, spec


def wrong_tool(plan):
    return inject_wrong_tool(plan, step_id=fan_in_id(plan))


def fan_in_id(plan):
    return next(step.id for step in plan.steps if len(step.input_from) > 1)


def test_only_the_tool_fields_of_a_mask_are_snapped():
    """A whole-step mask carries no tool field, so nothing in it may be rewritten."""

    def misordered(plan):
        return inject_wrong_ordering(plan, step_id=fan_in_id(plan))

    task, _, _, _, spec = masked(DATA_PIPELINE_B, misordered)
    assert [span.field for span in spec.spans] == [None]
    filling = {spec.spans[0].key: '{"id": "join", "tool": "join_db"}'}
    snapped, record = snap_tool_fillings(filling, spec, task.tool_names())
    assert snapped == filling
    assert record == {}


def test_a_dropped_step_is_not_snapped():
    task, _, _, _, spec = masked(DATA_PIPELINE_B, wrong_tool)
    snapped, record = snap_tool_fillings({spec.spans[0].key: None}, spec, task.tool_names())
    assert snapped == {spec.spans[0].key: None}
    assert record == {}


def test_the_record_names_the_span_that_was_snapped():
    task, _, _, _, spec = masked(DATA_PIPELINE_B, wrong_tool)
    key = spec.spans[0].key
    snapped, record = snap_tool_fillings({key: ' "join_db",'}, spec, task.tool_names())
    assert key == "join.tool"
    assert snapped[key] == '"join"'
    assert record[key].fired
    assert record[key].original == "join_db"


def test_a_refusal_is_recorded_too():
    """Otherwise the cost of being conservative cannot be counted afterwards."""
    task, _, _, _, spec = masked(DATA_PIPELINE_B, wrong_tool)
    key = spec.spans[0].key
    snapped, record = snap_tool_fillings({key: ' "merge_join",'}, spec, task.tool_names())
    assert snapped[key] == ' "merge_join",'
    assert not record[key].fired
    assert record[key].reason == BELOW_FLOOR


# --- the repairer ------------------------------------------------------------------------------


class RecordedBackend:
    """A backend that returns a filling that was measured, so a run needs no model."""

    name = "recorded"

    def __init__(self, filling):
        self._filling = filling

    def fill(self, request):
        return {span.key: self._filling for span in request.mask.spans}


def repaired_with(domain, filling, *, snap_tools):
    """Run one wrong_tool repair on a recorded filling and score what comes back."""
    task, plan, broken, _, _ = masked(domain, wrong_tool)
    repairer = LLaDARepairer(RecordedBackend(filling), CharTokenizer(), snap_tools=snap_tools)
    repaired, score = repair_and_score(
        repairer,
        reference_plan=plan,
        broken_plan=broken,
        task=task,
        damaged_step_ids=[fan_in_id(plan)],
    )
    return repairer, repaired, score


class CharTokenizer:
    """One token per character. The snap works on characters, so the vocabulary is irrelevant."""

    name = "char"

    def encode_with_offsets(self, text):
        return [ord(character) for character in text], [(i, i + 1) for i in range(len(text))]


def test_the_snap_is_off_unless_it_is_asked_for():
    """Every measurement before this ticket was taken without it, and has to stay reproducible."""
    _, _, score = repaired_with(DATA_PIPELINE_B, ' "join_db",', snap_tools=False)
    assert not score.solved
    assert score.errors_remaining == 1


def test_a_surplus_suffix_is_repaired_when_the_snap_is_on():
    _, repaired, score = repaired_with(DATA_PIPELINE_B, ' "join_db",', snap_tools=True)
    assert score.solved
    assert score.collateral_total == 0
    assert score.damaged_restored == 1
    assert next(step.tool for step in repaired.steps if step.id == "join") == "join"


def test_a_value_the_snap_refuses_still_fails():
    _, _, score = repaired_with(DATA_PIPELINE_B, ' "merge_join",', snap_tools=True)
    assert not score.solved
    assert score.errors_remaining == 1


def test_the_snap_records_what_it_did_in_the_diagnostics():
    repairer, _, _ = repaired_with(DATA_PIPELINE_B, ' "join_db",', snap_tools=True)
    assert repairer.last_diagnostics is not None
    snaps = repairer.last_diagnostics.snaps
    assert snaps["join.tool"].snapped == "join"
    assert snaps["join.tool"].original == "join_db"
    assert snaps["join.tool"].ratio == 1.0


def test_the_diagnostics_keep_the_model_output_that_the_snap_replaced():
    """Otherwise a solved case could not be read as the model's answer or as the snap's."""
    repairer, _, _ = repaired_with(DATA_PIPELINE_B, ' "join_db",', snap_tools=True)
    diagnostics = repairer.last_diagnostics
    assert diagnostics.fillings["join.tool"] == ' "join_db",'
    assert '"tool": "join"' in diagnostics.raw_text


def test_a_backend_that_changes_nothing_is_rescued_by_the_snap():
    """Why the snap is opt-in: with it on, doing nothing scores as a repair.

    ``EchoBackend`` hands back the broken value, and the corruption's ``_x`` suffix is exactly the
    kind of surplus the snap removes — so the control that is supposed to measure the plumbing
    would come out solved. The controls therefore run with the snap off, and this pins the reason.
    """
    _, _, without = repaired_with(DATA_PIPELINE_B, ' "join_x",', snap_tools=False)
    _, _, with_snap = repaired_with(DATA_PIPELINE_B, ' "join_x",', snap_tools=True)
    assert not without.solved
    assert with_snap.solved


# --- offline replay of the measured fillings ---------------------------------------------------

# What LLaDA actually returned, read from the result files rather than retyped:
#   results/diffusion_c1/c1_remeasure.json  (field mask, no hint)
#   results/diffusion_c2/c2_remeasure.json  (field mask + valid-tool hint)
#   results/diffusion_c3/c3_remeasure.json  (length-matched corruption)
# Replaying them is a calculation about outputs already on disk, not a new measurement: it says
# what the snap would have done to those answers, not what the model would write next time.
MEASURED = [
    ("C-1 domain A", DATA_PIPELINE_A, ' "deduplicate",', False),
    ("C-1 domain B", DATA_PIPELINE_B, ' "merge_join",', False),
    ("C-2 domain A", DATA_PIPELINE_A, ' "deduplicate",', False),
    ("C-2 domain B", DATA_PIPELINE_B, ' "join_db",', False),
    ("C-3 domain A", DATA_PIPELINE_A, ' "dedupe",', True),
    ("C-3 domain B", DATA_PIPELINE_B, ' "join",', True),
]


@pytest.mark.parametrize(("label", "domain", "filling", "solved_before"), MEASURED)
def test_the_snap_never_makes_a_measured_case_worse(label, domain, filling, solved_before):
    _, _, before = repaired_with(domain, filling, snap_tools=False)
    _, _, after = repaired_with(domain, filling, snap_tools=True)
    assert before.solved is solved_before, label
    assert after.solved or not before.solved, label
    assert after.collateral_total == 0, label


def test_the_snap_repairs_three_of_the_four_measured_failures():
    """The ticket's expectation, as a calculation over the recorded outputs."""
    failures = [case for case in MEASURED if not case[3]]
    solved = [
        label
        for label, domain, filling, _ in failures
        if repaired_with(domain, filling, snap_tools=True)[2].solved
    ]
    assert len(failures) == 4
    assert sorted(solved) == ["C-1 domain A", "C-2 domain A", "C-2 domain B"]


def test_the_length_matched_cases_are_untouched_by_the_snap():
    """C-3 was already solved without the snap, and must not be counted as its doing."""
    for label, domain, filling, solved_before in MEASURED:
        if not solved_before:
            continue
        repairer, _, score = repaired_with(domain, filling, snap_tools=True)
        assert score.solved, label
        assert not any(snap.fired for snap in repairer.last_diagnostics.snaps.values()), label
