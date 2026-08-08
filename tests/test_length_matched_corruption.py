"""The length-matched wrong_tool corruption — the control for a measurement about mask length.

The default corruption appends ``_x``, which costs exactly one token, so the region handed to the
model is always one token longer than the answer needs. Both models were seen writing the answer
and then filling the leftover cell: ``join`` came back as ``join_db``. This variant removes that
surplus and changes nothing else, so a repair that succeeds here and fails there says the surplus
was the cause.

Two conditions have to hold together, and each one alone is a trap:

* **the replacement is not in the task's tools.** A same-length name taken from ``available_tools``
  leaves the plan valid — the validator reports nothing, so there is no path, no mask and no
  repair, and a repairer that does nothing scores as having solved it. That is checked here,
  because it is the reason the pool is what it is.
* **the replacement costs what the answer costs.** Otherwise the mask is not the length under
  test, and the run measures something else while looking like it measured this.
"""

import json

import pytest

from plan_repair.canonical import canonicalize
from plan_repair.corruption import (
    LENGTH_MATCHED_MODE,
    CorruptionNotApplicableError,
    VocabularyLength,
    inject_wrong_tool,
    inject_wrong_tool_length_matched,
    length_matched_tools,
)
from plan_repair.data import (
    DATA_PIPELINE_A,
    DATA_PIPELINE_B,
    all_tool_names,
    load_reference,
)
from plan_repair.repair import IdentityRepairer, repair_and_score
from plan_repair.schema import WRONG_TOOL
from plan_repair.validation import UNKNOWN_TOOL, tool_path, validate_plan

DOMAINS = [DATA_PIPELINE_A, DATA_PIPELINE_B]


class Syllables:
    """A vocabulary small enough to reason about: one token per ``_``-part, plus both quotes.

    ``"join"`` costs 3 and ``"clean_missing"`` costs 4, which is enough structure to tell a match
    from a near miss without asking what a real vocabulary happens to do.
    """

    name = "syllables"

    def token_length(self, text: str) -> int:
        return 2 + len(text.strip('"').split("_"))


def fan_in(plan):
    return next(step.id for step in plan.steps if len(step.input_from) > 1)


def matched(domain, token_length=None):
    task, plan = load_reference(domain)
    result = inject_wrong_tool_length_matched(
        plan,
        task,
        step_id=fan_in(plan),
        pool=all_tool_names(),
        token_length=token_length or Syllables(),
    )
    return task, plan, result


def tool_of(plan, step_id):
    return next(step.tool for step in plan.steps if step.id == step_id)


# --- why the pool sits outside the task's tools ----------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_a_replacement_from_the_tasks_own_tools_would_not_be_an_error_at_all(domain):
    """The reason this corruption cannot draw from ``available_tools``, stated as a measurement.

    Every same-length tool the task offers leaves the plan valid. No error means no path, no
    mask, and nothing for a repairer to do — while the scoring would call it solved.
    """
    task, plan = load_reference(domain)
    target = fan_in(plan)
    length = Syllables()
    wanted = length.token_length(json.dumps(tool_of(plan, target)))
    same_length_and_allowed = [
        name
        for name in sorted(task.tool_names())
        if name != tool_of(plan, target) and length.token_length(json.dumps(name)) == wanted
    ]
    assert same_length_and_allowed, "the trap needs candidates to exist to be a trap"

    for name in same_length_and_allowed:
        broken = inject_wrong_tool(plan, step_id=target, new_tool=name).broken_plan

        assert validate_plan(broken, task).valid, name


def test_a_valid_replacement_scores_as_solved_without_any_repair():
    """What the previous test costs if it is ignored: a control group that always succeeds."""
    task, plan = load_reference(DATA_PIPELINE_B)
    corruption = inject_wrong_tool(plan, step_id="join", new_tool="aggregate")

    _, score = repair_and_score(
        IdentityRepairer(),
        reference_plan=plan,
        broken_plan=corruption.broken_plan,
        task=task,
        damaged_step_ids=corruption.injected[0].damaged_step_ids,
    )

    assert score.solved and score.errors_remaining == 0


# --- picking the replacement -------------------------------------------------------------------


def test_candidates_are_outside_the_task_the_right_length_and_sorted():
    task, _ = load_reference(DATA_PIPELINE_B)
    length = Syllables()

    candidates = length_matched_tools("join", task=task, pool=all_tool_names(), token_length=length)

    assert candidates == sorted(candidates)
    assert candidates
    for name in candidates:
        assert name not in task.tool_names(), name
        assert length.token_length(json.dumps(name)) == length.token_length('"join"'), name


def test_the_tool_being_replaced_is_never_its_own_replacement():
    """A corruption that writes back the answer is not a corruption."""
    task, _ = load_reference(DATA_PIPELINE_B)

    assert "join" not in length_matched_tools(
        "join", task=task, pool={"join", *all_tool_names()}, token_length=Syllables()
    )


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_corruption_is_wrong_unavailable_and_the_same_length(domain):
    task, plan, result = matched(domain)
    target = fan_in(plan)
    answer = tool_of(plan, target)
    written = tool_of(result.broken_plan, target)
    length = Syllables()

    assert written != answer
    assert written not in task.tool_names()
    assert length.token_length(json.dumps(written)) == length.token_length(json.dumps(answer))


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_validator_reports_it_exactly_where_the_mask_is_built_from(domain):
    task, plan, result = matched(domain)
    target = fan_in(plan)

    validation = validate_plan(result.broken_plan, task)

    assert [error.type for error in validation.errors] == [UNKNOWN_TOOL]
    assert validation.detected_paths() == {tool_path(target)}


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_metadata_says_which_vocabulary_matched_it(domain):
    """A count means nothing without the vocabulary that produced it."""
    _, plan, result = matched(domain)
    target = fan_in(plan)
    injected = result.injected[0]

    assert injected.corruption_type == WRONG_TOOL
    assert injected.damaged_step_ids == [target]
    assert injected.damaged_paths == [tool_path(target)]
    assert injected.detail["mode"] == LENGTH_MATCHED_MODE
    assert injected.detail["matched_with"] == "syllables"
    assert injected.detail["original_tool"] == tool_of(plan, target)
    assert injected.detail["new_tool"] == tool_of(result.broken_plan, target)
    assert injected.detail["new_tool"] in injected.detail["candidates"]


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_original_plan_is_untouched(domain):
    _, plan, result = matched(domain)

    assert canonicalize(plan)[0] == canonicalize(load_reference(domain)[1])[0]
    assert len(result.preserved_step_ids) == len(plan.steps) - 1


def test_the_choice_repeats():
    assert matched(DATA_PIPELINE_B)[2].broken_plan == matched(DATA_PIPELINE_B)[2].broken_plan


# --- no candidate ------------------------------------------------------------------------------


def test_a_step_with_no_name_of_the_right_length_is_refused_not_approximated():
    """Settling for a different length would leave the experiment's premise out of the run."""
    task, plan = load_reference(DATA_PIPELINE_B)

    with pytest.raises(CorruptionNotApplicableError, match="costs the 3 syllables tokens"):
        inject_wrong_tool_length_matched(
            plan,
            task,
            step_id="join",
            pool={"a_much_longer_name_than_this"},
            token_length=Syllables(),
        )


def test_an_empty_pool_is_refused():
    task, plan = load_reference(DATA_PIPELINE_B)

    with pytest.raises(CorruptionNotApplicableError):
        inject_wrong_tool_length_matched(
            plan, task, step_id="join", pool=(), token_length=Syllables()
        )


# --- the pool and the adapter --------------------------------------------------------------------


def test_the_pool_is_every_reference_tool_name():
    names = all_tool_names()

    assert load_reference(DATA_PIPELINE_A)[0].tool_names() <= names
    assert load_reference(DATA_PIPELINE_B)[0].tool_names() <= names
    assert "cite" in names and "join" in names


def test_the_vocabulary_adapter_counts_what_a_tokenizer_returns():
    """Duck-typed on purpose, so this package needs no transformers to be tested."""

    class Fake:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(len(text)))}

    length = VocabularyLength(Fake(), "fake")

    assert length.name == "fake"
    assert length.token_length("abcd") == 4


# --- the existing corruption is untouched ----------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_suffix_corruption_still_does_exactly_what_it_did(domain):
    """Every measurement before this ticket was taken against ``_x``; it has to stay comparable."""
    _, plan = load_reference(domain)
    target = fan_in(plan)

    result = inject_wrong_tool(plan, step_id=target)

    assert tool_of(result.broken_plan, target) == f"{tool_of(plan, target)}_x"
    assert result.injected[0].detail == {
        "original_tool": tool_of(plan, target),
        "new_tool": f"{tool_of(plan, target)}_x",
    }


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_two_corruptions_damage_the_same_step_in_different_ways(domain):
    _, plan, matched_result = matched(domain)
    target = fan_in(plan)
    suffix_result = inject_wrong_tool(plan, step_id=target)

    assert matched_result.injected[0].damaged_paths == suffix_result.injected[0].damaged_paths
    assert matched_result.broken_plan != suffix_result.broken_plan
