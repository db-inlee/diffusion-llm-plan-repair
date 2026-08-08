"""Diagnostics for a repair that did not parse.

The situation these exist for: every case of a real run failed with "the answer was not valid
JSON: Expecting value: line 15 column 123" and nothing else, so there was no way to tell an
unfinished denoise from a broken boundary from a model that cannot hold the format.

The backends here return deliberately damaged text — a step that is still half mask, a step whose
JSON is truncated — so each diagnostic can be checked against an answer that is known in advance.
No model and no GPU: what is under test is the recording, not the model.
"""

import json

import pytest

from plan_repair.corruption import UNKNOWN_MODE, inject_broken_dependency, inject_wrong_tool
from plan_repair.data import DATA_PIPELINE_A, DATA_PIPELINE_B, load_reference
from plan_repair.repair import (
    LLaDARepairer,
    OracleBackend,
    RepairDiagnostics,
    diagnose,
    excerpt_around,
    repair_and_score,
    summarise,
)
from plan_repair.repair.diffusion import LLADA_MASK_TOKEN
from plan_repair.repair.plan_io import PlanParseError, parse_plan

DOMAINS = [DATA_PIPELINE_A, DATA_PIPELINE_B]


class ScriptedFillBackend:
    """Puts a prepared string into every masked span."""

    name = "scripted"

    def __init__(self, filling: str) -> None:
        self._filling = filling

    def fill(self, request):
        return dict.fromkeys((span.key for span in request.mask.spans), self._filling)


def corrupted(domain=DATA_PIPELINE_B):
    task, plan = load_reference(domain)
    fan_in = next(step for step in plan.steps if len(step.input_from) > 1)
    return task, plan, inject_broken_dependency(plan, step_id=fan_in.id, mode=UNKNOWN_MODE)


def repair_with(backend, task, plan, corruption):
    repairer = LLaDARepairer(backend)
    _, score = repair_and_score(
        repairer,
        reference_plan=plan,
        broken_plan=corruption.broken_plan,
        task=task,
        damaged_step_ids=corruption.injected[0].damaged_step_ids,
    )
    return repairer, score


# --- the raw output is kept --------------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_a_successful_repair_still_records_what_the_model_produced(domain):
    """Kept on the way through, so a good output can be compared against a bad one later."""
    task, plan, corruption = corrupted(domain)

    repairer, score = repair_with(OracleBackend(plan), task, plan, corruption)

    assert score.solved
    assert repairer.last_diagnostics is not None
    assert repairer.last_diagnostics.parsed is True
    assert repairer.last_diagnostics.parse_failure is None
    assert repairer.last_diagnostics.raw_length == len(repairer.last_diagnostics.raw_text)
    assert parse_plan(repairer.last_diagnostics.raw_text) == plan


def test_a_failed_repair_records_the_text_that_failed():
    """The point of the whole change: the repaired plan is a fallback, this is the real output."""
    task, plan, corruption = corrupted()
    broken_json = '{"id": "join", "tool": "join", "arguments": {'

    repairer, score = repair_with(ScriptedFillBackend(broken_json), task, plan, corruption)

    assert not score.solved
    diagnostics = repairer.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.parsed is False
    assert broken_json in diagnostics.raw_text
    assert set(diagnostics.fillings) == {span.key for span in repairer.last_mask.spans}
    assert all(value == broken_json for value in diagnostics.fillings.values())


def test_the_raw_text_is_kept_whole():
    task, plan, corruption = corrupted()

    repairer, _ = repair_with(OracleBackend(plan), task, plan, corruption)

    raw = repairer.last_diagnostics.raw_text
    assert raw.startswith("{\n")
    assert raw.rstrip().endswith("}")
    assert raw.count('"id"') == len(plan.steps)  # nothing was truncated


# --- mask tokens left behind ---------------------------------------------------------------------


def test_mask_tokens_surviving_into_the_output_are_counted():
    """Above zero means denoising stopped before the sequence was finished."""
    task, plan, corruption = corrupted()
    half_filled = f'{{"id": "join", "tool": {LLADA_MASK_TOKEN}{LLADA_MASK_TOKEN}, "x": 1}}'

    repairer, _ = repair_with(ScriptedFillBackend(half_filled), task, plan, corruption)

    diagnostics = repairer.last_diagnostics
    assert diagnostics.mask_tokens_remaining == 2 * len(repairer.last_mask.spans)
    assert diagnostics.unfinished_denoising is True
    assert diagnostics.mask_token == LLADA_MASK_TOKEN


def test_a_finished_denoise_leaves_no_mask_tokens():
    task, plan, corruption = corrupted()

    repairer, _ = repair_with(OracleBackend(plan), task, plan, corruption)

    assert repairer.last_diagnostics.mask_tokens_remaining == 0
    assert repairer.last_diagnostics.unfinished_denoising is False


def test_the_two_causes_are_told_apart():
    """A parse failure with mask tokens is a budget problem; one without is not."""
    task, plan, corruption = corrupted()
    unfinished = f'{{"id": "join", {LLADA_MASK_TOKEN}}}'
    malformed = '{"id": "join", "tool": '

    still_masked, _ = repair_with(ScriptedFillBackend(unfinished), task, plan, corruption)
    just_broken, _ = repair_with(ScriptedFillBackend(malformed), task, plan, corruption)

    assert still_masked.last_diagnostics.parsed is False
    assert still_masked.last_diagnostics.unfinished_denoising is True
    assert just_broken.last_diagnostics.parsed is False
    assert just_broken.last_diagnostics.unfinished_denoising is False


# --- where it broke ------------------------------------------------------------------------------


def test_the_failure_carries_the_place_and_the_text_around_it():
    """ "line 15 column 123" is not something anyone can act on; the characters there are."""
    task, plan, corruption = corrupted()

    backend = ScriptedFillBackend('{"id": "join", "tool": }')
    repairer, _ = repair_with(backend, task, plan, corruption)

    failure = repairer.last_diagnostics.parse_failure
    assert failure is not None
    assert failure.position is not None
    assert failure.line is not None and failure.column is not None
    assert ">>>" in failure.excerpt
    before, after = failure.excerpt.split(">>>")
    assert repairer.last_diagnostics.raw_text[failure.position :].startswith(after[:20])
    assert before.endswith('"tool": ')


def test_a_schema_violation_is_reported_without_a_position():
    """Not every parse failure has an offset; a missing field has no place in the text."""
    task, plan, corruption = corrupted()
    valid_json_wrong_shape = '{"id": "join"}'

    repairer, _ = repair_with(ScriptedFillBackend(valid_json_wrong_shape), task, plan, corruption)

    failure = repairer.last_diagnostics.parse_failure
    assert failure is not None
    assert "did not satisfy the plan schema" in failure.message
    assert failure.position is None
    assert failure.excerpt == ""


@pytest.mark.parametrize(
    ("text", "position", "expected"),
    [
        ("abcdefghij", 5, "abcde>>>fghij"),
        ("abc", 0, ">>>abc"),
        ("abc", 3, "abc>>>"),
        ("abc", 99, "abc>>>"),
        ("", 0, ""),
    ],
)
def test_the_excerpt_marks_the_spot(text, position, expected):
    assert excerpt_around(text, position, radius=10) == expected


def test_the_excerpt_is_bounded():
    long_text = "x" * 1000

    excerpt = excerpt_around(long_text, 500, radius=20)

    assert excerpt == "x" * 20 + ">>>" + "x" * 20


# --- reading a batch -----------------------------------------------------------------------------


def json_error() -> PlanParseError:
    """A parse error chained to a real JSON failure, as ``parse_plan`` raises them."""
    try:
        json.loads("{oops")
    except json.JSONDecodeError as cause:
        error = PlanParseError("the answer was not valid JSON")
        error.__cause__ = cause
        return error
    raise AssertionError("that should not have parsed")


def failed_result(mask_tokens: int):
    return {
        "diagnostics": diagnose(
            raw_text="M" * mask_tokens + "{oops",
            fillings={},
            mask_token="M",
            error=json_error(),
        ).model_dump()
    }


def test_a_batch_reports_how_many_failures_were_still_masked():
    """The first fork when reading a run: budget, or format."""
    results = [failed_result(3), failed_result(1), failed_result(0)]

    summary = summarise(results)

    assert summary["parse_failures"] == 3
    assert summary["parse_failures_with_mask_tokens_left"] == 2
    assert summary["share_of_failures_still_masked"] == pytest.approx(0.667, abs=0.001)


def test_a_batch_with_nothing_to_report_says_so():
    assert summarise([])["with_diagnostics"] == 0
    assert summarise([{"diagnostics": None}])["with_diagnostics"] == 0
    assert summarise([{}])["share_of_failures_still_masked"] is None


def test_a_clean_batch_reports_no_failures():
    task, plan, corruption = corrupted()
    repairer, _ = repair_with(OracleBackend(plan), task, plan, corruption)

    summary = summarise([{"diagnostics": repairer.last_diagnostics.model_dump()}])

    assert summary["parse_failures"] == 0
    assert summary["share_of_failures_still_masked"] is None


# --- the recording changes nothing ---------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_recording_does_not_change_what_a_repair_returns(domain):
    """Observation only: the repaired plan and the score are what they were before."""
    task, plan = load_reference(domain)
    corruption = inject_wrong_tool(plan, step_id=plan.steps[-1].id)

    repairer = LLaDARepairer(OracleBackend(plan))
    repaired, score = repair_and_score(
        repairer,
        reference_plan=plan,
        broken_plan=corruption.broken_plan,
        task=task,
        damaged_step_ids=corruption.injected[0].damaged_step_ids,
    )

    assert repaired == plan
    assert score.solved
    assert (score.collateral_modified, score.collateral_renamed, score.collateral_removed) == (
        0,
        0,
        0,
    )
    assert isinstance(repairer.last_diagnostics, RepairDiagnostics)
