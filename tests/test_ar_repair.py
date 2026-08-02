"""AR repair logic, exercised without a key, a network or a bill.

Every answer the model could give is supplied by a scripted client, so what is under test is the
part this project owns: the prompt each mode builds, the parsing of the answer, the treatment of
a failed answer, and the scoring of whatever comes back.
"""

import json

import pytest

from plan_repair.corruption import (
    UNKNOWN_MODE,
    inject_broken_dependency,
    inject_missing_stop_condition,
)
from plan_repair.data import DATA_PIPELINE_A, DATA_PIPELINE_B, load_reference
from plan_repair.repair import (
    API_FAILURE,
    PARSE_FAILURE,
    ARFullRepairer,
    ARLocalRepairer,
    LLMError,
    Repairer,
    ScriptedLLMClient,
    plan_to_json,
    repair_and_score,
)
from plan_repair.validation import validate_plan

DOMAINS = [DATA_PIPELINE_A, DATA_PIPELINE_B]


def corrupted(domain=DATA_PIPELINE_B):
    """A plan with one unknown dependency, plus the ground truth of what it damaged."""
    task, plan = load_reference(domain)
    fan_in = next(step for step in plan.steps if len(step.input_from) > 1)
    return task, plan, inject_broken_dependency(plan, step_id=fan_in.id, mode=UNKNOWN_MODE)


def answers(*responses):
    return ScriptedLLMClient(list(responses))


def score_with(repairer, task, reference, corruption):
    return repair_and_score(
        repairer,
        reference_plan=reference,
        broken_plan=corruption.broken_plan,
        task=task,
        damaged_step_ids=corruption.injected[0].damaged_step_ids,
    )


# --- the port and the client seam ---------------------------------------------------------------


def test_both_modes_satisfy_the_repairer_port():
    client = answers()

    assert isinstance(ARFullRepairer(client), Repairer)
    assert isinstance(ARLocalRepairer(client), Repairer)


def test_no_api_key_is_needed_to_run_the_repair_path(monkeypatch):
    """The scripted client is the whole backend here; nothing reads the environment."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    task, plan, corruption = corrupted()

    _, score = score_with(ARFullRepairer(answers(plan_to_json(plan))), task, plan, corruption)

    assert score.solved


# --- prompts ------------------------------------------------------------------------------------


def test_the_full_mode_asks_for_a_rewrite_and_names_no_location():
    task, plan, corruption = corrupted()
    client = answers(plan_to_json(plan))

    ARFullRepairer(client).repair(
        corruption.broken_plan, validate_plan(corruption.broken_plan, task), task
    )
    prompt = client.calls[0]["user"]

    assert "from scratch" in prompt
    assert "The validator flagged" not in prompt


def test_the_local_mode_is_given_the_flagged_locations():
    task, plan, corruption = corrupted()
    broken = corruption.broken_plan
    validation = validate_plan(broken, task)
    client = answers(plan_to_json(plan))

    ARLocalRepairer(client).repair(broken, validation, task)
    prompt = client.calls[0]["user"]

    assert "The validator flagged these locations" in prompt
    for step_id in validation.detected_step_ids():
        assert step_id in prompt
    for path in validation.detected_paths():
        assert path in prompt
    assert "Leave every other step exactly as it is" in prompt


def test_the_local_prompt_withholds_the_error_types():
    """Location only for now, so the same decision can be made once for AR and diffusion."""
    task, _, corruption = corrupted()
    broken = corruption.broken_plan
    validation = validate_plan(broken, task)
    client = answers(plan_to_json(broken))

    ARLocalRepairer(client).repair(broken, validation, task)
    prompt = client.calls[0]["user"]

    assert {error.type for error in validation.errors}
    for error in validation.errors:
        assert error.type not in prompt
        assert error.message not in prompt


def test_the_prompt_carries_the_task_contract():
    task, plan, corruption = corrupted()
    client = answers(plan_to_json(plan))

    ARFullRepairer(client).repair(
        corruption.broken_plan, validate_plan(corruption.broken_plan, task), task
    )
    prompt = client.calls[0]["user"]

    for tool in task.available_tools:
        assert tool.name in prompt
    for requirement in task.required_evidence + task.required_operations:
        assert requirement in prompt


# --- parsing ------------------------------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_a_well_formed_answer_is_scored_as_a_repair(domain):
    task, plan, corruption = corrupted(domain)

    repaired, score = score_with(
        ARFullRepairer(answers(plan_to_json(plan))), task, plan, corruption
    )

    assert repaired == plan
    assert score.solved
    assert score.collateral_total == 0


def test_a_fenced_answer_is_accepted():
    """A code fence is a formatting habit, not a defect in the answer."""
    task, plan, corruption = corrupted()
    fenced = f"```json\n{plan_to_json(plan)}\n```"

    _, score = score_with(ARFullRepairer(answers(fenced)), task, plan, corruption)

    assert score.solved


@pytest.mark.parametrize(
    ("answer", "why"),
    [
        ("{not json at all", "broken JSON"),
        ("Sure! Here is the fixed plan.", "prose"),
        ("[]", "a list instead of an object"),
        ("", "nothing"),
        (json.dumps({"goal": "g"}), "a plan missing its steps"),
        (json.dumps({"goal": "g", "steps": [{"id": "a"}]}), "a step missing its tool"),
    ],
)
def test_an_unusable_answer_is_a_failed_repair_not_a_crash(answer, why):
    task, plan, corruption = corrupted()
    repairer = ARFullRepairer(answers(answer))

    repaired, score = score_with(repairer, task, plan, corruption)

    assert repaired == corruption.broken_plan, why  # handed back untouched
    assert not score.solved
    assert score.collateral_total == 0
    assert [failure.kind for failure in repairer.failures] == [PARSE_FAILURE]


def test_a_schema_violating_plan_is_not_massaged_into_shape():
    """An extra field means the model did not honour the contract; that is a failure."""
    task, plan, corruption = corrupted()
    payload = plan.model_dump()
    payload["steps"][0]["description"] = "load the quarterly sales file"
    repairer = ARFullRepairer(answers(json.dumps(payload)))

    repaired, _ = score_with(repairer, task, plan, corruption)

    assert repaired == corruption.broken_plan
    assert repairer.failures[0].kind == PARSE_FAILURE


def test_an_api_error_is_a_failed_repair_not_a_crash():
    task, plan, corruption = corrupted()
    repairer = ARLocalRepairer(answers(LLMError("connection reset")))

    repaired, score = score_with(repairer, task, plan, corruption)

    assert repaired == corruption.broken_plan
    assert not score.solved
    assert [failure.kind for failure in repairer.failures] == [API_FAILURE]
    assert "connection reset" in repairer.failures[0].detail


def test_failures_accumulate_across_calls():
    task, _, corruption = corrupted()
    repairer = ARFullRepairer(answers("nope", LLMError("timeout")))
    broken = corruption.broken_plan
    validation = validate_plan(broken, task)

    repairer.repair(broken, validation, task)
    repairer.repair(broken, validation, task)

    assert [failure.kind for failure in repairer.failures] == [PARSE_FAILURE, API_FAILURE]
    assert {failure.repairer for failure in repairer.failures} == {"ar_full"}


# --- scoring: what the modes are supposed to differ on -------------------------------------------


def test_collateral_catches_a_rewrite_that_disturbs_healthy_steps():
    """The reading full regeneration is expected to produce, forced here with a scripted answer."""
    task, plan, corruption = corrupted()
    rewritten = plan.model_copy(deep=True)
    for step in rewritten.steps[:3]:
        step.arguments = {**step.arguments, "rewritten": True}

    _, score = score_with(ARFullRepairer(answers(plan_to_json(rewritten))), task, plan, corruption)

    assert score.collateral_modified == 3
    assert score.solved  # a plan can be valid and still have cost healthy steps


def test_a_faithful_local_edit_costs_nothing():
    task, plan, corruption = corrupted()
    damaged = corruption.injected[0].damaged_step_ids[0]
    edited = corruption.broken_plan.model_copy(deep=True)
    original = next(step for step in plan.steps if step.id == damaged)
    next(step for step in edited.steps if step.id == damaged).input_from = list(original.input_from)

    _, score = score_with(ARLocalRepairer(answers(plan_to_json(edited))), task, plan, corruption)

    assert score.collateral_total == 0
    assert score.damaged_restored == 1
    assert score.solved


def test_the_two_modes_are_scored_on_the_same_scale():
    """Same corruption, same scorer; only the answers differ."""
    task, plan, corruption = corrupted()
    sloppy = plan.model_copy(deep=True)
    for step in sloppy.steps[:4]:
        step.arguments = {**step.arguments, "rewritten": True}

    _, full = score_with(ARFullRepairer(answers(plan_to_json(sloppy))), task, plan, corruption)
    _, local = score_with(ARLocalRepairer(answers(plan_to_json(plan))), task, plan, corruption)

    assert full.solved and local.solved
    assert full.collateral_total > local.collateral_total == 0


@pytest.mark.parametrize("domain", DOMAINS)
def test_missing_stop_condition_is_repairable_by_either_mode(domain):
    task, plan = load_reference(domain)
    corruption = inject_missing_stop_condition(plan)

    for repairer in (
        ARFullRepairer(answers(plan_to_json(plan))),
        ARLocalRepairer(answers(plan_to_json(plan))),
    ):
        _, score = repair_and_score(
            repairer,
            reference_plan=plan,
            broken_plan=corruption.broken_plan,
            task=task,
            damaged_step_ids=corruption.injected[0].damaged_step_ids,
        )
        assert score.solved
        assert score.collateral_total == 0


def test_one_call_per_repair():
    """No retry loop: a repairer that calls in a loop is how a small experiment gets expensive."""
    task, plan, corruption = corrupted()
    client = answers(plan_to_json(plan))

    ARFullRepairer(client).repair(
        corruption.broken_plan, validate_plan(corruption.broken_plan, task), task
    )

    assert client.call_count == 1
