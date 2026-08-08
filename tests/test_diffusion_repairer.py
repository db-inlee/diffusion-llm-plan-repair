"""The diffusion repairer skeleton, end to end on a mock backend.

No model and no network: the backend is an interface, and these fill it with known answers so the
plumbing around it — mask, align, fill, reassemble, parse, score — can be judged on its own.
"""

import pytest

from plan_repair.corruption import (
    UNKNOWN_MODE,
    inject_broken_dependency,
    inject_duplicate_step,
    inject_missing_stop_condition,
    inject_step_deletion,
)
from plan_repair.data import DATA_PIPELINE_A, DATA_PIPELINE_B, load_reference
from plan_repair.repair import (
    ALIGNMENT_FAILURE,
    BACKEND_FAILURE,
    PARSE_FAILURE,
    DreamRepairer,
    EchoBackend,
    FailingBackend,
    LLaDARepairer,
    OracleBackend,
    Repairer,
    repair_and_score,
)
from plan_repair.repair.diffusion import (
    DREAM_MASK_TOKEN_ID,
    LLADA_MASK_TOKEN_ID,
    DiffusionRepairer,
)
from tests.test_tokenization import CharTokenizer, ChunkTokenizer

DOMAINS = [DATA_PIPELINE_A, DATA_PIPELINE_B]
MODELS = [LLaDARepairer, DreamRepairer]


def corrupted(domain=DATA_PIPELINE_B):
    task, plan = load_reference(domain)
    fan_in = next(step for step in plan.steps if len(step.input_from) > 1)
    return task, plan, inject_broken_dependency(plan, step_id=fan_in.id, mode=UNKNOWN_MODE)


def scored(repairer, task, reference, corruption):
    return repair_and_score(
        repairer,
        reference_plan=reference,
        broken_plan=corruption.broken_plan,
        task=task,
        damaged_step_ids=corruption.injected[0].damaged_step_ids,
    )


def zero_collateral(score):
    return (score.collateral_modified, score.collateral_renamed, score.collateral_removed) == (
        0,
        0,
        0,
    )


@pytest.mark.parametrize("model", MODELS)
def test_both_models_satisfy_the_repairer_port(model):
    _, plan = load_reference()

    assert isinstance(model(OracleBackend(plan)), Repairer)


def test_the_two_models_differ_only_in_their_parts():
    """Same flow, different tokenizer and mask token — that is what makes them comparable."""
    assert LLaDARepairer.repair is DiffusionRepairer.repair
    assert DreamRepairer.repair is DiffusionRepairer.repair
    assert LLaDARepairer.mask_token_id == LLADA_MASK_TOKEN_ID != DREAM_MASK_TOKEN_ID
    assert DreamRepairer.mask_token_id == DREAM_MASK_TOKEN_ID
    assert LLaDARepairer.model_id != DreamRepairer.model_id


# --- the ceiling ---------------------------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("model", MODELS)
def test_a_correct_mask_filled_with_the_answer_disturbs_nothing(model, domain):
    task, plan, corruption = corrupted(domain)

    repaired, score = scored(model(OracleBackend(plan), CharTokenizer()), task, plan, corruption)

    assert score.solved
    assert zero_collateral(score)
    assert score.spurious_added == 0
    assert repaired == plan


@pytest.mark.parametrize("model", MODELS)
def test_the_pipeline_runs_without_a_tokenizer(model):
    """Alignment is optional plumbing for a backend that works on text."""
    task, plan, corruption = corrupted()

    repairer = model(OracleBackend(plan))
    _, score = scored(repairer, task, plan, corruption)

    assert repairer.last_alignment is None
    assert score.solved
    assert zero_collateral(score)


@pytest.mark.parametrize("model", MODELS)
def test_the_alignment_is_recorded_for_inspection(model):
    task, plan, corruption = corrupted()

    repairer = model(OracleBackend(plan), CharTokenizer())
    scored(repairer, task, plan, corruption)

    assert repairer.last_mask is not None
    assert repairer.last_alignment is not None
    assert repairer.last_alignment.masked_token_count > 0
    # Keyed by span: a field mask narrows 'join' to 'join.input_from'.
    assert set(repairer.last_alignment.token_ranges) == {
        span.key for span in repairer.last_mask.spans
    }


@pytest.mark.parametrize("width", [1, 4, 16])
def test_a_coarser_tokenizer_does_not_cost_healthy_steps(width):
    """Token width changes how much is masked, never whose step it is."""
    task, plan, corruption = corrupted()

    _, score = scored(
        LLaDARepairer(OracleBackend(plan), ChunkTokenizer(width)), task, plan, corruption
    )

    assert zero_collateral(score)
    assert score.solved


def test_an_echoing_backend_changes_nothing_at_all():
    """Separates a wrong mask from a wrong filling: nothing here can move but the plumbing."""
    task, plan, corruption = corrupted()

    repaired, score = scored(LLaDARepairer(EchoBackend(), CharTokenizer()), task, plan, corruption)

    assert repaired == corruption.broken_plan
    assert zero_collateral(score)
    assert not score.solved  # it echoed the damage back


def test_a_duplicated_step_is_filled_with_nothing():
    task, plan = load_reference()
    corruption = inject_duplicate_step(plan, step_id="agg")

    repaired, score = scored(
        LLaDARepairer(OracleBackend(plan), CharTokenizer()), task, plan, corruption
    )

    assert score.solved
    assert [step.id for step in repaired.steps] == [step.id for step in plan.steps]
    assert zero_collateral(score)


# --- range: what a step-level mask still cannot reach ---------------------------------------------


@pytest.mark.parametrize(
    ("corrupt", "why"),
    [
        (lambda p: inject_step_deletion(p, step_id="join"), "a deleted step has no span"),
        (lambda p: inject_missing_stop_condition(p), "a plan-level field belongs to no step"),
    ],
)
def test_out_of_range_corruptions_fail_without_collateral(corrupt, why):
    task, plan = load_reference()
    corruption = corrupt(plan)

    _, score = scored(LLaDARepairer(OracleBackend(plan), CharTokenizer()), task, plan, corruption)

    assert not score.solved, why
    assert zero_collateral(score)


# --- failure paths --------------------------------------------------------------------------------


def test_a_backend_error_is_a_failed_repair_not_a_crash():
    task, plan, corruption = corrupted()
    repairer = LLaDARepairer(FailingBackend("weights not loaded"), CharTokenizer())

    repaired, score = scored(repairer, task, plan, corruption)

    assert repaired == corruption.broken_plan
    assert not score.solved
    assert [failure.kind for failure in repairer.failures] == [BACKEND_FAILURE]
    assert "weights not loaded" in repairer.failures[0].detail


def test_a_tokenizer_error_is_a_failed_repair_not_a_crash():
    class BrokenTokenizer:
        name = "broken"

        def encode_with_offsets(self, text):
            raise RuntimeError("vocabulary missing")

    task, plan, corruption = corrupted()
    repairer = DreamRepairer(OracleBackend(plan), BrokenTokenizer())

    repaired, score = scored(repairer, task, plan, corruption)

    assert repaired == corruption.broken_plan
    assert [failure.kind for failure in repairer.failures] == [ALIGNMENT_FAILURE]
    assert not score.solved


def test_a_filling_that_is_not_a_step_is_a_failed_repair():
    class NonsenseBackend:
        name = "nonsense"

        def fill(self, request):
            return dict.fromkeys((span.key for span in request.mask.spans), "not a step")

    task, plan, corruption = corrupted()
    repairer = LLaDARepairer(NonsenseBackend(), CharTokenizer())

    repaired, score = scored(repairer, task, plan, corruption)

    assert repaired == corruption.broken_plan
    assert [failure.kind for failure in repairer.failures] == [PARSE_FAILURE]
    assert not score.solved


def test_a_backend_reaching_outside_the_mask_is_refused():
    """The Ticket B-3a guarantee survives having a model behind it."""

    class GreedyBackend:
        name = "greedy"

        def fill(self, request):
            filling = dict.fromkeys((span.key for span in request.mask.spans), "{}")
            filling[request.mask.preserved_step_ids[0]] = "{}"  # a healthy step
            return filling

    task, plan, corruption = corrupted()
    repairer = LLaDARepairer(GreedyBackend(), CharTokenizer())

    repaired, score = scored(repairer, task, plan, corruption)

    assert repaired == corruption.broken_plan
    assert repairer.failures[0].kind == BACKEND_FAILURE
    assert "not masked" in repairer.failures[0].detail
    assert zero_collateral(score)
