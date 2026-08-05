"""The PyTorch backend, on a machine that has no GPU and no torch.

Only one method of this backend touches tensors. Everything that decides *what happens* — which
positions may be revealed, in what order, and whether anything outside the mask moved — is plain
Python, and is checked here by standing a fake model in for the forward pass. So the guarantee
that matters, that a healthy step is unreachable from inside the denoising loop, is established
on the laptop rather than deferred to the GPU.

What is not checked anywhere in this repository is LLaDA or Dream actually filling a mask. That
needs the weights and a GPU; it belongs to whoever runs the experiment script.
"""

import sys

import pytest

from plan_repair.corruption import UNKNOWN_MODE, inject_broken_dependency
from plan_repair.data import load_reference
from plan_repair.repair import (
    DLLMError,
    FillRequest,
    TorchDLLMBackend,
    mask_spec,
    plan_to_sequence,
    torch_available,
)
from plan_repair.repair.diffusion import LLADA_MASK_TOKEN_ID, LLADA_MODEL
from plan_repair.repair.dllm_backend_torch import (
    DEFAULT_STEPS,
    _assert_frozen_positions_untouched,
    select_reveals,
)
from plan_repair.validation import validate_plan

MASK = LLADA_MASK_TOKEN_ID


class FakeModelBackend(TorchDLLMBackend):
    """A backend whose forward pass is a table instead of a model."""

    def __init__(self, answer: int = 7, confidence: list[float] | None = None, **overrides):
        super().__init__(LLADA_MODEL, MASK, **overrides)
        self.answer = answer
        self.confidence = confidence
        self.passes = 0
        self.seen: list[list[int]] = []

    def predict_pass(self, sequence):
        self.passes += 1
        self.seen.append(list(sequence))
        confidence = self.confidence or [0.9] * len(sequence)
        return [self.answer] * len(sequence), list(confidence)


def test_constructing_a_backend_loads_nothing():
    """A backend can be built, inspected and type-checked on a laptop."""
    made = TorchDLLMBackend(LLADA_MODEL, MASK, steps=8, temperature=0.5)

    assert made.model_id == LLADA_MODEL
    assert made.settings()["decoding"] == "sampled"
    assert TorchDLLMBackend(LLADA_MODEL, MASK).settings()["decoding"] == "greedy"
    assert TorchDLLMBackend(LLADA_MODEL, MASK).steps == DEFAULT_STEPS


def test_torch_availability_is_reported_not_assumed():
    assert torch_available() is False or torch_available() is True


def test_running_without_torch_says_what_to_install():
    with pytest.raises(DLLMError, match=r"torch is not installed.*\[gpu\]"):
        TorchDLLMBackend(LLADA_MODEL, MASK).predict_pass([1, 2, 3])


def test_the_module_imports_without_torch(monkeypatch):
    """Nothing about this module may require the GPU extra at import time."""
    monkeypatch.setitem(sys.modules, "torch", None)
    from plan_repair.repair.dllm_backend_torch import _require_torch

    with pytest.raises(DLLMError, match="torch is not installed"):
        _require_torch()


# --- selective filling: the loop ----------------------------------------------------------------


def test_denoising_only_writes_to_masked_positions():
    """The guarantee the whole approach rests on, at the level where a model could break it."""
    backend = FakeModelBackend(answer=7, steps=4)
    prompt = [1, 2, MASK, 4, MASK, 6]

    filled = backend.denoise(prompt)

    assert filled == [1, 2, 7, 4, 7, 6]
    assert len(filled) == len(prompt)


def test_a_model_answering_everything_still_changes_only_the_mask():
    """The fake predicts a new token at every position; only two of them are allowed to land."""
    backend = FakeModelBackend(answer=99, steps=1)
    prompt = [10, 11, MASK, 13, MASK, 15, 16]

    filled = backend.denoise(prompt)

    assert filled == [10, 11, 99, 13, 99, 15, 16]
    assert [i for i, (a, b) in enumerate(zip(prompt, filled, strict=True)) if a != b] == [2, 4]


def test_a_sequence_with_no_mask_is_returned_without_a_forward_pass():
    backend = FakeModelBackend()
    prompt = [1, 2, 3, 4]

    assert backend.denoise(prompt) == prompt
    assert backend.passes == 0


def test_every_masked_position_is_eventually_revealed():
    backend = FakeModelBackend(answer=9, steps=3)

    filled = backend.denoise([MASK] * 7)

    assert filled == [9] * 7
    assert MASK not in filled


@pytest.mark.parametrize("steps", [1, 2, 8, 64])
def test_the_step_budget_changes_the_number_of_passes_not_the_answer(steps):
    backend = FakeModelBackend(answer=5, steps=steps)
    prompt = [1, MASK, MASK, MASK, MASK, 2]

    assert backend.denoise(prompt) == [1, 5, 5, 5, 5, 2]
    assert backend.passes <= max(steps, 1) + 1


def test_each_pass_sees_what_the_previous_one_committed():
    """Later positions are predicted with the earlier ones already filled in — the point of
    iterating rather than deciding everything at once."""
    backend = FakeModelBackend(answer=8, steps=4, confidence=[0.1, 0.9, 0.8, 0.7, 0.1])
    prompt = [0, MASK, MASK, MASK, 0]

    backend.denoise(prompt)

    assert backend.passes == 3
    assert backend.seen[0] == [0, MASK, MASK, MASK, 0]
    assert backend.seen[1] == [0, 8, MASK, MASK, 0]
    assert backend.seen[2] == [0, 8, 8, MASK, 0]


# --- which positions get revealed ----------------------------------------------------------------


def test_reveals_are_the_most_confident_masked_positions():
    confidence = [0.9, 0.1, 0.8, 0.5, 0.99]

    assert select_reveals(confidence, remaining=[1, 2, 3], count=2) == [2, 3]


def test_a_confident_unmasked_position_is_never_chosen():
    """Confidence cannot buy a position into the mask."""
    confidence = [1.0, 0.1, 1.0]

    assert select_reveals(confidence, remaining=[1], count=3) == [1]


def test_ties_break_on_position_so_a_run_repeats():
    confidence = [0.5] * 6

    assert select_reveals(confidence, remaining=[4, 1, 3], count=2) == [1, 3]
    assert select_reveals(confidence, remaining=[4, 1, 3], count=2) == [1, 3]


def test_asking_for_more_reveals_than_remain_is_harmless():
    assert select_reveals([0.5, 0.5], remaining=[0, 1], count=99) == [0, 1]
    assert select_reveals([0.5, 0.5], remaining=[], count=3) == []


# --- the freeze check -----------------------------------------------------------------------------


def test_a_fill_that_moved_something_outside_the_mask_is_caught():
    """The check exists because a guarantee nobody verifies is a hope."""
    prompt = [1, 2, MASK, 4]
    rewritten = [7, 7, 7, 7]

    with pytest.raises(DLLMError, match="rewrote 3 position"):
        _assert_frozen_positions_untouched(prompt, rewritten, MASK, "test")


def test_the_freeze_check_passes_a_faithful_fill():
    _assert_frozen_positions_untouched([1, MASK, 3], [1, 42, 3], MASK, "test")


def test_the_freeze_check_names_the_first_offending_position():
    with pytest.raises(DLLMError, match="first at 1"):
        _assert_frozen_positions_untouched([1, 2, MASK], [1, 99, 42], MASK, "test")


# --- the backend interface ------------------------------------------------------------------------


def masked_request():
    task, plan = load_reference()
    broken = inject_broken_dependency(plan, step_id="join", mode=UNKNOWN_MODE).broken_plan
    sequence = plan_to_sequence(broken)
    spec = mask_spec(sequence, validate_plan(broken, task).detected_step_ids())
    return FillRequest(sequence=sequence, mask=spec, alignment=None, task=task)


def test_filling_without_an_alignment_is_refused():
    """A diffusion backend works in tokens; without a tokenizer there is nothing to fill."""
    with pytest.raises(DLLMError, match="no token alignment"):
        TorchDLLMBackend(LLADA_MODEL, MASK).fill(masked_request())


def test_an_empty_mask_needs_no_model():
    request = masked_request()
    empty = request.model_copy(
        update={"mask": request.mask.model_copy(update={"masked_step_ids": [], "spans": []})}
    )

    assert TorchDLLMBackend(LLADA_MODEL, MASK).fill(empty) == {}


def test_the_settings_record_what_a_run_used():
    """Every result file carries these, so a measurement can be traced to its configuration."""
    settings = TorchDLLMBackend(LLADA_MODEL, MASK, steps=32, temperature=0.2).settings()

    assert settings == {
        "backend": "torch",
        "model_id": LLADA_MODEL,
        "mask_token_id": MASK,
        "steps": 32,
        "temperature": 0.2,
        "quantize_4bit": True,
        "decoding": "sampled",
    }
