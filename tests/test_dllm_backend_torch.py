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

from plan_repair.corruption import (
    UNKNOWN_MODE,
    inject_broken_dependency,
    inject_missing_stop_condition,
    inject_wrong_ordering,
    inject_wrong_tool,
)
from plan_repair.data import DATA_PIPELINE_A, DATA_PIPELINE_B, load_reference
from plan_repair.repair import (
    DLLMError,
    FillRequest,
    TorchDLLMBackend,
    align_mask,
    decode_spans,
    mask_spec,
    mask_spec_from_paths,
    masked_token_ids,
    plan_to_sequence,
    torch_available,
    valid_tool_hint,
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


# --- the valid-tool hint ------------------------------------------------------------------------
#
# The hint is prepended to the model's input, which moves every plan token along by the length of
# the prefix. The alignment that says where a field sits was computed on the plan alone and is not
# moved, so the two disagree by exactly that offset — and a wrong offset does not raise anything.
# It reads the neighbouring characters and returns them as if they were the repair. That is what
# these tests are for.


class CharVocabulary:
    """One token per character, in both directions.

    A prefix is then a character count, so the arithmetic under test can be checked by reading
    rather than by trusting a vocabulary. The real ones are exercised in the integration tests.
    """

    name = "char"

    def encode_with_offsets(self, text):
        return [ord(character) for character in text], [(i, i + 1) for i in range(len(text))]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, token_ids):
        return "".join(chr(token) for token in token_ids)


class _MaskInjectingVocabulary(CharVocabulary):
    """A vocabulary that puts the mask token inside the hint, which must be refused."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [MASK, *(ord(character) for character in text)]}


class ReplayBackend(TorchDLLMBackend):
    """A model that predicts exactly the text that was masked out.

    Not a stand-in for competence — the point is that the answer is known, so a span that comes
    back as anything other than the original was read from the wrong place.
    """

    def __init__(self, answer: list[int] | None = None, tokenizer=None, **overrides):
        super().__init__(LLADA_MODEL, MASK, tokenizer=tokenizer or CharVocabulary(), **overrides)
        self._answer = answer
        self.seen_prompt: list[int] = []

    def predict_pass(self, sequence):
        if not self.seen_prompt:
            self.seen_prompt = list(sequence)
        return list(self._answer or [0] * len(sequence)), [0.9] * len(sequence)


def tool_request(domain=DATA_PIPELINE_B):
    """A wrong-tool repair, masked down to the field, aligned on the character vocabulary."""
    task, plan = load_reference(domain)
    target = next(step.id for step in plan.steps if len(step.input_from) > 1)
    broken = inject_wrong_tool(plan, step_id=target).broken_plan
    sequence = plan_to_sequence(broken)
    spec = mask_spec_from_paths(sequence, broken, validate_plan(broken, task).detected_paths())
    alignment = align_mask(sequence, spec, CharVocabulary())
    request = FillRequest(sequence=sequence, mask=spec, alignment=alignment, task=task)
    return request, f"{target}.tool"


def without_tools(request):
    """The same repair for a task that offers no tools — the hint has nothing to say."""
    return request.model_copy(
        update={"task": request.task.model_copy(update={"available_tools": []})}
    )


def hint_for(request) -> str:
    """The hint this request produces, asserted to exist: every caller builds a tool repair."""
    hint = valid_tool_hint(request.task, request.mask)
    assert hint is not None
    return hint


def test_the_hint_names_every_tool_the_task_allows():
    task, plan = load_reference(DATA_PIPELINE_B)
    request, _ = tool_request(DATA_PIPELINE_B)

    hint = hint_for(request)

    assert hint == (
        "valid tools: aggregate, clean_missing, clean_outlier, correlate, enrich, interpret, "
        "join, load_api, load_csv, load_db, normalize, pivot, profile, report, "
        "statistical_test, validate_schema, visualize\n"
    )
    assert set(hint.removeprefix("valid tools: ").strip().split(", ")) == task.tool_names()
    assert plan.steps[11].tool in hint


def test_the_tool_list_is_sorted_so_a_run_repeats():
    """``tool_names`` returns a set; a prompt that varies between runs is not a measurement."""
    request, _ = tool_request(DATA_PIPELINE_A)

    names = hint_for(request).removeprefix("valid tools: ").strip().split(", ")

    assert names == sorted(names)


@pytest.mark.parametrize(
    ("corrupt", "why"),
    [
        (
            lambda plan: inject_broken_dependency(plan, step_id="join", mode=UNKNOWN_MODE),
            "a dependency repair is out of this ticket's scope",
        ),
        (lambda plan: inject_wrong_ordering(plan, step_id="join"), "ordering names no tool"),
        (lambda plan: inject_missing_stop_condition(plan), "nothing is masked at all"),
    ],
)
def test_only_a_repair_that_rewrites_a_tool_gets_the_hint(corrupt, why):
    """Every other error type keeps the input it had, so its results stay comparable."""
    task, plan = load_reference(DATA_PIPELINE_B)
    broken = corrupt(plan).broken_plan
    sequence = plan_to_sequence(broken)
    spec = mask_spec_from_paths(sequence, broken, validate_plan(broken, task).detected_paths())

    assert valid_tool_hint(task, spec) is None, why


def test_a_whole_step_mask_gets_no_hint_either():
    """It regenerates a tool too, and is deliberately left alone: C-2 is scoped to wrong_tool."""
    task, plan = load_reference(DATA_PIPELINE_B)
    sequence = plan_to_sequence(plan)

    assert valid_tool_hint(task, mask_spec(sequence, ["join"])) is None


def test_a_task_offering_no_tools_gets_no_hint():
    request, _ = tool_request()

    assert valid_tool_hint(without_tools(request).task, request.mask) is None


def test_the_hint_goes_in_front_of_the_plan_and_the_plan_is_unchanged():
    request, _ = tool_request()
    hint = hint_for(request)
    backend = ReplayBackend(steps=4)

    backend.fill(request)
    prompt = backend.seen_prompt

    assert prompt[: len(hint)] == [ord(character) for character in hint]
    assert prompt[len(hint) :] == masked_token_ids(request.alignment, MASK)


def test_the_hint_is_fixed_context_and_never_a_position_to_fill():
    request, _ = tool_request()
    hint = hint_for(request)
    backend = ReplayBackend(steps=4)

    backend.fill(request)

    assert MASK not in backend.seen_prompt[: len(hint)]
    assert all(
        position >= len(hint) for position, token in enumerate(backend.seen_prompt) if token == MASK
    )


def test_a_field_reads_back_the_same_with_the_hint_as_without_it():
    """The correction, stated as the thing it has to preserve.

    Same plan, same mask, same alignment; the only difference is the prefix. A model that
    reproduces what was masked has to yield the original field either way — and it does not,
    if the offset is dropped.
    """
    request, key = tool_request()
    hint = hint_for(request)
    text = request.sequence.text

    with_hint = ReplayBackend([ord(c) for c in hint + text], steps=4).fill(request)
    without_hint = ReplayBackend([ord(c) for c in text], steps=4).fill(without_tools(request))

    assert with_hint == without_hint
    assert with_hint[key] == '"join_x"'


def test_forgetting_the_offset_would_read_the_wrong_characters():
    """What the correction is worth: the same read, done without it, is silently wrong.

    Nothing raises. The field comes back as whatever sits ``len(hint)`` characters earlier in the
    plan — a plausible-looking run of JSON from a neighbouring step. That is the failure this
    arithmetic exists to prevent, and it is why it is checked rather than reasoned about.
    """
    request, key = tool_request()
    hint = hint_for(request)
    span = next(span for span in request.mask.spans if span.key == key)
    filled = [ord(character) for character in hint + request.sequence.text]

    uncorrected = decode_spans(request.alignment, filled, CharVocabulary())

    assert uncorrected[key] != '"join_x"'
    assert uncorrected[key] == (hint + request.sequence.text)[span.start : span.end]


def test_a_hint_carrying_the_mask_token_is_refused():
    """Denoising fills every mask token it finds, including one inside its own instructions."""
    request, _ = tool_request()
    backend = ReplayBackend(tokenizer=_MaskInjectingVocabulary(), steps=4)

    with pytest.raises(DLLMError, match="tokenized to include the mask token"):
        backend.fill(request)


def test_a_model_that_rewrote_the_hint_is_caught():
    """The freeze check covers the prefix: it is outside the mask like anything else unmasked."""

    class Tampering(ReplayBackend):
        def denoise(self, token_ids):
            rewritten = list(token_ids)
            rewritten[0] = 999
            return rewritten

    request, _ = tool_request()

    with pytest.raises(DLLMError, match=r"rewrote 1 position.*first at 0"):
        Tampering(steps=4).fill(request)


def test_the_diagnostics_say_what_the_model_was_shown():
    """A result file from after the hint has to be distinguishable from one from before it."""
    request, _ = tool_request()
    hint = hint_for(request)
    backend = ReplayBackend([ord(c) for c in hint + request.sequence.text], steps=4)

    backend.fill(request)

    assert backend.diagnostics()["hint"] == hint
    assert backend.diagnostics()["hint_tokens"] == len(hint)


def test_a_fill_with_no_hint_builds_the_prompt_it_always_did():
    """The regression guard: no hint, no prefix, and the same token sequence as before C-2."""
    request, _ = tool_request()
    plain = without_tools(request)
    backend = ReplayBackend([ord(c) for c in plain.sequence.text], steps=4)

    backend.fill(plain)

    assert backend.seen_prompt == masked_token_ids(plain.alignment, MASK)
    assert backend.diagnostics()["hint"] is None
    assert backend.diagnostics()["hint_tokens"] == 0


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
