"""Character-to-token alignment, on a tokenizer small enough to reason about.

No network and no model here: a fake tokenizer with hand-chosen boundaries makes it possible to
state exactly which token should be masked and why. The real tokenizers are exercised separately,
in the integration tests.
"""

import pytest

from plan_repair.corruption import UNKNOWN_MODE, inject_broken_dependency
from plan_repair.data import DATA_PIPELINE_A, DATA_PIPELINE_B, load_reference
from plan_repair.repair import (
    OffsetTokenizer,
    TokenizationError,
    align_mask,
    mask_spec,
    masked_token_ids,
    plan_to_sequence,
)
from plan_repair.repair.tokenization import _byte_to_char_index
from plan_repair.validation import validate_plan

DOMAINS = [DATA_PIPELINE_A, DATA_PIPELINE_B]


class CharTokenizer:
    """One token per character — every boundary lands exactly where a span does."""

    name = "char"

    def encode_with_offsets(self, text):
        return [ord(c) for c in text], [(i, i + 1) for i in range(len(text))]


class ChunkTokenizer:
    """Fixed-width tokens, so span boundaries fall inside tokens on purpose."""

    def __init__(self, width: int) -> None:
        self.name = f"chunk{width}"
        self._width = width

    def encode_with_offsets(self, text):
        offsets = [(i, min(i + self._width, len(text))) for i in range(0, len(text), self._width)]
        return [hash(text[a:b]) % 1000 for a, b in offsets], offsets


def masked_case(domain=DATA_PIPELINE_B):
    task, plan = load_reference(domain)
    fan_in = next(step for step in plan.steps if len(step.input_from) > 1)
    broken = inject_broken_dependency(plan, step_id=fan_in.id, mode=UNKNOWN_MODE).broken_plan
    sequence = plan_to_sequence(broken)
    spec = mask_spec(sequence, validate_plan(broken, task).detected_step_ids())
    return sequence, spec


def test_the_fakes_satisfy_the_tokenizer_port():
    assert isinstance(CharTokenizer(), OffsetTokenizer)
    assert isinstance(ChunkTokenizer(4), OffsetTokenizer)


@pytest.mark.parametrize("domain", DOMAINS)
def test_masked_tokens_cover_the_masked_characters(domain):
    sequence, spec = masked_case(domain)

    alignment = align_mask(sequence, spec, CharTokenizer())

    masked_chars = {i for span in spec.spans for i in range(span.start, span.end)}
    assert {alignment.offsets[i][0] for i in alignment.masked_token_indices} == masked_chars


@pytest.mark.parametrize("domain", DOMAINS)
def test_no_masked_token_touches_a_healthy_step(domain):
    """The contract. Everything else in this file exists to make this one meaningful."""
    sequence, spec = masked_case(domain)
    preserved = {
        (span.start, span.end)
        for span in sequence.spans
        if span.step_id in set(spec.preserved_step_ids)
    }

    for tokenizer in (CharTokenizer(), ChunkTokenizer(3), ChunkTokenizer(7), ChunkTokenizer(64)):
        alignment = align_mask(sequence, spec, tokenizer)
        for index in alignment.masked_token_indices:
            start, end = alignment.offsets[index]
            for low, high in preserved:
                assert not (start < high and low < end), (
                    f"{tokenizer.name}: token {index} at {start}:{end} reaches into {low}:{high}"
                )


def test_a_token_straddling_two_steps_is_left_alone():
    """One healthy character in a token is enough to protect it, whatever else it carries."""
    sequence, spec = masked_case()
    # A window wide enough to swallow the gap between two step lines.
    alignment = align_mask(sequence, spec, ChunkTokenizer(200))

    assert alignment.masked_token_indices == [] or all(
        not any(
            start < span.end and span.start < end
            for span in sequence.spans
            if span.step_id in set(spec.preserved_step_ids)
        )
        for start, end in (alignment.offsets[i] for i in alignment.masked_token_indices)
    )


@pytest.mark.parametrize("width", [3, 5, 7, 13])
def test_a_masked_step_is_covered_except_where_a_healthy_step_shares_a_token(width):
    """The exact trade-off: overlap is enough to mask, unless a healthy step is in the way.

    Every character of a masked step is inside a masked token, and the only ones that are not sit
    in a token that a healthy step also occupies — where protecting the healthy step wins.
    """
    sequence, spec = masked_case()
    alignment = align_mask(sequence, spec, ChunkTokenizer(width))
    preserved = [
        (span.start, span.end)
        for span in sequence.spans
        if span.step_id in set(spec.preserved_step_ids)
    ]
    covered = {
        position
        for index in alignment.masked_token_indices
        for position in range(*alignment.offsets[index])
    }

    for span in spec.spans:
        for position in range(span.start, span.end):
            if position in covered:
                continue
            token = next(
                (start, end) for start, end in alignment.offsets if start <= position < end
            )
            assert any(token[0] < high and low < token[1] for low, high in preserved), (
                f"width {width}: char {position} of {span.step_id} is neither masked nor shared"
            )


def test_spilling_tokens_are_reported_not_hidden():
    """A masked token reaching into the JSON scaffolding is legal but visible."""
    sequence, spec = masked_case()

    fine = align_mask(sequence, spec, CharTokenizer())
    coarse = align_mask(sequence, spec, ChunkTokenizer(5))

    assert fine.spilling_token_indices == []
    assert coarse.spilling_token_indices  # wide tokens necessarily reach past the span
    assert set(coarse.spilling_token_indices) <= set(coarse.masked_token_indices)


def test_token_ranges_name_the_span_each_step_occupies():
    sequence, spec = masked_case()

    alignment = align_mask(sequence, spec, CharTokenizer())

    assert set(alignment.token_ranges) == set(spec.masked_step_ids)
    for step_id, (start, end) in alignment.token_ranges.items():
        span = next(s for s in spec.spans if s.step_id == step_id)
        assert alignment.offsets[start][0] == span.start
        assert alignment.offsets[end - 1][1] == span.end


def test_an_empty_mask_masks_no_tokens():
    _, plan = load_reference()
    sequence = plan_to_sequence(plan)
    spec = mask_spec(sequence, set())

    alignment = align_mask(sequence, spec, CharTokenizer())

    assert alignment.masked_token_indices == []
    assert alignment.token_ranges == {}


def test_mask_token_substitution_touches_only_masked_positions():
    sequence, spec = masked_case()
    alignment = align_mask(sequence, spec, CharTokenizer())

    substituted = masked_token_ids(alignment, mask_token_id=999999)

    assert len(substituted) == len(alignment.token_ids)
    for index, (before, after) in enumerate(zip(alignment.token_ids, substituted, strict=True)):
        assert after == (999999 if index in set(alignment.masked_token_indices) else before)


def test_a_tokenizer_that_miscounts_is_refused():
    class Broken:
        name = "broken"

        def encode_with_offsets(self, text):
            return [1, 2, 3], [(0, 1)]

    sequence, spec = masked_case()

    with pytest.raises(TokenizationError, match="3 ids for 1 offsets"):
        align_mask(sequence, spec, Broken())


# --- the byte index the slow path is built on ----------------------------------------------------


def test_byte_to_char_index_handles_multibyte_text():
    """The reason offsets are reconstructed through bytes rather than decoded characters."""
    text = "a한b"  # 1 + 3 + 1 bytes

    index = _byte_to_char_index(text)

    assert index == [0, 1, 1, 1, 2, 3]
    assert len(index) == len(text.encode("utf-8")) + 1


def test_byte_to_char_index_is_exact_on_the_reference_plans():
    for domain in DOMAINS:
        _, plan = load_reference(domain)
        text = plan_to_sequence(plan).text
        index = _byte_to_char_index(text)

        assert len(index) == len(text.encode("utf-8")) + 1
        for position, character in enumerate(text):
            byte_start = len(text[:position].encode("utf-8"))
            assert index[byte_start] == position
            assert text[index[byte_start]] == character


# --- the mutation: is the precision earned? -------------------------------------------------------


def widen(alignment, sequence, spec, by=1):
    """Push the mask ``by`` tokens further in each direction, as a wrong alignment would."""
    masked = set(alignment.masked_token_indices)
    for index in sorted(masked):
        for offset in range(1, by + 1):
            masked.add(max(index - offset, 0))
            masked.add(min(index + offset, len(alignment.token_ids) - 1))
    return sorted(masked)


@pytest.mark.parametrize("domain", DOMAINS)
def test_widening_the_alignment_reaches_healthy_steps(domain):
    """Correct alignment leaves healthy tokens alone; a slipped alignment does not.

    Two tokens rather than one: the separator between step lines is six characters, so a single
    four-character token does not always clear it. The point is that the precision is a property
    of the alignment, not of the tokenizer being coarse.
    """
    sequence, spec = masked_case(domain)
    alignment = align_mask(sequence, spec, ChunkTokenizer(4))
    preserved = [
        (span.start, span.end)
        for span in sequence.spans
        if span.step_id in set(spec.preserved_step_ids)
    ]

    def touches_healthy(indices):
        return any(
            start < high and low < end
            for index in indices
            for start, end in [alignment.offsets[index]]
            for low, high in preserved
        )

    assert not touches_healthy(alignment.masked_token_indices)
    assert not touches_healthy(widen(alignment, sequence, spec, by=0))
    assert touches_healthy(widen(alignment, sequence, spec, by=2))
