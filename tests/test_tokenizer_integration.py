"""The alignment against the tokenizers the models actually use.

Separated from the pure-logic tests because these need transformers and, the first time, the
network. They skip rather than fail where that is unavailable, so the suite stays runnable
offline — but skipping is not passing, and what they check cannot be checked any other way: the
boundaries a real vocabulary produces are not the boundaries a fake one produces.

Only tokenizers are loaded here. No weights, no model, no GPU.
"""

import itertools

import pytest

from plan_repair.corruption import UNKNOWN_MODE, inject_broken_dependency, inject_wrong_tool
from plan_repair.data import DATA_PIPELINE_A, DATA_PIPELINE_B, load_reference
from plan_repair.repair import (
    ByteOffsetTokenizer,
    HuggingFaceTokenizer,
    OracleBackend,
    align_mask,
    decode_spans,
    field_spans,
    fill_masked,
    mask_spec,
    mask_spec_from_paths,
    masked_token_ids,
    plan_to_sequence,
    repair_and_score,
    sequence_to_plan,
)
from plan_repair.repair.diffusion import (
    DREAM_MASK_TOKEN,
    DREAM_MASK_TOKEN_ID,
    DREAM_MODEL,
    LLADA_MASK_TOKEN,
    LLADA_MASK_TOKEN_ID,
    LLADA_MODEL,
    DreamRepairer,
    LLaDARepairer,
)
from plan_repair.validation import validate_plan

pytest.importorskip("transformers", reason="tokenizer integration needs transformers")

DOMAINS = [DATA_PIPELINE_A, DATA_PIPELINE_B]
_CACHE: dict[str, object] = {}


def load_tokenizer(model_id: str):
    """Load a tokenizer, skipping the test when it cannot be fetched."""
    if model_id in _CACHE:
        return _CACHE[model_id]
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:  # offline, rate limited, repo moved
        pytest.skip(f"cannot load {model_id}: {type(exc).__name__}: {exc}")
    _CACHE[model_id] = tokenizer
    return tokenizer


def llada():
    return HuggingFaceTokenizer(load_tokenizer(LLADA_MODEL), "llada")


def dream():
    return ByteOffsetTokenizer(load_tokenizer(DREAM_MODEL), "dream")


def masked_case(domain=DATA_PIPELINE_B):
    task, plan = load_reference(domain)
    fan_in = next(step for step in plan.steps if len(step.input_from) > 1)
    corruption = inject_broken_dependency(plan, step_id=fan_in.id, mode=UNKNOWN_MODE)
    broken = corruption.broken_plan
    sequence = plan_to_sequence(broken)
    spec = mask_spec(sequence, validate_plan(broken, task).detected_step_ids())
    return task, plan, corruption, sequence, spec


def field_case(domain, corrupt):
    """The same, with the mask narrowed to the fields the validator named."""
    task, plan = load_reference(domain)
    fan_in = next(step for step in plan.steps if len(step.input_from) > 1)
    corruption = corrupt(plan, fan_in.id)
    broken = corruption.broken_plan
    sequence = plan_to_sequence(broken)
    spec = mask_spec_from_paths(sequence, broken, validate_plan(broken, task).detected_paths())
    return task, plan, corruption, sequence, spec


WRONG_TOOL = (lambda plan, step_id: inject_wrong_tool(plan, step_id=step_id), "wrong_tool")
BROKEN_DEP = (
    lambda plan, step_id: inject_broken_dependency(plan, step_id=step_id, mode=UNKNOWN_MODE),
    "broken_dependency",
)
FIELD_CASES = [WRONG_TOOL, BROKEN_DEP]


# --- the reconstruction, checked against a tokenizer that knows the answer ---------------


QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def test_the_reconstruction_reproduces_a_fast_tokenizers_own_offsets():
    """The slow path is not trusted on its reasoning: it has to reproduce a known answer.

    Dream ships no fast tokenizer, so its offsets have no ground truth of their own. It is a
    conversion of Qwen2.5 though, and on these sequences the two produce identical token ids —
    asserted here first, because the comparison means nothing otherwise. Given identical ids,
    Qwen2.5's own offsets are the right answer for Dream's tokenization, and the byte-level
    reconstruction has to match them exactly.

    Qwen2.5 appears only as a measuring stick. Nothing in the repairers uses it.
    """
    dream_raw = load_tokenizer(DREAM_MODEL)
    qwen_raw = load_tokenizer(QWEN_MODEL)
    reconstructed = ByteOffsetTokenizer(dream_raw, "dream")
    truth = HuggingFaceTokenizer(qwen_raw, "qwen")

    for domain in DOMAINS:
        _, plan = load_reference(domain)
        text = plan_to_sequence(plan).text

        ids, offsets = reconstructed.encode_with_offsets(text)
        truth_ids, truth_offsets = truth.encode_with_offsets(text)

        assert ids == truth_ids, f"{domain}: Dream and Qwen2.5 no longer tokenize alike"
        assert offsets == truth_offsets, domain


@pytest.mark.parametrize("domain", DOMAINS)
def test_every_character_belongs_to_some_token(domain):
    """Coverage, not partition.

    A character that costs several tokens is reported by each of them, so token ranges overlap on
    multi-byte text — both vocabularies do this, and it is the safe direction: a token holding any
    byte of a character counts as touching it.
    """
    _, plan = load_reference(domain)
    text = plan_to_sequence(plan).text

    for tokenizer in (llada(), dream()):
        ids, offsets = tokenizer.encode_with_offsets(text)
        covered = {position for start, end in offsets for position in range(start, end)}

        assert len(ids) == len(offsets)
        assert offsets[0][0] == 0
        assert offsets[-1][1] == len(text)
        assert covered == set(range(len(text))), tokenizer.name
        for (_, end), (start, _) in itertools.pairwise(offsets):
            assert start <= end, f"{tokenizer.name}: offsets went backwards at {end}/{start}"


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_token_ids_decode_back_to_the_sequence(domain):
    """Korean goal strings are where a character-by-character reconstruction would drift."""
    _, plan = load_reference(domain)
    text = plan_to_sequence(plan).text
    assert any(ord(character) > 0x3000 for character in text)

    for tokenizer, raw in (
        (llada(), load_tokenizer(LLADA_MODEL)),
        (dream(), load_tokenizer(DREAM_MODEL)),
    ):
        ids, _ = tokenizer.encode_with_offsets(text)

        assert raw.decode(ids) == text, tokenizer.name


# --- the contract, on real vocabularies -----------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_no_real_token_of_a_healthy_step_is_ever_masked(domain):
    """The whole point of the alignment, checked where the boundaries are not of our choosing."""
    _, _, _, sequence, spec = masked_case(domain)
    preserved = [
        (span.start, span.end)
        for span in sequence.spans
        if span.step_id in set(spec.preserved_step_ids)
    ]

    for tokenizer in (llada(), dream()):
        alignment = align_mask(sequence, spec, tokenizer)

        assert alignment.masked_token_count > 0
        for index in alignment.masked_token_indices:
            start, end = alignment.offsets[index]
            for low, high in preserved:
                assert not (start < high and low < end), (
                    f"{tokenizer.name}/{domain}: token {index} at {start}:{end} "
                    f"reaches into a healthy step at {low}:{high}"
                )


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_masked_region_is_a_small_part_of_the_sequence(domain):
    """Selective in token terms, not only in character terms."""
    _, _, _, sequence, spec = masked_case(domain)

    for tokenizer in (llada(), dream()):
        alignment = align_mask(sequence, spec, tokenizer)
        share = alignment.masked_token_count / len(alignment.token_ids)

        assert 0 < share < 0.35, f"{tokenizer.name}/{domain}: masked {share:.1%} of the tokens"


def test_widening_a_real_alignment_reaches_healthy_steps():
    """Precision is a property of the alignment, not of these vocabularies being convenient.

    Real tokens are small, and the separator between two step lines is six characters, so the
    slip has to be a few tokens wide before it crosses into a neighbour. The number is reported
    rather than assumed: whatever it is, the correct alignment sits below it.
    """
    _, _, _, sequence, spec = masked_case()
    preserved = [
        (span.start, span.end)
        for span in sequence.spans
        if span.step_id in set(spec.preserved_step_ids)
    ]

    for tokenizer in (llada(), dream()):
        alignment = align_mask(sequence, spec, tokenizer)

        def touches(indices, offsets=alignment.offsets):
            return any(
                start < high and low < end
                for index in indices
                for start, end in [offsets[index]]
                for low, high in preserved
            )

        def widened(by, alignment=alignment):
            out = set(alignment.masked_token_indices)
            for index in list(out):
                for step in range(1, by + 1):
                    out.add(max(index - step, 0))
                    out.add(min(index + step, len(alignment.token_ids) - 1))
            return out

        assert not touches(alignment.masked_token_indices), tokenizer.name
        slips = [by for by in range(1, 9) if touches(widened(by))]
        assert slips, f"{tokenizer.name}: no slip up to 8 tokens reached a healthy step"


# --- the model-facing pieces ----------------------------------------------------------------------


def test_the_configured_mask_tokens_are_the_real_ones():
    """Neither id is discoverable from the tokenizer, so both are pinned against the vocabulary."""
    llada_raw = load_tokenizer(LLADA_MODEL)
    dream_raw = load_tokenizer(DREAM_MODEL)

    assert llada_raw.convert_ids_to_tokens(LLADA_MASK_TOKEN_ID) == LLADA_MASK_TOKEN
    assert llada_raw.mask_token is None  # which is why the id is configured, not read
    assert dream_raw.convert_ids_to_tokens(DREAM_MASK_TOKEN_ID) == DREAM_MASK_TOKEN
    assert dream_raw.mask_token == DREAM_MASK_TOKEN


def test_substituting_the_mask_token_keeps_the_sequence_length():
    """A masked diffusion model fills fixed positions, so the length must not move."""
    _, _, _, sequence, spec = masked_case()

    for tokenizer, mask_id in ((llada(), LLADA_MASK_TOKEN_ID), (dream(), DREAM_MASK_TOKEN_ID)):
        alignment = align_mask(sequence, spec, tokenizer)
        substituted = masked_token_ids(alignment, mask_id)

        assert len(substituted) == len(alignment.token_ids)
        assert substituted.count(mask_id) >= alignment.masked_token_count


def test_a_masked_step_can_be_read_back_out_of_its_token_range():
    """What a backend does after denoising: recover each step from the range it occupied."""
    _, _, _, sequence, spec = masked_case()

    for tokenizer, raw in (
        (llada(), load_tokenizer(LLADA_MODEL)),
        (dream(), load_tokenizer(DREAM_MODEL)),
    ):
        alignment = align_mask(sequence, spec, tokenizer)

        recovered = decode_spans(alignment, alignment.token_ids, raw)

        assert set(recovered) == set(spec.masked_step_ids)
        for step_id, text in recovered.items():
            assert f'"id": "{step_id}"' in text, tokenizer.name


# --- field masks, whose spans are short enough for tokenization to matter -------------------------


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize(("corrupt", "label"), FIELD_CASES)
def test_a_field_span_is_short_but_never_too_short_to_mask(corrupt, label, domain):
    """The risk field masking introduces: a span of eight characters may fall inside one token.

    A whole step is a hundred characters and always gets tokens of its own. ``"join_x"`` is eight,
    and if the tokenizer merged it into a token shared with a preserved step the mask would come
    back empty — nothing to regenerate, so no repair. It does not happen on either vocabulary, and
    that is a measurement rather than a guarantee: the assertion is here so the day it changes is
    the day a test fails rather than the day a result silently drops to zero.
    """
    _, _, _, sequence, spec = field_case(domain, corrupt)
    field_spans_masked = [span for span in spec.spans if span.field is not None]
    assert field_spans_masked, label

    for tokenizer in (llada(), dream()):
        alignment = align_mask(sequence, spec, tokenizer)

        for span in field_spans_masked:
            assert span.key in alignment.token_ranges, (
                f"{tokenizer.name}/{domain}/{label}: {span.key} covers "
                f"{span.end - span.start} characters and got no token of its own"
            )
            start, end = alignment.token_ranges[span.key]
            assert end > start


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize(("corrupt", "label"), FIELD_CASES)
def test_a_field_mask_reaches_no_further_than_the_separators_beside_it(corrupt, label, domain):
    """Where the token boundaries fall relative to a field, measured rather than assumed.

    A field value does not start on a token boundary — the token holding its opening quote also
    holds the space before it, and the one holding its last character also holds the comma after.
    So the mask covers one character more on each side than the span asks for. Those characters
    are the separator, which reassembly writes itself (``_rewrite_fields``), so what the model
    does with them is discarded. The next field's *value* is what must stay out of reach.
    """
    _, _, _, sequence, spec = field_case(domain, corrupt)

    for tokenizer in (llada(), dream()):
        alignment = align_mask(sequence, spec, tokenizer)

        for span in (span for span in spec.spans if span.field is not None):
            first, last = alignment.token_ranges[span.key]
            low = alignment.offsets[first][0]
            high = alignment.offsets[last - 1][1]

            assert span.start - low <= 1, f"{tokenizer.name}/{domain}/{label}/{span.key}"
            assert high - span.end <= 1, f"{tokenizer.name}/{domain}/{label}/{span.key}"
            assert sequence.text[low : span.start].strip(" ,") == ""
            assert sequence.text[span.end : high].strip(" ,") == ""


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize(("corrupt", "label"), FIELD_CASES)
def test_no_token_of_a_narrowed_steps_produces_tag_is_masked(corrupt, label, domain):
    """The reason for the ticket, at the level the model actually sees.

    ``align_mask`` protects healthy *steps*; within a step that is being repaired it protects
    nothing, so this is not implied by the earlier tests and has to be checked against real token
    boundaries. Steps masked whole — the dangling step a broken edge leaves behind, which names no
    field to narrow to — are excluded, because for those the tag is inside the mask by design.
    """
    _, _, _, sequence, spec = field_case(domain, corrupt)
    narrowed = {span.step_id for span in spec.spans if span.field is not None}
    whole = {span.step_id for span in spec.spans if span.field is None}
    plan_steps = {span.step_id: span for span in sequence.spans}
    assert narrowed - whole, label

    _, _, corruption, _, _ = field_case(domain, corrupt)
    broken = {step.id: step for step in corruption.broken_plan.steps}
    tags = {
        step_id: field_spans(broken[step_id], plan_steps[step_id].start)["produces"]
        for step_id in narrowed - whole
    }

    for tokenizer in (llada(), dream()):
        alignment = align_mask(sequence, spec, tokenizer)

        for index in alignment.masked_token_indices:
            start, end = alignment.offsets[index]
            for step_id, (low, high) in tags.items():
                assert not (start < high and low < end), (
                    f"{tokenizer.name}/{domain}/{label}: token {index} at {start}:{end} "
                    f"reaches {step_id}.produces at {low}:{high} — the tag would be regenerated"
                )


@pytest.mark.parametrize(("corrupt", "label"), FIELD_CASES)
def test_what_the_gpu_backend_reads_back_is_what_reassembly_accepts(corrupt, label):
    """The seam a real run goes through, with the denoising replaced by the identity.

    ``TorchDLLMBackend.fill`` ends in ``decode_spans``, so its keys are whatever the alignment
    recorded — span keys now, step ids before. Nothing in that backend changed, which is only
    safe if the two ends still agree; ``fill_masked`` rejects a key it does not know, so a
    disagreement would surface on the GPU as a failed repair rather than here.

    Decoding a field's token range hands back the separators either side of it (``' "join_x",'``)
    because the boundary tokens straddle them. Reassembly owns those characters, so the round
    trip has to come back clean anyway.
    """
    _, _, corruption, sequence, spec = field_case(DATA_PIPELINE_B, corrupt)

    for tokenizer, raw in (
        (llada(), load_tokenizer(LLADA_MODEL)),
        (dream(), load_tokenizer(DREAM_MODEL)),
    ):
        alignment = align_mask(sequence, spec, tokenizer)

        recovered = decode_spans(alignment, alignment.token_ids, raw)

        assert set(recovered) == {span.key for span in spec.spans}, f"{tokenizer.name}/{label}"
        assert sequence_to_plan(fill_masked(sequence, spec, recovered, corruption.broken_plan)) == (
            corruption.broken_plan
        ), f"{tokenizer.name}/{label}"


def test_narrowing_the_mask_shrinks_it_in_token_terms_too():
    """87-92% fewer characters was the point; this is the same reduction counted in tokens."""
    _, plan = load_reference()
    fan_in = next(step for step in plan.steps if len(step.input_from) > 1)
    _, _, _, sequence, narrow = field_case(DATA_PIPELINE_B, WRONG_TOOL[0])
    wide = mask_spec(sequence, [fan_in.id])

    for tokenizer in (llada(), dream()):
        narrowed = align_mask(sequence, narrow, tokenizer).masked_token_count
        whole = align_mask(sequence, wide, tokenizer).masked_token_count

        assert 0 < narrowed < whole / 4, f"{tokenizer.name}: {narrowed} of {whole} tokens"


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize(("corrupt", "label"), FIELD_CASES)
def test_a_field_mask_repairs_end_to_end_on_real_tokenizers(corrupt, label, domain):
    task, plan, corruption, _, _ = field_case(domain, corrupt)

    for repairer_class, tokenizer in ((LLaDARepairer, llada()), (DreamRepairer, dream())):
        repairer = repairer_class(OracleBackend(plan), tokenizer)
        repaired, score = repair_and_score(
            repairer,
            reference_plan=plan,
            broken_plan=corruption.broken_plan,
            task=task,
            damaged_step_ids=corruption.injected[0].damaged_step_ids,
        )

        assert score.solved, f"{repairer.name}/{domain}/{label}"
        assert (
            score.collateral_modified,
            score.collateral_renamed,
            score.collateral_removed,
        ) == (0, 0, 0), f"{repairer.name}/{domain}/{label}"
        assert repaired == plan


# --- the whole pipeline, on real tokenizers -------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_pipeline_runs_end_to_end_with_real_tokenizers(domain):
    task, plan, corruption, _, _ = masked_case(domain)

    for repairer_class, tokenizer in ((LLaDARepairer, llada()), (DreamRepairer, dream())):
        repairer = repairer_class(OracleBackend(plan), tokenizer)
        repaired, score = repair_and_score(
            repairer,
            reference_plan=plan,
            broken_plan=corruption.broken_plan,
            task=task,
            damaged_step_ids=corruption.injected[0].damaged_step_ids,
        )

        assert score.solved, repairer.name
        assert (
            score.collateral_modified,
            score.collateral_renamed,
            score.collateral_removed,
        ) == (0, 0, 0), repairer.name
        assert repaired == plan
