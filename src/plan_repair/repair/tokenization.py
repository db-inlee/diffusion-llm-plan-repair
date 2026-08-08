"""Character spans to token indices.

Ticket B-3a decided *which characters* a repairer may rewrite. A diffusion model rewrites
*tokens*. This module is the join, and it is the part where a quiet mistake is expensive: a mask
that spills one token past a step boundary hands a healthy step to the model to overwrite, and
nothing downstream would notice.

**The rule.** A token is masked when it overlaps a masked span and touches no preserved step.
Overlap rather than containment, so a step is fully rewritable even when the tokenizer merges its
first character with the indentation before it; the second clause is the one that matters, and it
is absolute — a token carrying any character of a healthy step is never masked, whatever else it
carries.

**Offsets.** Fast tokenizers report the character range of every token. Slow ones do not, and one
of the two models in this project ships only a slow tokenizer, so :class:`ByteOffsetTokenizer`
reconstructs the same information from the byte-level vocabulary: each token decodes to a known
run of bytes, byte positions accumulate, and UTF-8 boundaries turn them back into character
positions. Going through bytes rather than characters is what makes it correct for text where one
character costs several tokens — decoding tokens one at a time yields replacement characters and
silently shifts every offset after the first non-ASCII word.

The reconstruction is checked against a fast tokenizer's own offsets, so the slow path is not
trusted on the strength of its reasoning alone.
"""

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from plan_repair.repair.remask import MaskSpec, PlanSequence


class TokenizationError(RuntimeError):
    """The text could not be aligned to tokens."""


@runtime_checkable
class OffsetTokenizer(Protocol):
    """Tokenizes text and says where each token came from."""

    name: str

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Return token ids and, for each, the ``[start, end)`` character range it covers."""
        ...


class TokenAlignment(BaseModel):
    """The sequence as tokens, and which of them the mask covers."""

    model_config = ConfigDict(extra="forbid")

    tokenizer: str
    token_ids: list[int]
    offsets: list[tuple[int, int]]
    masked_token_indices: list[int]
    # Masked tokens reaching outside every masked span — into the JSON scaffolding between steps.
    # Not a violation of the contract (no healthy step is involved) but worth seeing, because the
    # model rewrites those characters too.
    spilling_token_indices: list[int]
    # Keyed by span key: a step id for a whole-step mask, ``step.field`` for a field mask.
    token_ranges: dict[str, tuple[int, int]]

    @property
    def masked_token_count(self) -> int:
        return len(self.masked_token_indices)


def align_mask(
    sequence: PlanSequence, spec: MaskSpec, tokenizer: OffsetTokenizer
) -> TokenAlignment:
    """Translate the character mask of ``spec`` into token indices."""
    token_ids, offsets = tokenizer.encode_with_offsets(sequence.text)
    if len(token_ids) != len(offsets):
        raise TokenizationError(
            f"{tokenizer.name} returned {len(token_ids)} ids for {len(offsets)} offsets"
        )

    masked = [(span.start, span.end) for span in spec.spans]
    preserved = [
        (span.start, span.end)
        for span in sequence.spans
        if span.step_id in set(spec.preserved_step_ids)
    ]

    masked_indices: list[int] = []
    spilling: list[int] = []
    for index, (start, end) in enumerate(offsets):
        if start == end:  # a token covering no characters cannot belong to a span
            continue
        if not any(_overlaps(start, end, low, high) for low, high in masked):
            continue
        if any(_overlaps(start, end, low, high) for low, high in preserved):
            continue  # a healthy step owns part of this token; it is never masked
        masked_indices.append(index)
        if not any(low <= start and end <= high for low, high in masked):
            spilling.append(index)

    ranges: dict[str, tuple[int, int]] = {}
    for span in spec.spans:
        covered = [
            index
            for index in masked_indices
            if _overlaps(offsets[index][0], offsets[index][1], span.start, span.end)
        ]
        if covered:
            ranges[span.key] = (covered[0], covered[-1] + 1)

    return TokenAlignment(
        tokenizer=tokenizer.name,
        token_ids=token_ids,
        offsets=offsets,
        masked_token_indices=masked_indices,
        spilling_token_indices=spilling,
        token_ranges=ranges,
    )


def _overlaps(start: int, end: int, low: int, high: int) -> bool:
    return start < high and low < end


class HuggingFaceTokenizer:
    """A fast tokenizer, which reports offsets itself."""

    def __init__(self, tokenizer: Any, name: str) -> None:
        self._tokenizer = tokenizer
        self.name = name

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoded = self._tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
        return [int(token) for token in encoded["input_ids"]], offsets


class ByteOffsetTokenizer:
    """A slow byte-level tokenizer, whose offsets are reconstructed from its vocabulary.

    Every token of a byte-level BPE vocabulary stands for a fixed run of bytes. Summing those runs
    gives each token's byte range; the byte-to-character index of the encoded text turns it into a
    character range. Nothing here depends on decoding a token in isolation, which is what breaks
    on text where a character spans several tokens.
    """

    def __init__(self, tokenizer: Any, name: str) -> None:
        self._tokenizer = tokenizer
        self.name = name
        byte_decoder = getattr(tokenizer, "byte_decoder", None)
        if byte_decoder is None:
            raise TokenizationError(f"{name} has no byte_decoder to reconstruct offsets from")
        self._byte_decoder: dict[str, int] = byte_decoder

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        token_ids = [
            int(token) for token in self._tokenizer(text, add_special_tokens=False)["input_ids"]
        ]
        pieces = self._tokenizer.convert_ids_to_tokens(token_ids)
        char_at_byte = _byte_to_char_index(text)

        offsets: list[tuple[int, int]] = []
        cursor = 0
        for token_id, piece in zip(token_ids, pieces, strict=True):
            length = self._byte_length(piece, token_id)
            start, end = cursor, cursor + length
            if end > len(char_at_byte) - 1:
                raise TokenizationError(
                    f"{self.name}: tokens run {end} bytes past a {len(text.encode())}-byte text"
                )
            # A token holding only part of a character reports that whole character, matching
            # what fast tokenizers do. Several tokens then share one character's range, which is
            # what makes the masking rule conservative in exactly the right direction: a token
            # holding any byte of a healthy step's character counts as touching it.
            offsets.append(
                (char_at_byte[start], char_at_byte[end - 1] + 1)
                if length
                else (
                    char_at_byte[start],
                    char_at_byte[start],
                )
            )
            cursor = end
        if cursor != len(text.encode("utf-8")):
            raise TokenizationError(
                f"{self.name}: tokens cover {cursor} bytes of {len(text.encode('utf-8'))}"
            )
        return token_ids, offsets

    def _byte_length(self, piece: str, token_id: int) -> int:
        """How many bytes of the original text this token stands for."""
        added = getattr(self._tokenizer, "added_tokens_decoder", {})
        if token_id in added:  # a special token is written literally, not byte-encoded
            return len(str(added[token_id]).encode("utf-8"))
        try:
            return len(bytes(self._byte_decoder[character] for character in piece))
        except KeyError as exc:
            raise TokenizationError(
                f"{self.name}: token {piece!r} contains a character outside the byte vocabulary"
            ) from exc


def _byte_to_char_index(text: str) -> list[int]:
    """``result[b]`` is the character index at byte offset ``b`` (length: bytes + 1)."""
    index: list[int] = []
    for position, character in enumerate(text):
        index.extend([position] * len(character.encode("utf-8")))
    index.append(len(text))
    return index


def masked_token_ids(alignment: TokenAlignment, mask_token_id: int) -> list[int]:
    """The token ids with every masked position replaced — what a dLLM is handed."""
    masked = set(alignment.masked_token_indices)
    return [
        mask_token_id if index in masked else token
        for index, token in enumerate(alignment.token_ids)
    ]


def decode_spans(
    alignment: TokenAlignment, token_ids: Sequence[int], tokenizer: Any
) -> dict[str, str]:
    """Read each masked step back out of a filled token sequence.

    A masked diffusion model keeps the sequence length fixed, so the token range a step occupied
    still holds that step after denoising.
    """
    return {
        step_id: tokenizer.decode(list(token_ids[start:end]))
        for step_id, (start, end) in alignment.token_ranges.items()
    }


__all__ = [
    "ByteOffsetTokenizer",
    "HuggingFaceTokenizer",
    "OffsetTokenizer",
    "TokenAlignment",
    "TokenizationError",
    "align_mask",
    "decode_spans",
    "masked_token_ids",
]
