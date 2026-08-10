"""What the model actually produced, kept so a failure can be explained.

A repair that fails to parse is recorded honestly, which is right — but "the answer was not valid
JSON: Expecting value: line 15 column 123" says nothing about *why*. The text the model produced
was assembled, handed to the parser and dropped, so nothing downstream could tell apart the
candidate causes: denoising that stopped with mask tokens still in the sequence, a boundary token
left outside the mask so the punctuation no longer matches, or a model that simply cannot hold the
format.

This module keeps the raw text and computes the few things that separate those cases. It reads
what the repair produced and records it; it decides nothing and changes nothing.

Both counts of leftover mask tokens matter and they are not the same number. The token-level one,
taken before decoding, is the truth about whether denoising finished. The text-level one is what
survived into the string the parser saw — a tokenizer that drops special tokens on decode would
report zero there while the sequence was still full of them.
"""

import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from plan_repair.repair.snap import ToolSnap

EXCERPT_RADIUS = 50


class ParseFailure(BaseModel):
    """Where the parser gave up, and what the text looks like there."""

    model_config = ConfigDict(extra="forbid")

    message: str
    line: int | None = None
    column: int | None = None
    position: int | None = None
    # The text around the failure. "line 15 column 123" is not something anyone can act on;
    # the characters at that spot usually are.
    excerpt: str = ""


class RepairDiagnostics(BaseModel):
    """The raw output of one repair attempt, and what can be read off it."""

    model_config = ConfigDict(extra="forbid")

    raw_text: str
    raw_length: int
    # What the model produced, before the snap saw it. Kept as the model's own answer so that a
    # repair which came out solved can be read as the model's or as the snap's.
    fillings: dict[str, str | None]
    mask_token: str
    mask_tokens_remaining: int
    parsed: bool
    parse_failure: ParseFailure | None = None
    # One entry per tool field the snap considered, whether or not it fired. Empty when the snap
    # was off, which is every measurement taken before it existed.
    snaps: dict[str, ToolSnap] = Field(default_factory=dict)

    @property
    def unfinished_denoising(self) -> bool:
        """A mask token in the parsed text means the sequence was still being filled in."""
        return self.mask_tokens_remaining > 0


def diagnose(
    *,
    raw_text: str,
    fillings: dict[str, str | None],
    mask_token: str,
    error: Exception | None = None,
    snaps: Mapping[str, ToolSnap] | None = None,
) -> RepairDiagnostics:
    """Record what a repair produced, whether or not it parsed.

    ``fillings`` is what the model returned; ``snaps`` is what was done to it afterwards. Passing
    the post-snap fillings here instead would lose the model's answer, and with it the only way to
    tell which of the two a solved case belongs to.
    """
    return RepairDiagnostics(
        raw_text=raw_text,
        raw_length=len(raw_text),
        fillings=dict(fillings),
        mask_token=mask_token,
        mask_tokens_remaining=raw_text.count(mask_token) if mask_token else 0,
        parsed=error is None,
        parse_failure=None if error is None else describe_failure(raw_text, error),
        snaps=dict(snaps or {}),
    )


def describe_failure(raw_text: str, error: Exception) -> ParseFailure:
    """Turn a parse error into a place in the text.

    ``PlanParseError`` chains the error it came from, so a JSON failure still carries the offset
    it happened at even though the message has been rewritten.
    """
    cause = error.__cause__ if isinstance(error.__cause__, json.JSONDecodeError) else None
    if cause is None:
        return ParseFailure(message=str(error))
    return ParseFailure(
        message=str(error),
        line=cause.lineno,
        column=cause.colno,
        position=cause.pos,
        excerpt=excerpt_around(raw_text, cause.pos),
    )


def excerpt_around(text: str, position: int, radius: int = EXCERPT_RADIUS) -> str:
    """The text on either side of ``position``, marked with ``>>>`` where it broke."""
    if not text:
        return ""
    position = max(0, min(position, len(text)))
    start = max(0, position - radius)
    end = min(len(text), position + radius)
    return f"{text[start:position]}>>>{text[position:end]}"


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Across a batch: how many failed to parse, and how many of those were still masked.

    A high share points at the denoising budget rather than at the model's grasp of the format —
    which is the first fork to take when reading a run.
    """
    diagnosed = [result.get("diagnostics") for result in results]
    present = [entry for entry in diagnosed if entry]
    failures = [entry for entry in present if not entry.get("parsed")]
    still_masked = [entry for entry in failures if entry.get("mask_tokens_remaining", 0) > 0]
    return {
        "cases": len(results),
        "with_diagnostics": len(present),
        "parse_failures": len(failures),
        "parse_failures_with_mask_tokens_left": len(still_masked),
        "share_of_failures_still_masked": (
            round(len(still_masked) / len(failures), 3) if failures else None
        ),
    }


__all__ = [
    "EXCERPT_RADIUS",
    "ParseFailure",
    "RepairDiagnostics",
    "describe_failure",
    "diagnose",
    "excerpt_around",
    "summarise",
]
