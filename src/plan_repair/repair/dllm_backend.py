"""The diffusion model, kept behind a seam.

The same shape as the LLM client of Ticket B-2: everything about *how* the masked positions get
filled lives behind one call, so a repairer reads as "mask, fill, parse" and a second model is a
second object rather than an edit.

**What a backend returns.** Not a sequence — a filling per masked step. That is deliberate. The
guarantee of Ticket B-3a is that text outside the mask is copied verbatim, and it holds because
``fill_masked`` reassembles the sequence rather than trusting anyone to leave it alone. A backend
that returned whole text could quietly rewrite a healthy step and the guarantee would become a
hope. A masked diffusion model keeps the sequence length fixed, so reading each step back out of
its own token range is well defined.

A filling of ``None`` means the step should not be there — how a regeneration says "delete this".

This module defines the interface and a mock. Loading weights and running denoising belongs to
the next ticket; nothing here imports torch or a model.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from plan_repair.repair.remask import MaskSpec, PlanSequence
from plan_repair.repair.tokenization import TokenAlignment
from plan_repair.schema.plan import AgentPlan
from plan_repair.schema.task import AgentTask


class DLLMError(RuntimeError):
    """The model could not be reached, or returned nothing usable."""


class FillRequest(BaseModel):
    """Everything a backend needs to regenerate the masked region.

    Both levels of the mask are carried: steps and character spans for a backend that works on
    text, token indices for one that works on tokens.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    sequence: PlanSequence
    mask: MaskSpec
    alignment: TokenAlignment | None
    task: AgentTask


@runtime_checkable
class DLLMBackend(Protocol):
    """Fills the masked region. Raises :class:`DLLMError` on failure."""

    name: str

    def fill(self, request: FillRequest) -> dict[str, str | None]:
        """Return the replacement text for each masked step id."""
        ...


class OracleBackend:
    """Fills each masked span with the step as it was before the corruption.

    The ceiling, and the reason the pipeline can be judged without a model: if the mask and the
    token alignment are right, filling with the known answer restores exactly the damaged steps
    and leaves every other one untouched. Any collateral here is a fault in the masking, not in
    the filling.
    """

    name = "oracle"

    def __init__(self, reference_plan: AgentPlan) -> None:
        self._reference = {step.id: step for step in reference_plan.steps}

    def fill(self, request: FillRequest) -> dict[str, str | None]:
        import json

        filling: dict[str, str | None] = {}
        for span in request.mask.spans:
            original = self._reference.get(span.step_id)
            if original is None:
                # No such step before the corruption, so the answer is that it should not exist.
                if span.field is None:
                    filling[span.key] = None
                continue
            payload = original.model_dump()
            filling[span.key] = json.dumps(
                payload if span.field is None else payload[span.field], ensure_ascii=False
            )
        return filling


class EchoBackend:
    """Returns each masked step unchanged — a model that regenerates what was already there.

    Useful for separating "the mask was wrong" from "the filling was wrong": with this backend a
    repair changes nothing at all, so anything the scoring reports came from the plumbing.
    """

    name = "echo"

    def fill(self, request: FillRequest) -> dict[str, str | None]:
        text = request.sequence.text
        return {span.key: text[span.start : span.end] for span in request.mask.spans}


class FailingBackend:
    """Raises, to exercise the failure path."""

    name = "failing"

    def __init__(self, message: str = "backend unavailable") -> None:
        self._message = message

    def fill(self, request: FillRequest) -> dict[str, str | None]:
        raise DLLMError(self._message)


__all__ = [
    "DLLMBackend",
    "DLLMError",
    "EchoBackend",
    "FailingBackend",
    "FillRequest",
    "OracleBackend",
]
