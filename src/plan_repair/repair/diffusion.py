"""Diffusion repairers — the same five steps for every model.

    mask the flagged steps  →  align the mask to tokens  →  fill  →  reassemble  →  parse

Only the last two lines of that differ between models, and both differences sit behind objects:
the tokenizer decides where the token boundaries fall, the backend decides what goes in the holes.
:class:`LLaDARepairer` and :class:`DreamRepairer` are therefore the same class with different
parts, which is what makes a comparison between them a comparison of models.

**What each model brings.** LLaDA was trained as a diffusion model from scratch and ships a fast
tokenizer; its mask token is ``<|mdm_mask|>`` but it is not declared as the tokenizer's
``mask_token``, so the id is configured explicitly rather than discovered. Dream was converted
from an autoregressive model and ships only a slow tokenizer, so its offsets are reconstructed
from the byte vocabulary (see :mod:`plan_repair.repair.tokenization`); it does declare
``<|mask|>``. Neither fact is inferable at runtime with any confidence, so both are stated here.

**Range.** A step-level mask cannot reach a step that was deleted, the order of the steps, or a
plan-level field — established in Ticket B-3a and unchanged by putting a model behind it. Those
repairs fail, and they fail without touching anything healthy.

Nothing here loads weights or imports torch. The backend is an interface; the next ticket puts a
model behind it.
"""

from plan_repair.repair.ar import PARSE_FAILURE, RepairFailure
from plan_repair.repair.diagnostics import RepairDiagnostics, diagnose
from plan_repair.repair.dllm_backend import DLLMBackend, DLLMError, FillRequest
from plan_repair.repair.plan_io import PlanParseError
from plan_repair.repair.remask import (
    MaskSpec,
    fill_masked,
    mask_spec_from_paths,
    plan_to_sequence,
    sequence_to_plan,
)
from plan_repair.repair.tokenization import OffsetTokenizer, TokenAlignment, align_mask
from plan_repair.schema.plan import AgentPlan
from plan_repair.schema.task import AgentTask
from plan_repair.validation.models import PlanValidationResult

BACKEND_FAILURE = "backend"
ALIGNMENT_FAILURE = "alignment"

# The token each model writes into a masked position. Neither is reliably discoverable:
# LLaDA does not declare a mask_token at all, and a wrong id here would be silent.
LLADA_MODEL = "GSAI-ML/LLaDA-8B-Instruct"
LLADA_MASK_TOKEN = "<|mdm_mask|>"
LLADA_MASK_TOKEN_ID = 126336

DREAM_MODEL = "Dream-org/Dream-v0-Instruct-7B"
DREAM_MASK_TOKEN = "<|mask|>"
DREAM_MASK_TOKEN_ID = 151666


class DiffusionRepairer:
    """Mask, align, fill, reassemble, parse."""

    name = "diffusion"
    model_id = ""
    mask_token = ""
    mask_token_id = -1

    def __init__(self, backend: DLLMBackend, tokenizer: OffsetTokenizer | None = None) -> None:
        self._backend = backend
        self._tokenizer = tokenizer
        self.failures: list[RepairFailure] = []
        self.last_mask: MaskSpec | None = None
        self.last_alignment: TokenAlignment | None = None
        # What the model produced last time, kept whether or not it parsed.
        self.last_diagnostics: RepairDiagnostics | None = None

    def repair(
        self,
        broken_plan: AgentPlan,
        validation: PlanValidationResult,
        task: AgentTask,
    ) -> AgentPlan:
        sequence = plan_to_sequence(broken_plan)
        spec = mask_spec_from_paths(sequence, broken_plan, validation.detected_paths())
        self.last_mask = spec

        alignment: TokenAlignment | None = None
        if self._tokenizer is not None:
            try:
                alignment = align_mask(sequence, spec, self._tokenizer)
            except Exception as exc:  # a tokenizer failure is a failed repair, not a crash
                return self._give_up(broken_plan, ALIGNMENT_FAILURE, str(exc))
        self.last_alignment = alignment

        request = FillRequest(sequence=sequence, mask=spec, alignment=alignment, task=task)
        try:
            filling = self._backend.fill(request)
        except DLLMError as exc:
            return self._give_up(broken_plan, BACKEND_FAILURE, str(exc))

        # fill_masked refuses anything outside the mask, so a backend cannot reach a healthy step
        # even by mistake.
        try:
            filled = fill_masked(sequence, spec, filling, broken_plan)
        except ValueError as exc:
            return self._give_up(broken_plan, BACKEND_FAILURE, str(exc))

        try:
            repaired = sequence_to_plan(filled)
        except PlanParseError as exc:
            # The text is kept before anything else happens to it: this is the only point where
            # what the model actually produced still exists.
            self.last_diagnostics = diagnose(
                raw_text=filled, fillings=filling, mask_token=self.mask_token, error=exc
            )
            return self._give_up(broken_plan, PARSE_FAILURE, str(exc))
        self.last_diagnostics = diagnose(
            raw_text=filled, fillings=filling, mask_token=self.mask_token
        )
        return repaired

    def _give_up(self, broken_plan: AgentPlan, kind: str, detail: str) -> AgentPlan:
        self.failures.append(RepairFailure(repairer=self.name, kind=kind, detail=detail))
        return broken_plan.model_copy(deep=True)


class LLaDARepairer(DiffusionRepairer):
    """Diffusion from scratch; fast tokenizer; undeclared mask token."""

    name = "llada"
    model_id = LLADA_MODEL
    mask_token = LLADA_MASK_TOKEN
    mask_token_id = LLADA_MASK_TOKEN_ID


class DreamRepairer(DiffusionRepairer):
    """Converted from an autoregressive model; slow tokenizer, reconstructed offsets."""

    name = "dream"
    model_id = DREAM_MODEL
    mask_token = DREAM_MASK_TOKEN
    mask_token_id = DREAM_MASK_TOKEN_ID


__all__ = [
    "ALIGNMENT_FAILURE",
    "BACKEND_FAILURE",
    "DREAM_MASK_TOKEN",
    "DREAM_MASK_TOKEN_ID",
    "DREAM_MODEL",
    "LLADA_MASK_TOKEN",
    "LLADA_MASK_TOKEN_ID",
    "LLADA_MODEL",
    "DiffusionRepairer",
    "DreamRepairer",
    "LLaDARepairer",
]
