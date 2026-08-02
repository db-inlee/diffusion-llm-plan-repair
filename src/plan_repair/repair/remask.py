"""Selective remask — turning a validator's findings into a region of text to regenerate.

This is where the project's central claim becomes machinery. An AR repairer is *asked* to leave
the rest of the plan alone; a diffusion repairer is not given the option, because the text outside
the mask is never regenerated. Whether that structural guarantee holds comes down to one thing:
whether the mask covers the damaged steps and nothing else.

**The representation.** A plan is emitted as JSON with every step on its own line, so a step is a
contiguous, unambiguous run of characters. JSON because the parser from Ticket B-2 then reads the
filled text back unchanged, which keeps one serialization contract for the whole project; one step
per line because a boundary that needs no parsing to find is a boundary that cannot drift.

**The boundary with the model.** What ends here is *which characters belong to which step*. What
begins in the next ticket is subword tokenization, the model's own mask token and denoising:
:func:`render_masked` takes the placeholder as an argument precisely so the model can supply its
own. Nothing in this module knows what a token is.

**What step-level masking cannot reach.** A mask is a region of existing text, so a step that is
not in the broken plan has no span and cannot be regenerated, and the order of the steps is a
property of the layout rather than of any one span. Both limits are structural, not oversights;
they are recorded in the mask spec (:attr:`MaskSpec.unmaskable_step_ids`) and in the tests rather
than papered over.
"""

import json
from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ConfigDict

from plan_repair.repair.plan_io import parse_plan
from plan_repair.schema.plan import AgentPlan

STEP_INDENT = "    "
DEFAULT_PLACEHOLDER = "[MASK]"


class StepSpan(BaseModel):
    """Where one step lives in the sequence: ``text[start:end]`` is exactly that step."""

    model_config = ConfigDict(extra="forbid")

    step_id: str
    start: int
    end: int


class PlanSequence(BaseModel):
    """A plan as text, with the character range of every step recorded."""

    model_config = ConfigDict(extra="forbid")

    text: str
    spans: list[StepSpan]
    header: str
    footer: str
    step_texts: list[str]

    def span_of(self, step_id: str) -> StepSpan | None:
        return next((span for span in self.spans if span.step_id == step_id), None)

    def step_ids(self) -> list[str]:
        return [span.step_id for span in self.spans]


class MaskSpec(BaseModel):
    """Which steps are to be regenerated, and which are guaranteed to survive untouched."""

    model_config = ConfigDict(extra="forbid")

    masked_step_ids: list[str]
    preserved_step_ids: list[str]
    spans: list[StepSpan]
    # Steps the validator pointed at that have no text to mask — a deleted step, most often.
    # Selective remask cannot regenerate what is not there, and this names those cases.
    unmaskable_step_ids: list[str]

    @property
    def masked_characters(self) -> int:
        return sum(span.end - span.start for span in self.spans)


def plan_to_sequence(plan: AgentPlan) -> PlanSequence:
    """Serialize ``plan`` so that each step occupies one line and one span."""
    step_texts = [_step_text(step.model_dump()) for step in plan.steps]
    header = f'{{\n  "goal": {json.dumps(plan.goal, ensure_ascii=False)},\n  "steps": [\n'
    footer = f'  ],\n  "stop_condition": {json.dumps(plan.stop_condition, ensure_ascii=False)}\n}}'
    text, spans = _assemble(header, step_texts, footer, [step.id for step in plan.steps])
    return PlanSequence(text=text, spans=spans, header=header, footer=footer, step_texts=step_texts)


def sequence_to_plan(text: str) -> AgentPlan:
    """Read a sequence back as a plan, raising ``PlanParseError`` if it is not one.

    The same parser the AR repairers use: a filled sequence that does not come back as a valid
    plan is a repair that failed, not a text to be coaxed into shape.
    """
    return parse_plan(text)


def mask_spec(sequence: PlanSequence, step_ids: Iterable[str]) -> MaskSpec:
    """Mark ``step_ids`` for regeneration; everything else is preserved by construction."""
    wanted = set(step_ids)
    present = set(sequence.step_ids())
    spans = [span for span in sequence.spans if span.step_id in wanted]
    return MaskSpec(
        masked_step_ids=[span.step_id for span in spans],
        preserved_step_ids=[span.step_id for span in sequence.spans if span.step_id not in wanted],
        spans=spans,
        unmaskable_step_ids=sorted(wanted - present),
    )


def render_masked(
    sequence: PlanSequence, spec: MaskSpec, placeholder: str = DEFAULT_PLACEHOLDER
) -> str:
    """The sequence with every masked step replaced by ``placeholder``.

    ``placeholder`` is a parameter because the mask token belongs to whichever model fills it in.
    """
    return _rebuild(sequence, {step_id: placeholder for step_id in spec.masked_step_ids})


def fill_masked(
    sequence: PlanSequence, spec: MaskSpec, replacements: Mapping[str, str | None]
) -> str:
    """Put ``replacements`` into the masked spans and return the resulting sequence.

    A replacement of ``None`` drops that step, which is how a regeneration expresses "this step
    should not exist". Steps outside the mask are copied over verbatim — that is the guarantee
    the whole approach rests on, so it is enforced here rather than asked for.
    """
    unknown = set(replacements) - set(spec.masked_step_ids)
    if unknown:
        raise ValueError(f"cannot fill steps that are not masked: {sorted(unknown)}")
    return _rebuild(sequence, replacements)


def _rebuild(sequence: PlanSequence, replacements: Mapping[str, str | None]) -> str:
    kept: list[str] = []
    for span, original in zip(sequence.spans, sequence.step_texts, strict=True):
        if span.step_id in replacements:
            replacement = replacements[span.step_id]
            if replacement is None:
                continue
            kept.append(replacement)
        else:
            kept.append(original)
    text, _ = _assemble(sequence.header, kept, sequence.footer, [])
    return text


def _assemble(
    header: str, step_texts: list[str], footer: str, step_ids: list[str]
) -> tuple[str, list[StepSpan]]:
    parts = [header]
    spans: list[StepSpan] = []
    offset = len(header)
    last = len(step_texts) - 1
    for index, body in enumerate(step_texts):
        line = f"{STEP_INDENT}{body}{',' if index < last else ''}\n"
        if index < len(step_ids):
            start = offset + len(STEP_INDENT)
            spans.append(StepSpan(step_id=step_ids[index], start=start, end=start + len(body)))
        parts.append(line)
        offset += len(line)
    parts.append(footer)
    return "".join(parts), spans


def _step_text(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False)


__all__ = [
    "DEFAULT_PLACEHOLDER",
    "MaskSpec",
    "PlanSequence",
    "StepSpan",
    "fill_masked",
    "mask_spec",
    "plan_to_sequence",
    "render_masked",
    "sequence_to_plan",
]
