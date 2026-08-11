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

**Where steps meet.** The separator between two steps belongs to this module, not to whatever
fills the mask — see :func:`normalise_filling`. Two adjacent masked steps share that separator,
so a regeneration hands it back along with the step and the template would otherwise add a second
one.
"""

import json
from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ConfigDict

from plan_repair.repair.plan_io import parse_plan
from plan_repair.schema.plan import AgentPlan, Step
from plan_repair.validation.models import (
    DANGLING_STEP,
    UNKNOWN_DEPENDENCY,
    PlanValidationResult,
)
from plan_repair.validation.paths import parse_path

STEP_INDENT = "    "
DEFAULT_PLACEHOLDER = "[MASK]"


class StepSpan(BaseModel):
    """A region of the sequence that belongs to one step.

    With ``field`` unset the span is the whole step; with it set the span is just that field's
    value, which is what lets a repair rewrite a broken ``tool`` without the rest of the step —
    its ``produces`` above all — passing through the model at all.
    """

    model_config = ConfigDict(extra="forbid")

    step_id: str
    start: int
    end: int
    field: str | None = None

    @property
    def key(self) -> str:
        """What a filling for this span is keyed by. Equal to the step id for a whole step."""
        return self.step_id if self.field is None else f"{self.step_id}.{self.field}"


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


def field_spans(step: Step, span_start: int) -> dict[str, tuple[int, int]]:
    """Where each field's *value* sits inside the step's span.

    A step is one line of JSON written by ``json.dumps`` in the model's field order, so rendering
    the fields again one at a time and accumulating their lengths reproduces the same offsets —
    the separator between fields is always ``", "`` and the line always opens with ``{``. That is
    checked against the real text rather than assumed.
    """
    payload = step.model_dump()
    spans: dict[str, tuple[int, int]] = {}
    cursor = span_start + 1  # past the opening brace
    for index, (name, value) in enumerate(payload.items()):
        rendered_key = json.dumps(name, ensure_ascii=False) + ": "
        rendered_value = json.dumps(value, ensure_ascii=False)
        start = cursor + len(rendered_key)
        spans[name] = (start, start + len(rendered_value))
        cursor = start + len(rendered_value) + (2 if index < len(payload) - 1 else 0)
    return spans


def derived_dangling_step_ids(validation: PlanValidationResult, plan: AgentPlan) -> set[str]:
    """Steps reported as dangling only because some reference was broken.

    Breaking one dependency edge reports two things: the reference that now points nowhere, and
    the step whose output nobody consumes any more. The second is a consequence of the first —
    that step is untouched and goes back to being consumed the moment the reference is repaired.

    The causal test is what keeps this from becoming "ignore every dangling step". A dangling step
    counts as derived only when some step carrying an unknown reference *could* have been its
    consumer, and a step may only consume steps listed before it (:class:`~plan_repair.schema.plan
    .AgentPlan`), so the broken reference has to sit later in the plan. A step dangling for its own
    reasons — the spare copy a duplicate leaves behind, with no unknown reference anywhere — fails
    that test and stays in the mask, because nothing else is going to bring it back.

    The validator makes the same kind of judgement in the other direction: it skips the ordering
    check once it finds a cycle, because an ordering violation there is a derived symptom. This is
    that reasoning applied to the mask, and it is applied *here* rather than there on purpose —
    the validator is the instrument, and it goes on reporting everything it sees.
    """
    position = {step.id: index for index, step in enumerate(plan.steps)}
    consumers = [
        position[step_id]
        for error in validation.errors_of_type(UNKNOWN_DEPENDENCY)
        for step_id in error.step_ids
        if step_id in position
    ]
    if not consumers:
        return set()
    latest = max(consumers)
    return {
        step_id
        for error in validation.errors_of_type(DANGLING_STEP)
        for step_id in error.step_ids
        if position.get(step_id, latest) < latest
    }


def paths_to_mask(validation: PlanValidationResult, plan: AgentPlan) -> set[str]:
    """The findings a mask should act on — everything except a derived dangling step.

    Dropped per *error* rather than per path, so a step that is dangling **and** something else
    keeps the mask its other finding earns it. Only the dangling report goes.
    """
    derived = derived_dangling_step_ids(validation, plan)
    return {
        path
        for error in validation.errors
        if not (error.type == DANGLING_STEP and set(error.step_ids) <= derived)
        for path in error.paths
    }


def mask_spec_from_paths(sequence: PlanSequence, plan: AgentPlan, paths: Iterable[str]) -> MaskSpec:
    """Build a mask from validator paths, narrowing to a field wherever a path names one.

    A path like ``$.steps[?join].tool`` masks that field alone; ``$.steps[?join]`` masks the whole
    step, which is what the errors that have no field — ordering, duplicates, dangling steps —
    still produce. Paths pointing outside the plan (the stop condition, a task requirement) name
    nothing a mask can cover and are ignored here.

    Two paths for one step can disagree about how much to take: if anything asks for the whole
    step, the whole step is masked, because a narrower mask alongside it would be meaningless.
    """
    by_id = {step.id: step for step in plan.steps}
    spans_by_id = {span.step_id: span for span in sequence.spans}

    whole: set[str] = set()
    fields: dict[str, set[str]] = {}
    unmaskable: set[str] = set()
    for path in paths:
        parsed = parse_path(path)
        if parsed.step_id is None:
            continue
        if parsed.step_id not in spans_by_id:
            unmaskable.add(parsed.step_id)
            continue
        if parsed.field is None:
            whole.add(parsed.step_id)
        else:
            fields.setdefault(parsed.step_id, set()).add(parsed.field)

    spans: list[StepSpan] = []
    for span in sequence.spans:  # plan order, so the mask reads in the order of the sequence
        step_id = span.step_id
        if step_id in whole:
            spans.append(span)
            continue
        for field in sorted(fields.get(step_id, ())):
            boundaries = field_spans(by_id[step_id], span.start).get(field)
            if boundaries is None:
                continue
            spans.append(
                StepSpan(step_id=step_id, start=boundaries[0], end=boundaries[1], field=field)
            )

    touched = whole | set(fields)
    return MaskSpec(
        masked_step_ids=[span.step_id for span in sequence.spans if span.step_id in touched],
        preserved_step_ids=[span.step_id for span in sequence.spans if span.step_id not in touched],
        spans=spans,
        unmaskable_step_ids=sorted(unmaskable),
    )


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
    sequence: PlanSequence,
    spec: MaskSpec,
    placeholder: str = DEFAULT_PLACEHOLDER,
    plan: AgentPlan | None = None,
) -> str:
    """The sequence with every masked step replaced by ``placeholder``.

    ``placeholder`` is a parameter because the mask token belongs to whichever model fills it in.
    """
    return _rebuild(sequence, {span.key: placeholder for span in spec.spans}, spec, plan)


def fill_masked(
    sequence: PlanSequence,
    spec: MaskSpec,
    replacements: Mapping[str, str | None],
    plan: AgentPlan | None = None,
) -> str:
    """Put ``replacements`` into the masked spans and return the resulting sequence.

    A replacement of ``None`` drops that step, which is how a regeneration expresses "this step
    should not exist". Steps outside the mask are copied over verbatim — that is the guarantee
    the whole approach rests on, so it is enforced here rather than asked for.
    """
    allowed = {span.key for span in spec.spans}
    unknown = set(replacements) - allowed
    if unknown:
        raise ValueError(f"cannot fill steps that are not masked: {sorted(unknown)}")
    normalised = {
        key: None if text is None else normalise_filling(text) for key, text in replacements.items()
    }
    return _rebuild(sequence, normalised, spec, plan)


def normalise_filling(text: str) -> str:
    """Drop the separators a filling arrives with; the template owns them.

    Two masked steps that sit next to each other share the comma and newline between them, and a
    token holding that separator belongs to the mask — so a regeneration hands back the step
    *and* the punctuation around it. Reassembly then adds its own separator and the result reads
    ``},,``, which is not valid JSON however good the step was.

    The rule this settles is a division of labour: a filling contributes the body of one step,
    and where steps meet is decided by the sequence it is being put back into. Only the outermost
    characters are touched, so commas inside a step — in ``input_from``, in ``arguments`` — are
    left exactly as the model wrote them.

    This is not repairing malformed output. Nothing is inserted, nothing is re-punctuated, and a
    filling that is not a step still fails to parse.
    """
    return text.strip(" \t\r\n,")


def _rebuild(
    sequence: PlanSequence,
    replacements: Mapping[str, str | None],
    spec: MaskSpec | None = None,
    plan: AgentPlan | None = None,
) -> str:
    """Put the replacements back and return the sequence.

    A whole-step replacement swaps the line. A field replacement rewrites one value inside the
    line and copies every other field across character for character — which is the point of the
    whole exercise: ``produces`` is not in the mask, so it cannot come back different.
    """
    field_spans_by_step = _field_replacements(replacements, spec)
    by_id = {step.id: step for step in (plan.steps if plan else [])}

    kept: list[str] = []
    for span, original in zip(sequence.spans, sequence.step_texts, strict=True):
        if span.step_id in replacements:  # the whole step was masked
            replacement = replacements[span.step_id]
            if replacement is None:
                continue
            kept.append(replacement)
        elif span.step_id in field_spans_by_step:
            step = by_id.get(span.step_id)
            if step is None:
                kept.append(original)
                continue
            kept.append(_rewrite_fields(step, field_spans_by_step[span.step_id]))
        else:
            kept.append(original)
    text, _ = _assemble(sequence.header, kept, sequence.footer, [])
    return text


def _field_replacements(
    replacements: Mapping[str, str | None], spec: MaskSpec | None
) -> dict[str, dict[str, str]]:
    """Group field-keyed replacements by step, dropping the ones with nothing to put."""
    grouped: dict[str, dict[str, str]] = {}
    for span in spec.spans if spec else []:
        if span.field is None or span.key not in replacements:
            continue
        value = replacements[span.key]
        if value is None:
            continue
        grouped.setdefault(span.step_id, {})[span.field] = value
    return grouped


def _rewrite_fields(step: Step, values: Mapping[str, str]) -> str:
    """Render the step again with the named fields replaced by the given text.

    The separators are the template's, not the filling's — the same division of labour that
    settled the doubled comma between steps, applied one level down. A filling that arrives with
    the surrounding ``,`` or spaces attached contributes only its value.
    """
    parts = []
    for name, value in step.model_dump().items():
        rendered = (
            normalise_filling(values[name])
            if name in values
            else json.dumps(value, ensure_ascii=False)
        )
        parts.append(f"{json.dumps(name, ensure_ascii=False)}: {rendered}")
    return "{" + ", ".join(parts) + "}"


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
    "derived_dangling_step_ids",
    "field_spans",
    "fill_masked",
    "mask_spec",
    "mask_spec_from_paths",
    "normalise_filling",
    "paths_to_mask",
    "plan_to_sequence",
    "render_masked",
    "sequence_to_plan",
]
