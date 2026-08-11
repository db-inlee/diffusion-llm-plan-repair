"""Completing a filled tool name — the one place the pipeline improves on what the model wrote.

Everything up to stage C left the repair untouched on purpose: the question was what a diffusion
model can do, and a pipeline that quietly fixes the answer measures the pipeline. Stage D changes
the goal from measuring to working, and this module is the first step of it. It is still not a
change to the repair — nothing here runs, prompts or re-decodes a model; it reads a value the
model has already produced.

**What it is for.** C-3 showed the failure exactly: the mask is a token longer than the answer
needs, the model writes the right tool and then has a cell left over, and it fills that cell.
``join`` comes back as ``join_db``, ``dedupe`` as ``deduplicate``. The answer is present, at the
front, with surplus attached.

**Where the line is.** A snap that always picked the nearest valid name would let a model that
wrote nonsense pass, and the measurement would stop meaning anything. So the rule is a *prefix
completion*: the value has to reproduce a valid name from the front — all of it, or all but the
last character — and no other valid name may do as well. ``merge_join`` is therefore left broken
even though ``join`` is the right answer and sits inside it. Finding a name in the middle of a
value is a search, and a search is what turns a snap into a rubber stamp.

**Why 0.8 is not a number fitted to four measurements.** The ratio between any two valid tool
names in either domain is at most 0.71, so no valid name can complete another one; the floor sits
above the vocabulary's own confusability with room on both sides of it. That property is pinned
by a test rather than asserted here, because a tool added later could break it.

**What this does not know.** Whether the completed name is the *right* tool. A value that
completes a valid name it was never meant to is snapped and the plan then reads as valid — the
validator cannot object to a tool that exists. No such case has been observed, and none is ruled
out; the snap is recorded per repair (:class:`ToolSnap`) so that a result can be read as the
model's or as the snap's.
"""

import json
from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ConfigDict

from plan_repair.repair.hint import TOOL_FIELD
from plan_repair.repair.remask import MaskSpec, normalise_filling
from plan_repair.schema.plan import AgentPlan

# How much of a valid name a value has to reproduce from the front. Above the vocabulary's own
# maximum self-similarity (0.71), and below the closest observed completion (0.83).
SNAP_RATIO_FLOOR = 0.8

SNAPPED = "snapped"
ALREADY_VALID = "already a valid tool"
BELOW_FLOOR = "no valid tool is completed clearly enough"
AMBIGUOUS = "more than one valid tool is completed equally well"
NOT_A_STRING = "the filling is not a JSON string"

# The dependency field, and the outcomes of cleaning one.
DEPENDENCY_FIELD = "input_from"

CLEANED = "cleaned"
UNCHANGED = "every reference reads as one already"
REFUSED = "nothing was resolvable without guessing"
NOT_A_LIST = "the filling is not a JSON list of strings"
AMBIGUOUS_TAG = "more than one step produces that tag"
UNRESOLVED = "no step this one may depend on produces that"


class ToolSnap(BaseModel):
    """What the snap did to one filled tool value, and why.

    Recorded whether or not it fired. A repair that came out solved has to be readable as the
    model's answer or as this module's, and a refusal has to be readable too — otherwise the cost
    of being conservative is invisible.
    """

    model_config = ConfigDict(extra="forbid")

    original: str
    snapped: str | None
    ratio: float
    runner_up: float
    reason: str

    @property
    def fired(self) -> bool:
        return self.snapped is not None

    @property
    def replacement(self) -> str | None:
        """The value to put back into the plan, rendered as JSON — the template owns nothing here.

        The masked span covers the quotes as well as the name (``field_spans`` measures
        ``json.dumps`` of the value), so a replacement that is not itself quoted would not parse.
        """
        return None if self.snapped is None else json.dumps(self.snapped, ensure_ascii=False)


def prefix_ratio(value: str, tool: str) -> float:
    """How much of ``tool`` the front of ``value`` reproduces, as a fraction of ``tool``.

    Characters rather than tokens. The surplus the model attaches is a token, but where the
    tokenizer splits a name has nothing to do with whether the name was reproduced: ``dedupe`` is
    ``ded``+``upe`` and ``deduplicate`` is ``ded``+``u``+``plicate``, so a token-prefix rule sees
    only ``ded`` and misses a completion that is plainly one at the character level.
    """
    if not tool:
        return 0.0
    shared = 0
    for left, right in zip(value, tool, strict=False):
        if left != right:
            break
        shared += 1
    return shared / len(tool)


def snap_tool_value(text: str, tools: Iterable[str], floor: float = SNAP_RATIO_FLOOR) -> ToolSnap:
    """Decide what a filled tool value should become, and record why.

    Always returns a decision. ``snapped is None`` means the value stands as the model wrote it,
    which is the outcome for a value that is already valid, one that completes nothing clearly
    enough, one that completes two names equally well, and one that is not a string at all.
    """
    value = _as_string(text)
    if value is None:
        return ToolSnap(
            original=text.strip(), snapped=None, ratio=0.0, runner_up=0.0, reason=NOT_A_STRING
        )

    names = set(tools)
    if value in names:
        return ToolSnap(
            original=value, snapped=None, ratio=1.0, runner_up=0.0, reason=ALREADY_VALID
        )

    ranked = sorted(
        ((prefix_ratio(value, name), name) for name in names), key=lambda pair: (-pair[0], pair[1])
    )
    best_ratio, best_name = ranked[0] if ranked else (0.0, "")
    runner_up = ranked[1][0] if len(ranked) > 1 else 0.0

    if best_ratio < floor:
        reason, snapped = BELOW_FLOOR, None
    elif best_ratio <= runner_up:  # a tie is the case the snap must not decide
        reason, snapped = AMBIGUOUS, None
    else:
        reason, snapped = SNAPPED, best_name
    return ToolSnap(
        original=value, snapped=snapped, ratio=best_ratio, runner_up=runner_up, reason=reason
    )


def snap_tool_fillings(
    filling: Mapping[str, str | None],
    spec: MaskSpec,
    tools: Iterable[str],
    floor: float = SNAP_RATIO_FLOOR,
) -> tuple[dict[str, str | None], dict[str, ToolSnap]]:
    """Apply the snap to the tool fields of ``filling``, returning the fillings and the record.

    Which spans are tool fields is read off the mask rather than parsed out of the keys, and it is
    the same predicate the valid-tool hint fires on (:mod:`plan_repair.repair.hint`) — one place
    decides what "this repair is about a tool" means.

    A filling of ``None`` is a step being dropped, and a whole-step span is not a tool field.
    Neither is touched, and neither appears in the record.
    """
    snapped = dict(filling)
    record: dict[str, ToolSnap] = {}
    for span in spec.spans:
        if span.field != TOOL_FIELD:
            continue
        text = filling.get(span.key)
        if text is None:
            continue
        decision = snap_tool_value(text, tools, floor)
        record[span.key] = decision
        if decision.replacement is not None:
            snapped[span.key] = decision.replacement
    return snapped, record


class DependencySnap(BaseModel):
    """What the cleaning did to one filled ``input_from``, and what it declined to do.

    ``refused`` is the half that has to be read: an element left alone is a repair that did not
    happen, and the reason separates "the model wrote something nobody produces" from "two steps
    claim that tag and choosing would be inventing an answer".
    """

    model_config = ConfigDict(extra="forbid")

    original: list[str]
    snapped: list[str] | None
    dropped: list[str]
    resolved: dict[str, str]
    refused: dict[str, str]
    reason: str

    @property
    def fired(self) -> bool:
        return self.snapped is not None

    @property
    def replacement(self) -> str | None:
        """The list to put back, rendered as JSON — the masked span covers the brackets too."""
        return None if self.snapped is None else json.dumps(self.snapped, ensure_ascii=False)


def snap_dependency_value(text: str, plan: AgentPlan, step_id: str) -> DependencySnap:
    """Read a regenerated dependency list back as references, as far as it can be read.

    Three things happen to an element, in this order:

    * an empty string names nothing and is dropped — that is the surplus mask cell of Ticket C-3
      arriving in a list rather than inside a name;
    * a step id is a reference already and is kept;
    * anything else is looked up among the ``produces`` tags of the steps this one could legally
      depend on, and becomes that step's id **when exactly one step qualifies.**

    The legality clause is what stops the cleaning from trading one error for another. A step may
    only consume steps listed before it, so a tag produced by ``step_id`` itself — or by anything
    after it — is not a candidate; resolving those would answer a broken reference with a cycle or
    an ordering violation. Ambiguity is refused for the plainer reason that domain B has three
    tags two steps each claim, and there is no reading of the model's answer that says which.
    """
    value = _as_list(text)
    if value is None:
        return DependencySnap(
            original=[], snapped=None, dropped=[], resolved={}, refused={}, reason=NOT_A_LIST
        )

    step_ids = {step.id for step in plan.steps}
    producers = _producers_before(plan, step_id)

    cleaned: list[str] = []
    dropped: list[str] = []
    resolved: dict[str, str] = {}
    refused: dict[str, str] = {}
    for element in value:
        if not element.strip():
            dropped.append(element)
            continue
        if element in step_ids:
            cleaned.append(element)
            continue
        owners = producers.get(element, [])
        if len(owners) == 1:
            cleaned.append(owners[0])
            resolved[element] = owners[0]
        else:
            cleaned.append(element)
            refused[element] = AMBIGUOUS_TAG if owners else UNRESOLVED

    # "Nothing to do" and "nothing I was willing to do" are different outcomes, and a record that
    # called them both unchanged would hide every case conservatism gave up on.
    changed = cleaned != value
    reason = CLEANED if changed else REFUSED if refused else UNCHANGED
    return DependencySnap(
        original=value,
        snapped=cleaned if changed else None,
        dropped=dropped,
        resolved=resolved,
        refused=refused,
        reason=reason,
    )


def snap_dependency_fillings(
    filling: Mapping[str, str | None], spec: MaskSpec, plan: AgentPlan
) -> tuple[dict[str, str | None], dict[str, DependencySnap]]:
    """Apply the cleaning to the dependency fields of ``filling``.

    Which spans those are is read off the mask, the same way the tool snap reads its own — a
    filling of ``None`` is a step being dropped and a whole-step span is not a field, so neither
    is touched or recorded.
    """
    cleaned = dict(filling)
    record: dict[str, DependencySnap] = {}
    for span in spec.spans:
        if span.field != DEPENDENCY_FIELD:
            continue
        text = filling.get(span.key)
        if text is None:
            continue
        decision = snap_dependency_value(text, plan, span.step_id)
        record[span.key] = decision
        if decision.replacement is not None:
            cleaned[span.key] = decision.replacement
    return cleaned, record


def _producers_before(plan: AgentPlan, step_id: str) -> dict[str, list[str]]:
    """Which steps claim each ``produces`` tag, among those ``step_id`` may depend on."""
    producers: dict[str, list[str]] = {}
    for step in plan.steps:
        if step.id == step_id:
            break  # a step may only consume steps listed before it
        for tag in step.produces:
            producers.setdefault(tag, []).append(step.id)
    return producers


def _as_string(text: str) -> str | None:
    """The filling as the string it stands for, or ``None`` if it is not one.

    The separators come off first, because they belong to the reassembly rather than to the value
    (:func:`plan_repair.repair.remask.normalise_filling`) — a filling arrives as ``' "join_db",'``
    and the value in it is ``join_db``.
    """
    try:
        value = json.loads(normalise_filling(text))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, str) else None


def _as_list(text: str) -> list[str] | None:
    """The filling as the list of strings it stands for, or ``None`` if it is not one."""
    try:
        value = json.loads(normalise_filling(text))
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return [str(item) for item in value]


__all__ = [
    "ALREADY_VALID",
    "AMBIGUOUS",
    "AMBIGUOUS_TAG",
    "BELOW_FLOOR",
    "CLEANED",
    "DEPENDENCY_FIELD",
    "NOT_A_LIST",
    "NOT_A_STRING",
    "REFUSED",
    "SNAPPED",
    "SNAP_RATIO_FLOOR",
    "UNCHANGED",
    "UNRESOLVED",
    "DependencySnap",
    "ToolSnap",
    "prefix_ratio",
    "snap_dependency_fillings",
    "snap_dependency_value",
    "snap_tool_fillings",
    "snap_tool_value",
]
