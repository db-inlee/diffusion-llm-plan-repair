"""What the model is told besides the plan — the fairness gap, closed.

Until now a diffusion repairer was handed a masked plan and nothing else: no instruction, no
system message, no vocabulary. The autoregressive repairer it is being compared against gets the
task JSON, which lists every tool the agent is allowed to call, and is told in so many words to
use only those. That is not a difference between architectures; it is a difference in what the two
were shown, and it sat in the comparison unexamined.

**Why it matters here specifically.** A tool name usually appears once in a plan. Mask that field
and the correct string is nowhere in the model's input — it has to be recalled rather than copied,
and what came back instead was semantically right and lexically invented (``dedupe`` as
``deduplicate``, ``join`` as ``merge_join``). Names the validator then rejects as unknown tools.

**What this does not claim.** That the list will fix it. The same measurement showed the model
declining to copy the correct string from the adjacent ``id`` field, which is evidence that the
problem may not be one of missing information at all. Supplying the list makes the comparison
fair; whether a fair comparison changes the outcome is what the re-measurement is for, and both
answers are worth having.

The hint is emitted only when the mask has narrowed to a ``tool`` field — the wrong-tool case.
Every other error type keeps the input it had, so their results stay comparable across tickets.
"""

from plan_repair.repair.remask import MaskSpec
from plan_repair.schema.task import AgentTask

TOOL_FIELD = "tool"
HINT_LABEL = "valid tools: "


def valid_tool_hint(task: AgentTask, mask: MaskSpec) -> str | None:
    """The tool vocabulary as one line of text, or ``None`` when it does not apply.

    Sorted, because ``tool_names`` is a set and a run that hands the model a different string
    each time is not a run that can be repeated.

    One line of prose rather than a JSON fragment: fewer tokens, and it reads as something said
    to the model rather than more data to be continued — the sequence it is prepended to is
    already JSON, and a second JSON object in front of the first would invite the model to treat
    the pair as one document.
    """
    if not any(span.field == TOOL_FIELD for span in mask.spans):
        return None
    names = sorted(task.tool_names())
    if not names:
        return None
    return f"{HINT_LABEL}{', '.join(names)}\n"


__all__ = ["HINT_LABEL", "TOOL_FIELD", "valid_tool_hint"]
