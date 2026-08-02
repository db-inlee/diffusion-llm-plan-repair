"""Autoregressive repairers — the honest competitor.

Two modes, because they answer different questions:

* :class:`ARFullRepairer` rewrites the plan from scratch. It is the reference point for what
  regeneration costs: every healthy step passes through the model again and may come back
  changed.
* :class:`ARLocalRepairer` is told where the errors are and asked to touch nothing else. This is
  the repairer a diffusion model actually has to beat — not full regeneration, which is the easy
  opponent.

**Fairness.** The local mode is given exactly what the validator found: the ids and paths its
errors point at. That is the same region a diffusion repairer will later be handed, so the two
are answering the same question with the same hint. Error *types* and messages are deliberately
withheld for now, so that the decision to reveal them can be made once, for both.

**Failure is an outcome, not a crash.** A model can return prose, broken JSON, or a plan that
violates the schema. None of that is repaired into shape: the broken plan comes back unchanged,
which scores exactly as what it is — a repair that did not happen — and the reason is recorded in
:attr:`failures` for reporting. How often that occurs is part of what an AR repairer is worth.

**Not deterministic.** Even at temperature zero the same plan can come back different. The
repeat-and-compare checks that hold for the rule-based repairer do not apply here.
"""

from pydantic import BaseModel, ConfigDict

from plan_repair.repair.llm_client import LLMClient, LLMError
from plan_repair.repair.plan_io import PlanParseError, parse_plan, plan_to_json, task_to_json
from plan_repair.schema.plan import AgentPlan
from plan_repair.schema.task import AgentTask
from plan_repair.validation.models import PlanValidationResult

PARSE_FAILURE = "parse"
API_FAILURE = "api"

SYSTEM_PROMPT = (
    "You repair execution plans for a tool-using agent. "
    "Answer with a single JSON object for the plan and nothing else: no prose, no explanation. "
    "The object has exactly the keys 'goal' (string), 'steps' (array) and 'stop_condition' "
    "(string or null). Each step has exactly the keys 'id' (string), 'tool' (string), "
    "'arguments' (object), 'input_from' (array of step ids) and 'produces' (array of strings). "
    "Use only tools listed as available. A step may only consume steps that appear before it."
)


class RepairFailure(BaseModel):
    """Why one repair attempt produced nothing."""

    model_config = ConfigDict(extra="forbid")

    repairer: str
    kind: str
    detail: str


class _ARRepairer:
    """Shared call-and-parse loop; the modes differ only in the prompt they build."""

    name = "ar"

    def __init__(self, client: LLMClient) -> None:
        self._client = client
        self.failures: list[RepairFailure] = []

    def repair(
        self,
        broken_plan: AgentPlan,
        validation: PlanValidationResult,
        task: AgentTask,
    ) -> AgentPlan:
        try:
            answer = self._client.complete(
                system=SYSTEM_PROMPT, user=self._prompt(broken_plan, validation, task)
            )
        except LLMError as exc:
            return self._give_up(broken_plan, API_FAILURE, str(exc))
        try:
            return parse_plan(answer)
        except PlanParseError as exc:
            return self._give_up(broken_plan, PARSE_FAILURE, str(exc))

    def _prompt(
        self, broken_plan: AgentPlan, validation: PlanValidationResult, task: AgentTask
    ) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def _give_up(self, broken_plan: AgentPlan, kind: str, detail: str) -> AgentPlan:
        self.failures.append(RepairFailure(repairer=self.name, kind=kind, detail=detail))
        return broken_plan.model_copy(deep=True)


class ARFullRepairer(_ARRepairer):
    """Regenerates the whole plan."""

    name = "ar_full"

    def _prompt(
        self, broken_plan: AgentPlan, validation: PlanValidationResult, task: AgentTask
    ) -> str:
        return (
            f"Task:\n{task_to_json(task)}\n\n"
            f"This plan is broken:\n{plan_to_json(broken_plan)}\n\n"
            "Write a correct plan for this task from scratch. "
            "Return the complete plan as JSON."
        )


class ARLocalRepairer(_ARRepairer):
    """Edits only the region the validator pointed at."""

    name = "ar_local"

    def _prompt(
        self, broken_plan: AgentPlan, validation: PlanValidationResult, task: AgentTask
    ) -> str:
        return (
            f"Task:\n{task_to_json(task)}\n\n"
            f"This plan is broken:\n{plan_to_json(broken_plan)}\n\n"
            f"{_error_region(validation)}\n\n"
            "Fix only what is wrong at those locations. Leave every other step exactly as it is: "
            "same id, same tool, same arguments, same input_from, same produces, same position. "
            "Return the complete plan as JSON."
        )


def _error_region(validation: PlanValidationResult) -> str:
    """The locations the validator flagged — ids and paths, without saying what is wrong."""
    step_ids = sorted(validation.detected_step_ids())
    paths = sorted(validation.detected_paths())
    if not step_ids and not paths:
        return "The validator flagged no location."
    lines = ["The validator flagged these locations:"]
    if step_ids:
        lines.append(f"  steps: {', '.join(step_ids)}")
    if paths:
        lines.append("  paths:")
        lines.extend(f"    {path}" for path in paths)
    return "\n".join(lines)


__all__ = [
    "API_FAILURE",
    "PARSE_FAILURE",
    "SYSTEM_PROMPT",
    "ARFullRepairer",
    "ARLocalRepairer",
    "RepairFailure",
]
