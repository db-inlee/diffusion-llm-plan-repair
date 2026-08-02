"""Deterministic execution environment.

No data flows: a tool call is judged only on whether it *could* have been made. From the scenario
document's rules, a step succeeds when

1. its tool is offered by the task,
2. its arguments are usable, and
3. every step it consumes succeeded.

and a plan succeeds when every step succeeded and the stop condition was reached.

Two simplifications are deliberate and worth stating plainly:

* **Stop condition.** Reaching it means the terminal step (``plan.steps[-1]``, the structural
  definition from Ticket 002) succeeded. The natural-language text of ``stop_condition`` is never
  interpreted — no part of this project judges what it means.
* **Arguments.** The task contract names the tools it offers but not the arguments they require,
  so there is nothing to check a step's arguments against. What is checked is that the values a
  step does carry are usable: a blank string is a placeholder nobody could call a tool with.

Execution is dependency-driven, so the *list order* of steps does not affect the outcome; a plan
whose steps are ordered badly but wired correctly still runs. That is a real difference from the
static ordering check, and it is reported by :mod:`plan_repair.runtime.agreement` rather than
hidden.

Determinism: outcomes are reported in plan order and no step's result depends on iteration order,
so the same plan always yields the same result.
"""

from pydantic import BaseModel, ConfigDict

from plan_repair.schema.plan import AgentPlan, Step
from plan_repair.schema.task import AgentTask

UNKNOWN_TOOL = "unknown_tool"
UNUSABLE_ARGUMENT = "unusable_argument"
FAILED_DEPENDENCY = "failed_dependency"
UNKNOWN_DEPENDENCY = "unknown_dependency"
UNRESOLVABLE = "unresolvable_dependency"


class StepOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    succeeded: bool
    reason: str | None = None


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    succeeded: bool
    stop_condition_reached: bool
    outcomes: list[StepOutcome]

    def failed_step_ids(self) -> list[str]:
        return [outcome.step_id for outcome in self.outcomes if not outcome.succeeded]

    def reasons(self) -> set[str]:
        return {outcome.reason for outcome in self.outcomes if outcome.reason is not None}


def run_plan(plan: AgentPlan, task: AgentTask) -> RunResult:
    """Execute ``plan`` against ``task`` and report what succeeded."""
    available = task.tool_names()
    by_id: dict[str, list[int]] = {}
    for index, step in enumerate(plan.steps):
        by_id.setdefault(step.id, []).append(index)

    decided: dict[int, bool] = {}
    reasons: dict[int, str] = {}
    pending = list(range(len(plan.steps)))

    def state_of(step_id: str) -> bool | None:
        """Whether every step carrying ``step_id`` succeeded; ``None`` while undecided."""
        indices = by_id.get(step_id)
        if indices is None:
            return False  # a reference to a step that does not exist can never be satisfied
        if any(index not in decided for index in indices):
            return None
        return all(decided[index] for index in indices)

    while pending:
        progressed = False
        for index in list(pending):
            step = plan.steps[index]
            local = _local_failure(step, available)
            if local is not None:
                _settle(index, False, local, decided, reasons, pending)
                progressed = True
                continue
            if any(dep not in by_id for dep in step.input_from):
                _settle(index, False, UNKNOWN_DEPENDENCY, decided, reasons, pending)
                progressed = True
                continue
            states = [state_of(dep) for dep in step.input_from]
            if None in states:
                continue  # a predecessor is still undecided; revisit on the next pass
            if all(states):
                _settle(index, True, None, decided, reasons, pending)
            else:
                _settle(index, False, FAILED_DEPENDENCY, decided, reasons, pending)
            progressed = True
        if not progressed:
            break

    # Whatever is left waits on itself: a dependency cycle, so it never runs.
    for index in pending:
        decided[index] = False
        reasons[index] = UNRESOLVABLE

    outcomes = [
        StepOutcome(step_id=step.id, succeeded=decided[index], reason=reasons.get(index))
        for index, step in enumerate(plan.steps)
    ]
    reached = bool(plan.steps) and decided[len(plan.steps) - 1]
    return RunResult(
        succeeded=all(outcome.succeeded for outcome in outcomes) and reached,
        stop_condition_reached=reached,
        outcomes=outcomes,
    )


def _local_failure(step: Step, available: set[str]) -> str | None:
    """Failures a step carries on its own, independent of any predecessor."""
    if step.tool not in available:
        return UNKNOWN_TOOL
    if any(isinstance(value, str) and not value.strip() for value in step.arguments.values()):
        return UNUSABLE_ARGUMENT
    return None


def _settle(
    index: int,
    succeeded: bool,
    reason: str | None,
    decided: dict[int, bool],
    reasons: dict[int, str],
    pending: list[int],
) -> None:
    decided[index] = succeeded
    if reason is not None:
        reasons[index] = reason
    pending.remove(index)


__all__ = ["RunResult", "StepOutcome", "run_plan"]
