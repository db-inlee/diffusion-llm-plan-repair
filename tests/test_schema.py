"""Schema contract: parsing, defaults and rejection of malformed payloads."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from plan_repair.data import load_reference
from plan_repair.schema import (
    BROKEN_DEPENDENCY,
    STEP_DELETION,
    AgentPlan,
    AgentTask,
    CorruptionResult,
    InjectedError,
    Step,
    ToolSpec,
)


def test_step_defaults():
    step = Step(id="s1", tool="load_csv")
    assert step.arguments == {}
    assert step.input_from == []


def test_plan_defaults_stop_condition_to_none():
    plan = AgentPlan(goal="g", steps=[Step(id="s1", tool="load_csv")])
    assert plan.stop_condition is None


def test_reference_payload_parses():
    task, plan = load_reference()
    assert task.task_id == "data_pipeline_B"
    assert len(task.available_tools) == 17
    assert len(plan.steps) == 20
    assert [step.id for step in plan.steps][:3] == ["l_csv", "l_db", "l_api"]


def test_task_tool_names():
    task = AgentTask(
        task_id="t",
        user_query="q",
        available_tools=[ToolSpec(name="a"), ToolSpec(name="b")],
    )
    assert task.tool_names() == {"a", "b"}
    assert task.required_evidence == []
    assert task.max_tool_calls is None


def test_missing_required_field_is_rejected():
    with pytest.raises(PydanticValidationError):
        AgentPlan.model_validate({"goal": "g", "steps": [{"id": "s1"}]})


def test_wrong_type_is_rejected():
    with pytest.raises(PydanticValidationError):
        AgentPlan.model_validate(
            {"goal": "g", "steps": [{"id": "s1", "tool": "t", "input_from": "s0"}]}
        )


def test_free_text_fields_are_forbidden():
    """No description-like field may sneak into the contract (surface edits stay unexpressible)."""
    with pytest.raises(PydanticValidationError):
        Step.model_validate({"id": "s1", "tool": "t", "description": "does something"})


def test_corruption_metadata_roundtrip():
    plan = AgentPlan(goal="g", steps=[Step(id="s1", tool="load_csv")])
    injected = InjectedError(
        corruption_type=BROKEN_DEPENDENCY,
        damaged_step_ids=["s1"],
        damaged_paths=["$.steps[?s1].input_from"],
    )
    result = CorruptionResult(broken_plan=plan, injected=[injected], preserved_step_ids=[])
    restored = CorruptionResult.model_validate(result.model_dump())
    assert restored == result
    assert injected.detail == {}
    assert STEP_DELETION == "step_deletion"
