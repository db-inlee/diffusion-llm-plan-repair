"""Injector contracts: immutability, id preservation and metadata accuracy."""

import pytest

from plan_repair.canonical import canonicalize
from plan_repair.corruption import (
    CYCLE_MODE,
    UNKNOWN_MODE,
    inject_broken_dependency,
    inject_step_deletion,
)
from plan_repair.data import load_reference_plan
from plan_repair.schema import BROKEN_DEPENDENCY, STEP_DELETION
from plan_repair.validation import input_from_path


def step(plan, step_id):
    return next(s for s in plan.steps if s.id == step_id)


def test_broken_dependency_leaves_the_original_plan_untouched():
    plan = load_reference_plan()
    before = canonicalize(plan)[0]

    inject_broken_dependency(plan, step_id="enrich", mode=UNKNOWN_MODE)

    assert canonicalize(plan)[0] == before


def test_step_deletion_leaves_the_original_plan_untouched():
    plan = load_reference_plan()
    before = canonicalize(plan)[0]

    inject_step_deletion(plan, step_id="join")

    assert canonicalize(plan)[0] == before
    assert len(plan.steps) == 20


def test_broken_dependency_unknown_mode_metadata():
    plan = load_reference_plan()
    result = inject_broken_dependency(plan, step_id="enrich", mode=UNKNOWN_MODE)
    injected = result.injected[0]

    assert injected.corruption_type == BROKEN_DEPENDENCY
    assert injected.damaged_step_ids == ["enrich"]
    assert injected.damaged_paths == [input_from_path("enrich")]
    assert injected.detail["mode"] == UNKNOWN_MODE
    assert injected.detail["original_input_from"] == ["join", "l_api"]
    assert injected.detail["removed_dep"] == "l_api"
    assert step(result.broken_plan, "enrich").input_from == ["join", "l_api_x"]


def test_broken_dependency_can_target_a_specific_edge():
    plan = load_reference_plan()
    result = inject_broken_dependency(plan, step_id="enrich", mode=UNKNOWN_MODE, dep="join")

    assert step(result.broken_plan, "enrich").input_from == ["join_x", "l_api"]


def test_broken_dependency_cycle_mode_metadata():
    plan = load_reference_plan()
    result = inject_broken_dependency(plan, step_id="l_csv", mode=CYCLE_MODE, cycle_with="report")
    injected = result.injected[0]

    assert injected.damaged_step_ids == ["l_csv"]
    assert injected.damaged_paths == [input_from_path("l_csv")]
    assert injected.detail["added_dep"] == "report"
    assert step(result.broken_plan, "l_csv").input_from == ["report"]


def test_cycle_mode_refuses_an_edge_that_would_not_close_a_cycle():
    plan = load_reference_plan()
    with pytest.raises(ValueError, match="would not close a cycle"):
        inject_broken_dependency(plan, step_id="report", mode=CYCLE_MODE, cycle_with="l_csv")


def test_unknown_mode_refuses_a_step_without_dependencies():
    plan = load_reference_plan()
    with pytest.raises(ValueError, match="no dependency edge"):
        inject_broken_dependency(plan, step_id="l_csv", mode=UNKNOWN_MODE)


def test_injectors_reject_unknown_step_ids():
    plan = load_reference_plan()
    with pytest.raises(ValueError, match="no such step"):
        inject_step_deletion(plan, step_id="nope")


def test_step_deletion_metadata_and_preserved_ids():
    plan = load_reference_plan()
    result = inject_step_deletion(plan, step_id="join")
    injected = result.injected[0]

    assert injected.corruption_type == STEP_DELETION
    assert injected.damaged_step_ids == ["enrich"]
    assert injected.damaged_paths == [input_from_path("enrich")]
    assert injected.detail["deleted_step_id"] == "join"
    assert injected.detail["deleted_step"] == {
        "id": "join",
        "tool": "join",
        "arguments": {},
        "input_from": ["n_csv", "n_db"],
    }
    assert len(result.broken_plan.steps) == 19
    assert "join" not in {s.id for s in result.broken_plan.steps}
    assert set(result.preserved_step_ids) == {s.id for s in result.broken_plan.steps} - {"enrich"}
    assert len(result.preserved_step_ids) == 18


def test_step_ids_of_surviving_steps_are_never_rewritten():
    plan = load_reference_plan()
    original_ids = [s.id for s in plan.steps]

    deletion = inject_step_deletion(plan, step_id="co")
    dependency = inject_broken_dependency(plan, step_id="enrich", mode=UNKNOWN_MODE)

    assert [s.id for s in deletion.broken_plan.steps] == [i for i in original_ids if i != "co"]
    assert [s.id for s in dependency.broken_plan.steps] == original_ids
