"""Canonicalization determinism and its two-sided ordering rule."""

from plan_repair.canonical import canonicalize, step_hash
from plan_repair.data import load_reference_plan
from plan_repair.schema import AgentPlan, Step


def test_canonicalize_is_byte_stable_across_runs():
    plan = load_reference_plan()
    first_json, first_hashes = canonicalize(plan)
    second_json, second_hashes = canonicalize(load_reference_plan())
    assert first_json == second_json
    assert first_hashes == second_hashes


def test_input_from_order_does_not_change_canonical_form():
    plan = load_reference_plan()
    permuted = plan.model_copy(deep=True)
    join = next(step for step in permuted.steps if step.id == "join")
    join.input_from = list(reversed(join.input_from))

    assert join.input_from != next(s for s in plan.steps if s.id == "join").input_from
    assert canonicalize(permuted)[0] == canonicalize(plan)[0]
    assert canonicalize(permuted)[1] == canonicalize(plan)[1]


def test_step_order_does_change_canonical_form():
    plan = load_reference_plan()
    reordered = plan.model_copy(deep=True)
    reordered.steps[0], reordered.steps[1] = reordered.steps[1], reordered.steps[0]

    assert canonicalize(reordered)[0] != canonicalize(plan)[0]
    # The steps themselves are untouched, so their individual hashes are unchanged.
    assert canonicalize(reordered)[1] == canonicalize(plan)[1]


def test_arguments_keys_are_sorted_recursively():
    left = AgentPlan(
        goal="g",
        steps=[Step(id="s1", tool="t", arguments={"b": 1, "a": {"z": 1, "y": 2}})],
    )
    right = AgentPlan(
        goal="g",
        steps=[Step(id="s1", tool="t", arguments={"a": {"y": 2, "z": 1}, "b": 1})],
    )
    assert canonicalize(left)[0] == canonicalize(right)[0]


def test_argument_list_order_is_meaningful():
    left = AgentPlan(goal="g", steps=[Step(id="s1", tool="t", arguments={"cols": ["a", "b"]})])
    right = AgentPlan(goal="g", steps=[Step(id="s1", tool="t", arguments={"cols": ["b", "a"]})])
    assert canonicalize(left)[0] != canonicalize(right)[0]


def test_step_hashes_cover_every_step_and_react_to_changes():
    plan = load_reference_plan()
    _, hashes = canonicalize(plan)
    assert set(hashes) == {step.id for step in plan.steps}

    changed = plan.model_copy(deep=True)
    next(step for step in changed.steps if step.id == "join").tool = "join_x"
    _, changed_hashes = canonicalize(changed)

    assert changed_hashes["join"] != hashes["join"]
    assert {k: v for k, v in changed_hashes.items() if k != "join"} == {
        k: v for k, v in hashes.items() if k != "join"
    }


def test_step_hash_matches_canonicalize_output():
    plan = load_reference_plan()
    _, hashes = canonicalize(plan)
    for step in plan.steps:
        assert step_hash(step) == hashes[step.id]
