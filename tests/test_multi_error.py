"""Multi-error detection under the strict criterion.

Tickets 001 and 002 scored recall by containment. Here the detected error set must match the
expected set *exactly*: a miss fails, and so does a spurious error.

Every golden below was computed by hand from the pipeline graph before the validator was run
(see the combination comments for the derivation). Reading goldens back out of the validator
would make the check self-fulfilling — whatever it emitted would define the truth.

Each expected set holds the errors a corruption causes directly (primary) plus the errors that
follow from it structurally (derived); the derived entries are the judgement calls, and each one
is justified in the comment above its combination.
"""

import pytest

from plan_repair.corruption import CorruptionSpec, inject_multi
from plan_repair.data import DATA_PIPELINE_A, DATA_PIPELINE_B, load_reference
from plan_repair.schema import (
    BROKEN_DEPENDENCY,
    DUPLICATE_STEP,
    MISSING_STOP_CONDITION,
    STEP_DELETION,
    WRONG_ORDERING,
    WRONG_TOOL,
)
from plan_repair.validation import (
    DEP_CYCLE,
    ORDERING,
    detection_metrics,
    validate_plan,
)


def spec(corruption_type, step_id=None, **options):
    return CorruptionSpec(corruption_type=corruption_type, step_id=step_id, options=options)


def cycle_golden(members):
    """A dep_cycle error carries its whole strongly connected component, in plan order."""
    return (DEP_CYCLE, tuple(members), tuple(f"$.steps[?{m}].input_from" for m in members))


# --- domain B combinations ---------------------------------------------------------------------
#
# M-1  independent: renaming a tool touches no edge and no position, and the stop condition is
#      plan level, so neither corruption can produce a derived error.
M1_SPECS = [spec(WRONG_TOOL, "join"), spec(MISSING_STOP_CONDITION)]
M1_GOLDEN = {
    ("unknown_tool", ("join",), ("$.steps[?join].tool",)),
    ("missing_stop_condition", (), ("$.stop_condition",)),
}

# M-2  independent: moving agg in front of enrich changes positions only — edges, consumers and
#      terminals are untouched — so the single ordering violation is the whole effect.
M2_SPECS = [spec(WRONG_ORDERING, "agg"), spec(WRONG_TOOL, "viz")]
M2_GOLDEN = {
    ("ordering", ("agg", "enrich"), ("$.steps[?agg]",)),
    ("unknown_tool", ("viz",), ("$.steps[?viz].tool",)),
}

# M-3  interaction: deleting join breaks enrich's reference and strands n_csv and n_db; deleting
#      co breaks n_csv's reference and strands cm. Both are the derived dangling pattern already
#      accepted in Ticket 002 (B-3, B-4). The two deletions meet at n_csv, which is reported
#      twice for two different true facts: its own reference is broken (co is gone) *and* nothing
#      consumes it any more (join is gone).
M3_SPECS = [spec(STEP_DELETION, "join"), spec(STEP_DELETION, "co")]
M3_GOLDEN = {
    ("unknown_dependency", ("n_csv",), ("$.steps[?n_csv].input_from",)),
    ("unknown_dependency", ("enrich",), ("$.steps[?enrich].input_from",)),
    ("dangling_step", ("cm",), ("$.steps[?cm]",)),
    ("dangling_step", ("n_csv",), ("$.steps[?n_csv]",)),
    ("dangling_step", ("n_db",), ("$.steps[?n_db]",)),
}

# M-5  masking: the cycle makes topological order undefined, so the ordering check is skipped
#      (Ticket 001 contract) and the injected wrong_ordering is deliberately not reported as an
#      ordering error. The moved step is still named, as a member of the cycle component.
#      Component members are listed in the order of the *final* broken plan, where agg sits in
#      front of enrich — evidence that the move did happen.
M5_SPECS = [
    spec(BROKEN_DEPENDENCY, "l_csv", mode="cycle", cycle_with="report"),
    spec(WRONG_ORDERING, "agg"),
]
M5_CYCLE_MEMBERS = [
    "l_csv", "pr_csv", "vs_csv", "cm", "co", "n_csv", "join", "agg",
    "enrich", "pivot", "stat", "corr", "viz", "interp", "report",
]  # fmt: skip
M5_GOLDEN = {cycle_golden(M5_CYCLE_MEMBERS)}

# M-6  triple: three independent corruptions. The copy of agg is derived-dangling — pivot still
#      consumes the original, so nothing consumes agg_dup (Ticket 002, B-7).
M6_SPECS = [spec(WRONG_TOOL, "join"), spec(MISSING_STOP_CONDITION), spec(DUPLICATE_STEP, "agg")]
M6_GOLDEN = {
    ("unknown_tool", ("join",), ("$.steps[?join].tool",)),
    ("missing_stop_condition", (), ("$.stop_condition",)),
    ("duplicate_step", ("agg", "agg_dup"), ("$.steps[?agg]", "$.steps[?agg_dup]")),
    ("dangling_step", ("agg_dup",), ("$.steps[?agg_dup]",)),
}

# --- domain A combinations (same shapes, mapped onto the research pipeline) ---------------------
#
# MA-3 mirrors M-3: dedupe is A's fan-in, p_paper its chain middle. Deleting dedupe strands all
# three extract steps; deleting p_paper strands f_paper. e_paper plays n_csv's double role.
MA1_SPECS = [spec(WRONG_TOOL, "dedupe"), spec(MISSING_STOP_CONDITION)]
MA1_GOLDEN = {
    ("unknown_tool", ("dedupe",), ("$.steps[?dedupe].tool",)),
    ("missing_stop_condition", (), ("$.stop_condition",)),
}

MA2_SPECS = [spec(WRONG_ORDERING, "xcheck"), spec(WRONG_TOOL, "vsrc")]
MA2_GOLDEN = {
    ("ordering", ("xcheck", "dedupe"), ("$.steps[?xcheck]",)),
    ("unknown_tool", ("vsrc",), ("$.steps[?vsrc].tool",)),
}

MA3_SPECS = [spec(STEP_DELETION, "dedupe"), spec(STEP_DELETION, "p_paper")]
MA3_GOLDEN = {
    ("unknown_dependency", ("e_paper",), ("$.steps[?e_paper].input_from",)),
    ("unknown_dependency", ("xcheck",), ("$.steps[?xcheck].input_from",)),
    ("dangling_step", ("f_paper",), ("$.steps[?f_paper]",)),
    ("dangling_step", ("e_web",), ("$.steps[?e_web]",)),
    ("dangling_step", ("e_paper",), ("$.steps[?e_paper]",)),
    ("dangling_step", ("e_news",), ("$.steps[?e_news]",)),
}

# The paper and news branches never lead back to s_web, so they stay out of the component.
MA5_SPECS = [
    spec(BROKEN_DEPENDENCY, "s_web", mode="cycle", cycle_with="fmt"),
    spec(WRONG_ORDERING, "xcheck"),
]
MA5_CYCLE_MEMBERS = [
    "s_web", "f_web", "p_web", "e_web", "xcheck", "dedupe",
    "vsrc", "rank", "cite", "synth", "fmt",
]  # fmt: skip
MA5_GOLDEN = {cycle_golden(MA5_CYCLE_MEMBERS)}

MA6_SPECS = [
    spec(WRONG_TOOL, "dedupe"),
    spec(MISSING_STOP_CONDITION),
    spec(DUPLICATE_STEP, "xcheck"),
]
MA6_GOLDEN = {
    ("unknown_tool", ("dedupe",), ("$.steps[?dedupe].tool",)),
    ("missing_stop_condition", (), ("$.stop_condition",)),
    ("duplicate_step", ("xcheck", "xcheck_dup"), ("$.steps[?xcheck]", "$.steps[?xcheck_dup]")),
    ("dangling_step", ("xcheck_dup",), ("$.steps[?xcheck_dup]",)),
}

COMBINATIONS = {
    "M-1": (DATA_PIPELINE_B, M1_SPECS, M1_GOLDEN),
    "M-2": (DATA_PIPELINE_B, M2_SPECS, M2_GOLDEN),
    "M-3": (DATA_PIPELINE_B, M3_SPECS, M3_GOLDEN),
    "M-5": (DATA_PIPELINE_B, M5_SPECS, M5_GOLDEN),
    "M-6": (DATA_PIPELINE_B, M6_SPECS, M6_GOLDEN),
    "MA-1": (DATA_PIPELINE_A, MA1_SPECS, MA1_GOLDEN),
    "MA-2": (DATA_PIPELINE_A, MA2_SPECS, MA2_GOLDEN),
    "MA-3": (DATA_PIPELINE_A, MA3_SPECS, MA3_GOLDEN),
    "MA-5": (DATA_PIPELINE_A, MA5_SPECS, MA5_GOLDEN),
    "MA-6": (DATA_PIPELINE_A, MA6_SPECS, MA6_GOLDEN),
}


def run(case):
    domain, specs, golden = COMBINATIONS[case]
    task, plan = load_reference(domain)
    corruption = inject_multi(plan, specs)
    result = validate_plan(corruption.broken_plan, task)
    return plan, corruption, result, golden


@pytest.mark.parametrize("case", sorted(COMBINATIONS))
def test_detection_matches_the_expected_error_set_exactly(case):
    _, _, result, golden = run(case)
    metrics = detection_metrics(result, golden)

    assert metrics.missed == [], f"{case}: missed errors"
    assert metrics.spurious == [], f"{case}: spurious errors"
    assert metrics.recall == 1.0
    assert metrics.precision == 1.0
    assert metrics.exact


@pytest.mark.parametrize("case", sorted(COMBINATIONS))
def test_each_corruption_is_tracked_individually(case):
    _, corruption, _, _ = run(case)
    _, specs, _ = COMBINATIONS[case]

    assert len(corruption.injected) == len(specs)
    assert [error.corruption_type for error in corruption.injected] == [
        s.corruption_type for s in specs
    ]


@pytest.mark.parametrize("case", sorted(COMBINATIONS))
def test_the_reference_plan_is_never_mutated(case):
    plan, corruption, _, _ = run(case)
    domain, _, _ = COMBINATIONS[case]
    _, pristine = load_reference(domain)

    assert plan == pristine
    assert corruption.broken_plan != pristine


def test_m3_reports_the_overlapping_step_under_both_true_facts():
    """n_csv lost the step it consumed *and* the step that consumed it — two distinct errors."""
    _, _, result, _ = run("M-3")

    types_for_n_csv = {error.type for error in result.errors if error.step_ids == ["n_csv"]}

    assert types_for_n_csv == {"unknown_dependency", "dangling_step"}


def test_m5_ordering_stays_masked_by_the_cycle():
    """The Ticket 001 hierarchy has to survive combination: no ordering error next to a cycle."""
    _, corruption, result, _ = run("M-5")

    assert result.errors_of_type(ORDERING) == []
    cycle = result.errors_of_type(DEP_CYCLE)
    assert len(cycle) == 1
    assert "agg" in cycle[0].step_ids
    # The injected ordering corruption is still recorded as ground truth, masked or not.
    assert [error.corruption_type for error in corruption.injected][1] == WRONG_ORDERING
    assert corruption.injected[1].damaged_paths == ["$.steps[?agg]"]


def test_m6_duplicate_detail_names_the_original():
    _, corruption, result, _ = run("M-6")

    duplicate = result.errors_of_type(DUPLICATE_STEP)[0]

    assert duplicate.detail["kind"] == "identical_content"
    assert corruption.injected[2].detail["duplicate_of"] == "agg"


# --- interference guard --------------------------------------------------------------------------


def test_two_corruptions_on_the_same_step_are_refused():
    """The M-4 shape (two errors on one step) never reaches the validator."""
    _, plan = load_reference()

    with pytest.raises(ValueError, match="both touch step 'join'"):
        inject_multi(plan, [spec(STEP_DELETION, "join"), spec(WRONG_TOOL, "join")])


def test_a_step_named_by_an_option_also_counts_as_touched():
    """cycle_with names an existing step, so another corruption may not take it."""
    _, plan = load_reference()

    with pytest.raises(ValueError, match="both touch step 'report'"):
        inject_multi(
            plan,
            [
                spec(BROKEN_DEPENDENCY, "l_csv", mode="cycle", cycle_with="report"),
                spec(WRONG_TOOL, "report"),
            ],
        )


def test_deletions_whose_consequences_meet_are_allowed():
    """M-3 is the interaction this ticket measures: distinct targets, overlapping fallout."""
    _, plan = load_reference()

    corruption = inject_multi(plan, M3_SPECS)

    assert len(corruption.injected) == 2


def test_unknown_step_is_refused_before_anything_is_applied():
    _, plan = load_reference()

    with pytest.raises(ValueError, match="no such step: 'nope'"):
        inject_multi(plan, [spec(WRONG_TOOL, "nope"), spec(MISSING_STOP_CONDITION)])


def test_a_combination_needs_at_least_two_corruptions():
    _, plan = load_reference()

    with pytest.raises(ValueError, match="at least two corruptions"):
        inject_multi(plan, [spec(WRONG_TOOL, "join")])


def test_unknown_corruption_type_is_refused():
    _, plan = load_reference()

    with pytest.raises(ValueError, match="unknown corruption type"):
        inject_multi(plan, [spec("wrong_arguments", "join"), spec(MISSING_STOP_CONDITION)])


def test_plan_level_corruption_takes_no_step():
    _, plan = load_reference()

    with pytest.raises(ValueError, match="plan level and takes no step_id"):
        inject_multi(plan, [spec(MISSING_STOP_CONDITION, "join"), spec(WRONG_TOOL, "agg")])


def test_step_level_corruption_needs_a_step():
    _, plan = load_reference()

    with pytest.raises(ValueError, match="needs a step_id"):
        inject_multi(plan, [spec(WRONG_TOOL), spec(MISSING_STOP_CONDITION)])
