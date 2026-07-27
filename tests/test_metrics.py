"""Detection metrics: the scorer itself has to fail when it should."""

from plan_repair.validation import PlanValidationResult, ValidationError, detection_metrics
from plan_repair.validation.metrics import error_signature

TOOL_ERROR = ValidationError(
    type="unknown_tool",
    step_ids=["join"],
    paths=["$.steps[?join].tool"],
    message="join uses an unknown tool",
)
STOP_ERROR = ValidationError(
    type="missing_stop_condition",
    step_ids=[],
    paths=["$.stop_condition"],
    message="plan has no stop_condition",
)


def result_of(*errors):
    return PlanValidationResult(valid=not errors, errors=list(errors))


def test_exact_match_scores_one():
    metrics = detection_metrics(
        result_of(TOOL_ERROR, STOP_ERROR),
        {error_signature(TOOL_ERROR), error_signature(STOP_ERROR)},
    )

    assert metrics.recall == 1.0
    assert metrics.precision == 1.0
    assert metrics.exact


def test_a_missed_error_lowers_recall():
    metrics = detection_metrics(
        result_of(TOOL_ERROR), {error_signature(TOOL_ERROR), error_signature(STOP_ERROR)}
    )

    assert metrics.recall == 0.5
    assert metrics.precision == 1.0
    assert metrics.missed == [error_signature(STOP_ERROR)]
    assert not metrics.exact


def test_a_spurious_error_lowers_precision():
    metrics = detection_metrics(result_of(TOOL_ERROR, STOP_ERROR), {error_signature(TOOL_ERROR)})

    assert metrics.recall == 1.0
    assert metrics.precision == 0.5
    assert metrics.spurious == [error_signature(STOP_ERROR)]
    assert not metrics.exact


def test_the_same_type_on_a_different_step_is_not_a_match():
    """Signatures compare step ids and paths, not just the error type."""
    elsewhere = TOOL_ERROR.model_copy(update={"step_ids": ["viz"], "paths": ["$.steps[?viz].tool"]})

    metrics = detection_metrics(result_of(elsewhere), {error_signature(TOOL_ERROR)})

    assert metrics.recall == 0.0
    assert metrics.precision == 0.0


def test_a_clean_plan_against_an_empty_golden_scores_one():
    metrics = detection_metrics(result_of(), set())

    assert metrics.recall == 1.0
    assert metrics.precision == 1.0
    assert metrics.exact


def test_detail_is_not_part_of_the_signature():
    annotated = TOOL_ERROR.model_copy(update={"detail": {"note": "extra"}})

    assert error_signature(annotated) == error_signature(TOOL_ERROR)
