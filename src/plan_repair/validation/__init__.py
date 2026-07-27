"""Plan validation — five structural checks and their contract."""

from plan_repair.validation.metrics import (
    DetectionMetrics,
    ErrorSignature,
    detection_metrics,
    error_signature,
)
from plan_repair.validation.models import (
    DANGLING_STEP,
    DEP_CYCLE,
    DUPLICATE_STEP,
    MISSING_STOP_CONDITION,
    ORDERING,
    SCHEMA,
    UNKNOWN_DEPENDENCY,
    UNKNOWN_TOOL,
    PlanValidationResult,
    ValidationError,
)
from plan_repair.validation.paths import (
    input_from_path,
    step_path,
    stop_condition_path,
    tool_path,
)
from plan_repair.validation.validator import validate_plan

__all__ = [
    "DANGLING_STEP",
    "DEP_CYCLE",
    "DUPLICATE_STEP",
    "MISSING_STOP_CONDITION",
    "ORDERING",
    "SCHEMA",
    "UNKNOWN_DEPENDENCY",
    "UNKNOWN_TOOL",
    "DetectionMetrics",
    "ErrorSignature",
    "PlanValidationResult",
    "ValidationError",
    "detection_metrics",
    "error_signature",
    "input_from_path",
    "step_path",
    "stop_condition_path",
    "tool_path",
    "validate_plan",
]
