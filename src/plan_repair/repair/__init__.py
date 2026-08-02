"""Repairers and the scoring of what they return.

The port stage A left empty. Every repairer — rule-based here, autoregressive and diffusion
later — implements the same :class:`Repairer` signature, so a comparison between them is a
comparison of repairs and not of interfaces.

Nothing in this package imports torch: the interface layer knows nothing about models or
hardware, and the model backends will sit behind it as a separate layer.
"""

from plan_repair.repair.ar import (
    API_FAILURE,
    PARSE_FAILURE,
    ARFullRepairer,
    ARLocalRepairer,
    RepairFailure,
)
from plan_repair.repair.base import Repairer
from plan_repair.repair.deterministic import (
    DECLINED_ERROR_TYPES,
    HANDLED_ERROR_TYPES,
    DeterministicRepairer,
)
from plan_repair.repair.diffusion_mock import NoisyDiffusion, OracleDiffusion
from plan_repair.repair.llm_client import (
    LLMClient,
    LLMError,
    OpenAIClient,
    ScriptedLLMClient,
)
from plan_repair.repair.mock import IdentityRepairer, OracleRepairer
from plan_repair.repair.plan_io import PlanParseError, parse_plan, plan_to_json, task_to_json
from plan_repair.repair.remask import (
    DEFAULT_PLACEHOLDER,
    MaskSpec,
    PlanSequence,
    StepSpan,
    fill_masked,
    mask_spec,
    plan_to_sequence,
    render_masked,
    sequence_to_plan,
)
from plan_repair.repair.scoring import RepairScore, repair_and_score, score_repair

__all__ = [
    "API_FAILURE",
    "DECLINED_ERROR_TYPES",
    "DEFAULT_PLACEHOLDER",
    "HANDLED_ERROR_TYPES",
    "PARSE_FAILURE",
    "ARFullRepairer",
    "ARLocalRepairer",
    "DeterministicRepairer",
    "IdentityRepairer",
    "LLMClient",
    "LLMError",
    "MaskSpec",
    "NoisyDiffusion",
    "OpenAIClient",
    "OracleDiffusion",
    "OracleRepairer",
    "PlanParseError",
    "PlanSequence",
    "RepairFailure",
    "RepairScore",
    "Repairer",
    "ScriptedLLMClient",
    "StepSpan",
    "fill_masked",
    "mask_spec",
    "parse_plan",
    "plan_to_json",
    "plan_to_sequence",
    "render_masked",
    "repair_and_score",
    "score_repair",
    "sequence_to_plan",
    "task_to_json",
]
