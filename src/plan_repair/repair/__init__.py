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
from plan_repair.repair.diagnostics import (
    ParseFailure,
    RepairDiagnostics,
    diagnose,
    excerpt_around,
    summarise,
)
from plan_repair.repair.diffusion import (
    ALIGNMENT_FAILURE,
    BACKEND_FAILURE,
    DiffusionRepairer,
    DreamRepairer,
    LLaDARepairer,
)
from plan_repair.repair.diffusion_mock import NoisyDiffusion, OracleDiffusion
from plan_repair.repair.dllm_backend import (
    DLLMBackend,
    DLLMError,
    EchoBackend,
    FailingBackend,
    FillRequest,
    OracleBackend,
)
from plan_repair.repair.dllm_backend_torch import TorchDLLMBackend, torch_available
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
    field_spans,
    fill_masked,
    mask_spec,
    mask_spec_from_paths,
    normalise_filling,
    plan_to_sequence,
    render_masked,
    sequence_to_plan,
)
from plan_repair.repair.scoring import RepairScore, repair_and_score, score_repair
from plan_repair.repair.tokenization import (
    ByteOffsetTokenizer,
    HuggingFaceTokenizer,
    OffsetTokenizer,
    TokenAlignment,
    TokenizationError,
    align_mask,
    decode_spans,
    masked_token_ids,
)

__all__ = [
    "ALIGNMENT_FAILURE",
    "API_FAILURE",
    "BACKEND_FAILURE",
    "DECLINED_ERROR_TYPES",
    "DEFAULT_PLACEHOLDER",
    "HANDLED_ERROR_TYPES",
    "PARSE_FAILURE",
    "ARFullRepairer",
    "ARLocalRepairer",
    "ByteOffsetTokenizer",
    "DLLMBackend",
    "DLLMError",
    "DeterministicRepairer",
    "DiffusionRepairer",
    "DreamRepairer",
    "EchoBackend",
    "FailingBackend",
    "FillRequest",
    "HuggingFaceTokenizer",
    "IdentityRepairer",
    "LLMClient",
    "LLMError",
    "LLaDARepairer",
    "MaskSpec",
    "NoisyDiffusion",
    "OffsetTokenizer",
    "OpenAIClient",
    "OracleBackend",
    "OracleDiffusion",
    "OracleRepairer",
    "ParseFailure",
    "PlanParseError",
    "PlanSequence",
    "RepairDiagnostics",
    "RepairFailure",
    "RepairScore",
    "Repairer",
    "ScriptedLLMClient",
    "StepSpan",
    "TokenAlignment",
    "TokenizationError",
    "TorchDLLMBackend",
    "align_mask",
    "decode_spans",
    "diagnose",
    "excerpt_around",
    "field_spans",
    "fill_masked",
    "mask_spec",
    "mask_spec_from_paths",
    "masked_token_ids",
    "normalise_filling",
    "parse_plan",
    "plan_to_json",
    "plan_to_sequence",
    "render_masked",
    "repair_and_score",
    "score_repair",
    "sequence_to_plan",
    "summarise",
    "task_to_json",
    "torch_available",
]
