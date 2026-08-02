"""Repairers and the scoring of what they return.

The port stage A left empty. Every repairer — rule-based here, autoregressive and diffusion
later — implements the same :class:`Repairer` signature, so a comparison between them is a
comparison of repairs and not of interfaces.

Nothing in this package imports torch: the interface layer knows nothing about models or
hardware, and the model backends will sit behind it as a separate layer.
"""

from plan_repair.repair.base import Repairer
from plan_repair.repair.deterministic import (
    DECLINED_ERROR_TYPES,
    HANDLED_ERROR_TYPES,
    DeterministicRepairer,
)
from plan_repair.repair.mock import IdentityRepairer, OracleRepairer
from plan_repair.repair.scoring import RepairScore, repair_and_score, score_repair

__all__ = [
    "DECLINED_ERROR_TYPES",
    "HANDLED_ERROR_TYPES",
    "DeterministicRepairer",
    "IdentityRepairer",
    "OracleRepairer",
    "RepairScore",
    "Repairer",
    "repair_and_score",
    "score_repair",
]
