"""Pure, GT-free rules for R3 post-refresh persistent-state commit gating."""
from __future__ import annotations
import math

POLICIES = (
    "AlwaysCommit",
    "NeverCommit",
    "AgeGate",
    "ConfidenceGate",
    "ResidualGate",
    "CombinedGate",
)


def _finite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def decide(policy: str, features: dict, age_threshold: float = 16.0,
           residual_threshold: float = 0.10) -> bool:
    """Return commit decision using only causal, non-GT features.

    Missing/non-finite required features fail closed to DISCARD.
    ResidualGate commits when relative candidate/propagation depth disagreement
    is <= the predeclared threshold. CombinedGate additionally requires the
    refresh candidate's mean confidence to be no lower than propagation's.
    """
    if policy == "AlwaysCommit":
        return True
    if policy == "NeverCommit":
        return False
    if policy == "AgeGate":
        age = features.get("distance_kf")
        return _finite(age) and float(age) >= float(age_threshold)
    if policy == "ConfidenceGate":
        delta = features.get("confidence_delta_candidate_minus_prop")
        return _finite(delta) and float(delta) >= 0.0
    if policy == "ResidualGate":
        residual = features.get("depth_disagreement_rel_mean")
        return _finite(residual) and float(residual) <= float(residual_threshold)
    if policy == "CombinedGate":
        delta = features.get("confidence_delta_candidate_minus_prop")
        residual = features.get("depth_disagreement_rel_mean")
        return (_finite(delta) and float(delta) >= 0.0
                and _finite(residual) and float(residual) <= float(residual_threshold))
    raise ValueError(f"Unknown R3 policy: {policy}")
