"""Pure GT-free rules for R3 post-refresh geometry-state commit gating."""
from __future__ import annotations
import math

# Only rules with interpretable, currently comparable inputs are active.
POLICIES = (
    "AlwaysCommit",
    "NeverCommit",
    "AgeGate",
    "ResidualGate",
)

# Kept explicit in metadata so older R3 results remain interpretable.
DISABLED_POLICIES = {
    "ConfidenceGate": (
        "Disabled: candidate confidence comes from the keyframe confidence_mlp, "
        "while the fast propagation path currently sets logconf_student = logdepth_student; "
        "their values are not calibrated/comparable, so a candidate-minus-propagation "
        "confidence threshold is not a valid gate."
    ),
    "CombinedGate": (
        "Disabled because it uses the same non-comparable candidate/propagation confidence "
        "difference as ConfidenceGate. ResidualGate remains active independently."
    ),
}


def _finite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def decide(policy: str, features: dict, age_threshold: float = 16.0,
           residual_threshold: float = 0.10) -> bool:
    """Return a commit decision from current/causal non-GT features only.

    Missing or non-finite required features fail closed to DISCARD. Raw confidence
    values are diagnostic-only until both paths expose calibrated comparable values.
    """
    if policy == "AlwaysCommit":
        return True
    if policy == "NeverCommit":
        return False
    if policy == "AgeGate":
        age = features.get("distance_kf")
        return _finite(age) and float(age) >= float(age_threshold)
    if policy == "ResidualGate":
        residual = features.get("depth_disagreement_rel_mean")
        return _finite(residual) and float(residual) <= float(residual_threshold)
    if policy in DISABLED_POLICIES:
        raise ValueError(f"Policy {policy} is disabled: {DISABLED_POLICIES[policy]}")
    raise ValueError(f"Unknown R3 policy: {policy}")
