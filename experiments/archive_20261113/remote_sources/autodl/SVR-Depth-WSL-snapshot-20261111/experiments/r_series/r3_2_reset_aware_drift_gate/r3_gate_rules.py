"""Pure GT-free rules for R3.2 post-refresh geometry-state commit gating."""
from __future__ import annotations
import math

POLICIES = (
    "AlwaysCommit",
    "NeverCommit",
    "AgeGate",
    "ResidualGate",       # Original rule: residual <= threshold; retained as a negative/control comparison.
    "DriftGate",          # New: commit only when residual >= threshold.
    "AgeDriftGate",       # New: require both sufficient keyframe age and sufficient residual.
)

DISABLED_POLICIES = {
    "ConfidenceGate": (
        "Disabled: candidate confidence comes from the keyframe confidence_mlp, "
        "while the fast propagation path currently sets logconf_student = logdepth_student; "
        "their values are not calibrated/comparable, so a candidate-minus-propagation "
        "confidence threshold is not a valid gate."
    ),
    "CombinedGate": (
        "Disabled because it uses the same non-comparable candidate/propagation confidence "
        "difference as ConfidenceGate."
    ),
}


def _finite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def decide(policy: str, features: dict, age_threshold: float = 16.0,
           residual_threshold: float = 0.04) -> bool:
    """Return a causal, GT-free decision. Missing required features fail closed to DISCARD."""
    if policy == "AlwaysCommit":
        return True
    if policy == "NeverCommit":
        return False

    if policy == "AgeGate":
        age = features.get("distance_kf")
        return _finite(age) and float(age) >= float(age_threshold)

    if policy == "ResidualGate":
        # Keep the R3 rule unchanged as a control, despite its poor LOSO results.
        residual = features.get("depth_disagreement_rel_mean")
        return _finite(residual) and float(residual) <= float(residual_threshold)

    if policy == "DriftGate":
        residual = features.get("depth_disagreement_rel_mean")
        return _finite(residual) and float(residual) >= float(residual_threshold)

    if policy == "AgeDriftGate":
        age = features.get("distance_kf")
        residual = features.get("depth_disagreement_rel_mean")
        return (
            _finite(age) and _finite(residual)
            and float(age) >= float(age_threshold)
            and float(residual) >= float(residual_threshold)
        )

    if policy in DISABLED_POLICIES:
        raise ValueError(f"Policy {policy} is disabled: {DISABLED_POLICIES[policy]}")
    raise ValueError(f"Unknown R3.2 policy: {policy}")
