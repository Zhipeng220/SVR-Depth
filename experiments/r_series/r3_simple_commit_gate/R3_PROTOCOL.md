# R3 protocol: causal post-refresh commit gating (revised)

## Research question
Can simple current-state signals control post-refresh geometry-state writeback without future GT?

## Fixed protocol
- Frozen VeloDepth weights; no predictor training.
- Three optional anchors; Smoke: zero-based frames 5,14,23 over 32 frames. Full: 9,37,65 over 86 frames.
- Same refresh-candidate compute at each anchor for all policies.
- Native restart takes priority; no optional commit is made when a native restart occurs on an anchor.
- Only the four audited geometry fields are committed. Scheduler and fast-tail state are preserved.
- Gate selection uses current propagation output, current refresh candidate and current StreamState only. GT metrics are computed after the decision.

## Active policies
- `AlwaysCommit` and `NeverCommit` endpoints.
- `AgeGate`: `distance_kf >= 16` by default.
- `ResidualGate`: mean relative depth disagreement `mean(abs(candidate-prop)/max(abs(prop), eps)) <= threshold`.

## Explicitly disabled policies
- `ConfidenceGate` and `CombinedGate` are disabled in this revision. The candidate/keyframe confidence is emitted by `confidence_mlp`, while the fast propagation path currently sets `logconf_student = logdepth_student` before exponentiation. Their raw means therefore do not have demonstrated comparable/calibrated semantics. Logging raw means is diagnostic only; do not gate on their difference.

## Residual threshold sensitivity
- The `sensitivity` stage reruns ResidualGate from a clean initial state at thresholds `0.03, 0.04, 0.05, 0.06, 0.08, 0.10` on Crowd2, Balloon2, and Crowd3 (86 frames, anchors 9/37/65).
- Each sequence-threshold pair gets a fresh closed-loop trajectory, because earlier actions change later state/features.
- Decisions are GT-free. GT metrics are recorded only after actions and must not be used to pick a threshold in this sweep.
- Inspect action patterns and feature distributions first. Any candidate threshold still needs independent-sequence validation before claiming generalization.

## Readouts
Primary: full-sequence mean AbsRel. Secondary: RMSE, delta1. Also inspect commits, native restarts, gate features, per-anchor decisions, metric masks, R0 audit, and wall/step costs.

## Cost and interpretation
Post-refresh gating cannot save the refresh inference already executed at the anchor. A gate that improves one development sequence is not a validated policy. If no simple gate is robust across sequences, inspect feature/action relationships before training a predictor.
