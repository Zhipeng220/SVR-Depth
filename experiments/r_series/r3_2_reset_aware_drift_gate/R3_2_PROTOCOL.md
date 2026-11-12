# R3.2 protocol: reset-aware drift commit gate

## Research question
Can the failed original residual gate be improved by reversing its decision direction, and does requiring both keyframe age and residual drift improve robustness over AgeGate?

## Fixed setup
- Frozen VeloDepth weights; no predictor training.
- 86 frames; zero-based optional refresh anchors 9, 37, 65.
- Same three candidate computations for every policy. The gate controls only the four audited persistent geometry fields for future frames.
- Native restart takes priority; no optional commit at an anchor if a native restart replaces the candidate.
- GT is never used by a gate. GT is used only after the action to evaluate the emitted depth.
- Missing or non-finite required features fail closed to DISCARD.

## Policies
- `AlwaysCommit`: commit every available optional candidate.
- `NeverCommit`: never commit optional candidates.
- `AgeGate`: `distance_kf >= 16`.
- `ResidualGate`: original R3 control, `residual <= threshold`.
- `DriftGate`: new rule, `residual >= threshold`.
- `AgeDriftGate`: new rule, `distance_kf >= 16 AND residual >= threshold`.

The residual is the mean relative depth disagreement between the current refresh candidate and current propagation output, `mean(abs(candidate-prop)/max(abs(prop), eps))`. It is a disagreement feature, not a direct estimate of future benefit.

## Runs
1. `preflight`: unit tests and static source audit.
2. `smoke`: Crowd2 32 frames, anchors 5, 14, 23, all six policies.
3. `baseline`: all three sequences, 86 frames, anchors 9, 37, 65; all six policies at residual threshold 0.04.
4. `sensitivity`: all three sequences × thresholds 0.03, 0.04, 0.05, 0.06, 0.08, 0.10 × `ResidualGate`, `DriftGate`, `AgeDriftGate`. Every sequence-threshold pair starts from a clean initial state.

## Interpretation
- Inspect per-anchor features and action patterns before interpreting metric changes.
- Primary metric: full-sequence mean AbsRel. Secondary: RMSE and delta1. Report per-sequence results, not only macro averages.
- Compare new rules with `AgeGate`, `AlwaysCommit`, and `NeverCommit` on the same protocol.
- Do not call a threshold generalizable merely because it wins on these three development sequences. Any later LOSO is exploratory because these sequences already influenced rule design; a new independent sequence is needed for an external test.
- Post-refresh gating does not skip candidate inference. Do not claim refresh compute savings. Report actual path costs and native restarts.
- Do not add motion accumulation or scale-shift residual to this experiment; decide whether those diagnostics are justified after this sweep.
