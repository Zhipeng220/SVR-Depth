# R3-C1 / R3-C2: full gate comparison and leave-one-sequence-out validation

## Question
Do the simple post-refresh commit gates improve full-sequence depth quality, and does a residual threshold selected on other sequences transfer to an unseen sequence?

## Stage 1: full same-protocol baseline comparison
For Crowd2, Balloon2, and Crowd3, run 86 frames with zero-based optional refresh anchors 9, 37, and 65. Compare:
- `AlwaysCommit`
- `NeverCommit`
- `AgeGate` with the predeclared age threshold 16
- `ResidualGate` with reference threshold 0.10

All weights are frozen. Each strategy gets its own closed-loop trajectory. All strategies compute the scheduled refresh candidates; the commit gate only controls persistent geometry-state writeback.

## Stage 2: leave-one-sequence-out threshold selection
Use the already completed six-threshold sensitivity sweep (0.03, 0.04, 0.05, 0.06, 0.08, 0.10). For each held-out sequence:
1. The other two sequences are the training fold.
2. Select the threshold minimizing the equal-weight mean of per-sequence AbsRel ratios relative to that sequence's `NeverCommit` baseline.
3. Ties are broken by lower mean RMSE ratio, then larger delta1 gain, then the smaller threshold.
4. Only after the threshold is selected, read the held-out sequence metrics and compare against `AlwaysCommit`, `NeverCommit`, and `AgeGate`.

The held-out sequence's GT-derived metrics are never used to select its threshold. This is model selection using two development sequences, not independent external test evidence.

## Outputs
- `full_gate_comparison.csv`
- `loco_threshold_selection.csv`
- `loco_heldout_comparison.csv`
- `loco_fold_summary.csv`
- `crossval_metadata.json`
- `R3_完整门控对照与LOSO验证报告.md`
- `<run_root>_review.zip`

## Interpretation boundary
Only three sequences are available, so results are exploratory. If residual threshold choices vary across folds or lose to AgeGate on held-out sequences, do not claim a robust residual gate. ConfidenceGate and CombinedGate remain disabled until candidate and propagation confidence semantics are comparable. This is post-refresh commit gating and does not demonstrate skipped refresh compute.


## Data-interface hotfix (2026-10-22)
The sweep summary stores trajectory metrics only. The `pattern` and `commits` fields are derived at runtime by joining `residual_threshold_sweep_actions.csv` on sequence and threshold, using the fixed anchor order 9, 37, 65. This avoids requiring action-derived columns to exist in the summary CSV.
