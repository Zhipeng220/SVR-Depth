# R3 Simple Causal Commit Gate (revised)

This experiment tests post-refresh writeback decisions with frozen weights. It does not train a predictor and does not skip refresh-candidate inference.

## Deployment
Upload this directory to `experiments/r_series/r3_simple_commit_gate/` and ensure the R2-A dependency directory and audited P1 files are already present.

## Commands (AutoDL)
```bash
bash experiments/r_series/r3_simple_commit_gate/run_r3.sh preflight
bash experiments/r_series/r3_simple_commit_gate/run_r3.sh smoke
bash experiments/r_series/r3_simple_commit_gate/run_r3.sh sensitivity
# Only after reviewing sensitivity outputs:
bash experiments/r_series/r3_simple_commit_gate/run_r3.sh full
```

Active policies in this revision: `AlwaysCommit`, `NeverCommit`, `AgeGate`, `ResidualGate`.

`ConfidenceGate` and `CombinedGate` are deliberately disabled: keyframe confidence and fast-path confidence currently have different, uncalibrated semantics. Raw confidence summaries remain in the decision log for diagnosis only.

The `sensitivity` stage performs 18 separate closed-loop ResidualGate runs: thresholds 0.03, 0.04, 0.05, 0.06, 0.08, and 0.10 across Crowd2, Balloon2, and Crowd3 (86 frames each). It writes combined action/summary CSVs. Do not select a threshold using GT metrics; use action/feature distributions first and require independent-sequence validation.

For the final `full` run, set `R3_RESIDUAL_THRESHOLD` to the preselected threshold, for example `R3_RESIDUAL_THRESHOLD=0.05 bash experiments/r_series/r3_simple_commit_gate/run_r3.sh full`. If unset, the default is 0.10; do not assume this is optimal.


## R3-C1 / R3-C2 full gate comparison and cross-sequence validation

The revised `run_r3.sh` adds:
- `baseline`: full 86-frame same-protocol comparison of AlwaysCommit, NeverCommit, AgeGate, ResidualGate@0.10 on all three sequences.
- `crossval`: use the latest baseline root and existing residual sensitivity sweep to perform leave-one-sequence-out threshold selection.
- `full`: run the baseline comparison, then LOSO validation. Requires an existing sensitivity sweep.

Run:
```bash
bash experiments/r_series/r3_simple_commit_gate/run_r3.sh preflight
bash experiments/r_series/r3_simple_commit_gate/run_r3.sh full
```

For reproducibility, pin the sensitivity sweep explicitly if needed:
```bash
R3_SENSITIVITY_ROOT=outputs/r_series/r3_residual_sensitivity/20261021T150941Z \
  bash experiments/r_series/r3_simple_commit_gate/run_r3.sh full
```

LOSO chooses each fold's residual threshold using only the other two sequences. Primary score is equal-weight mean AbsRel ratio to each training sequence's NeverCommit baseline; RMSE ratio, delta1 gain, then smaller threshold are tie-breakers. Three-sequence LOSO is exploratory and not an external test set. See `R3_CROSSVAL_PROTOCOL.md`.


### Cross-validation CSV schema hotfix
`r3_cross_sequence_validation.py` reads trajectory metrics from `residual_threshold_sweep_summary.csv` and derives action patterns/counts from `residual_threshold_sweep_actions.csv`. Do not manually add or edit columns in the original CSVs.
