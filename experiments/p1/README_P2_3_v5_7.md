# P2.3 v5.7 — Equal-Budget Online Policy Comparison

## Objective
Test whether the simple state-age signal found in v5.6 LOSO validation translates into better online state-writeback scheduling under a pre-specified writeback budget, compared with an evenly spaced fixed-budget baseline and the current Motion + Learned controller.

This is a frozen-weight online replay experiment. It does not train/update VeloDepth weights and does not establish end-to-end latency or compute savings.

## Policies
All policies use the same VeloDepth stream, P1.5 B1-aligned persistent geometry state writeback (`--scheduled-action state_writeback`), candidate cadence, minimum gap, and pre-specified budget.

1. `auto`: original automatic keyframe/restart behavior, reference baseline.
2. `budget_fixed`: evenly spaced, gap-feasible candidate anchors targeting K successful state writebacks. If an automatic restart preempts a planned action, a later eligible candidate may replace the missed action; this is logged.
3. `causal_budget_state_age`: uses the v5.6 LOSO simple heuristic `scheduler_distance_kf` as the current score, with only past scores used by the causal budget controller.
4. `causal_budget_motion_plus_learned`: existing Ridge predictor trained on P2.2.2 H=16 rows from the other two sequences; OOF calibration excludes the test sequence.

### Important state-age caveat
B1 state writeback commits persistent geometry fields but preserves scheduler fields, including `distance_kf`. Therefore `scheduler_distance_kf` may continue increasing after writeback. The v5.7 logs must be inspected for whether the raw age score causes budget actions to concentrate early in the sequence. This is a deliberate mechanism test of the v5.6 signal, not an assumption that the signal resets after writeback.

## Budget and online causality
- Smoke: Crowd2, 32 frames, K=3, candidate stride 4, offset 1, minimum action gap 8.
- Full run: Crowd2, Balloon2, Crowd3, 86 frames each, K=8.
- Candidate anchors: t = 1, 5, 9, 13, ... (zero-based).
- A scheduled action is eligible only when at least 8 frames have elapsed since the last policy action.
- The causal controller uses the current score, strictly past scores, current writeback count, and the known number of future *gap-feasible candidate slots*. It force-fills only when remaining feasible slots equal or are fewer than the remaining budget. No future gain label is used for action decisions.
- Automatic restart may preempt a scheduled writeback. The experiment logs requested actions, successful state writebacks, restart count, forced decisions, and action frames. If any policy fails to reach K because restarts consume too many remaining eligible slots, report actual writes and do not describe the comparison as exactly equal realized budget.

## Files to upload
Upload these files into the remote project directory `experiments/p1/`:
- `p2_3_online_replay_v5_7_equal_budget.py`
- `run_p2_3_v5_7_smoke.sh`
- `run_p2_3_v5_7_full.sh`
- `README_P2_3_v5_7.md`

## Step 1 — smoke test
In the AutoDL Linux terminal:
```bash
cd /root/autodl-tmp/SVR-Depth/velodepth
conda activate svrdepth
bash experiments/p1/run_p2_3_v5_7_smoke.sh
```
Check that all four policy summaries exist, metrics are finite on GT-valid frames, action counts never exceed K=3, and there is no traceback/OOM. Review the logged action frames. If the state-age policy writes back at nearly every earliest eligible anchor, that is an important finding but not a runtime failure.

Review archive:
`outputs/p2_3_v5_7_equal_budget/crowd2_32_smoke_review.zip`

## Step 2 — full experiment
Only after the smoke test completes successfully:
```bash
cd /root/autodl-tmp/SVR-Depth/velodepth
conda activate svrdepth
bash experiments/p1/run_p2_3_v5_7_full.sh
```

Review archive:
`outputs/p2_3_v5_7_equal_budget/p2_3_v5_7_equal_budget_review.zip`

## Outputs and required analysis
Each sequence directory contains:
- `p2_3_online_frame_metrics.csv`
- `p2_3_online_summary.csv`
- `p2_3_online_metadata.json`
- OOF calibration CSVs for learned predictors

Compare, per sequence and policy:
- mean AbsRel, RMSE, delta1;
- successful state writebacks versus K;
- action frame indices and intervals;
- auto-restart count and forced-budget decisions;
- decision time and model-action timing.

Interpret the main comparison only if realized writeback counts match. If counts differ, first report the discrepancy and either compare the matched-budget subset or rerun a protocol that guarantees equal successful writes. The scheduled action still computes a full keyframe candidate, so do not claim compute savings. Timing excludes full preprocessing/optical flow and metric evaluation and is not end-to-end FPS.

## Static validation status
The package is checked for Python syntax, Shell syntax, and ZIP integrity. The GPU replay itself has not been run by this package creator; actual results must come from the user's AutoDL run.
