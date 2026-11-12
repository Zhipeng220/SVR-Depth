# P2.3 v5.2 — label-supported fixed-budget score-ranking diagnostic

## Why this experiment
P2.3 v5.1 showed sequence-dependent score distributions and threshold transfer failure. Also, H=16 eligible labels do not cover every candidate in an 86-frame replay: later anchors (e.g. 73, 81, 85) lack a full 16-frame future window. Balloon2 also has a restart-crossing gap in eligible labels.

This experiment restricts scheduled actions and scored candidate anchors to the sequence-specific P2.2.2 eligible H=16 anchors. It then compares:
- `candidate_fixed`: fixed-interval actions only on those eligible anchors.
- `budget_topk`: choose exactly K candidate frames by sorting scores from a separate no-writeback pass, where K equals the number of eligible fixed-baseline action frames.

The score-ranked schedule is **retrospective and non-causal**: it uses the whole score list from the no-writeback pass to choose K frames. It is a diagnostic to separate ranking quality from absolute threshold calibration, not a deployable online scheduler. Selection does not read labels.

## Files to upload
Upload into `experiments/p1/`:
- `p2_3_online_replay_v5_2_budget_rank.py`
- `p2_3_select_topk_budget_frames.py`
- `run_p2_3_v5_2_budget_rank.sh`

The existing dataset must be present at `outputs/p2_2_2/p2_2_2_valid_dataset.csv`.

## Run
```bash
cd /root/autodl-tmp/SVR-Depth/velodepth
conda activate svrdepth
bash experiments/p1/run_p2_3_v5_2_budget_rank.sh
```

The script runs one no-writeback score pass per sequence, then two fixed-budget replay comparisons per sequence (state-only score ranking and motion+state score ranking). VeloDepth weights remain frozen. Automatic restarts remain enabled.

## Outputs
`outputs/p2_3_v5_2_budget_rank/p2_3_v5_2_budget_rank_review.zip`

Includes per-sequence score-only replay, selected frame lists, budget-matched replays, summaries, per-frame CSVs, metadata and logs.

## Interpretation
- Compare `budget_topk` against `candidate_fixed` at the same requested action count K.
- Report `scheduled_action_attempt_count` separately from actual `state_writeback_count`, because an automatic restart can override a requested writeback at that frame.
- All metrics use the same GT-valid frame mask within each sequence.
- Do not claim compute savings: each state-writeback still computes a full keyframe candidate.
- Since top-K selection is retrospective, even a positive result only demonstrates score-ranking potential. A causal online budget controller would be a later experiment.
