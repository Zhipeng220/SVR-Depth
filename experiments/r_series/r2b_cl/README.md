# R2-B-CL: Dynamic closed-loop horizon oracle diagnostic

This package compares fixed commit endpoints (`000`, `111`) with two dynamic diagnostic oracles (`H1_Oracle_CL`, `H8_Oracle_CL`) under the existing K=3 fixed schedule (Smoke uses anchors 5, 14, 23 so H=8 ends before the next optional anchor; full uses 9, 37, 65). It does not train a predictor.

**Important:** H1/H8 use future ground-truth depth to choose COMMIT/DISCARD. They are non-deployable diagnostic upper bounds only. At every scheduled anchor, each oracle rolls out both branches from its own current state, chooses one branch, restores that branch's full runtime snapshot, continues along that selected state, and re-evaluates at the next anchor. The horizon is shorter than the gap between anchors, so there are no elective future anchors inside the lookahead; native restarts remain active.

## Install

Copy `experiments/r_series/r2b_cl/` into the same directory in the project. It requires the existing, audited `experiments/r_series/r2a_commit_enum/` folder and the historical P1 runtime helpers. It does not overwrite P1 or R2-A.

## Run

```bash
bash experiments/r_series/r2b_cl/run_r2b_cl.sh preflight
bash experiments/r_series/r2b_cl/run_r2b_cl.sh smoke
# only after Smoke passes:
bash experiments/r_series/r2b_cl/run_r2b_cl.sh full
```

Outputs are stored under `outputs/r_series/r2b_cl/`. Each run produces a review ZIP with metadata, source audit, trajectory metrics, oracle decision records, and per-branch future metrics.

## Interpretation

- `Fixed_000` and `Fixed_111`: fixed-schedule endpoint baselines.
- `H1_Oracle_CL` and `H8_Oracle_CL`: dynamic state-recursive future-GT diagnostics, not online policies.
- `oracle_branch_rollout_ms_total` and `oracle_branch_future_frames_evaluated` quantify the extra diagnostic work. Do not claim equal-cost or deployable speedup.
- AbsRel is the primary metric; RMSE and delta1 are secondary. Strategy metric masks must match.
