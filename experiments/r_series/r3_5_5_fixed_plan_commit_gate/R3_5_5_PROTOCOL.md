# R3.5.5 Protocol Lock

## Goal
One final feasibility test of post-candidate persistent-geometry writeback gating under a fixed candidate schedule. This is not a new predictor or compute-saving claim.

## Fixed settings
- Sequences: crowd2, balloon2, crowd3; 86 frames each.
- Zero-based action slots: 5, 17, 29, 41, 49, 61 (six per sequence; 18 planned slots total).
- Feature: `raw_rel_median` only.
- Counterfactual horizon H=16; discount gamma=0.9.
- Frozen model weights; no neural predictor training.
- Threshold rule: COMMIT iff `raw_rel_median >= threshold`.
- Threshold calibration: leave one sequence out; select threshold by maximizing the sum of same-state local gains on the other two sequences; ties prefer more commits.
- Local gain: `gain_absrel_discounted_per_valid_frame`; positive means committing lowers discounted future AbsRel.
- Offline stop condition: OOF local utility exceeds AlwaysCommit on at least 2/3 held-out sequences and pooled OOF utility delta > 0.
- Only if offline stop condition passes: run each held-out sequence using its training-fold threshold.
- Closed-loop pass: at least 2/3 sequences improve mean AbsRel; macro mean improves; no sequence has >2% relative AbsRel regression.
- Candidate refresh is always computed before gating. No compute savings are claimed.
- The three sequences are development data, not a locked external test set.
