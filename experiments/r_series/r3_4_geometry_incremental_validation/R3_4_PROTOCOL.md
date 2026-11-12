# R3.4 Protocol Lock

- Purpose: offline LOSO incremental validation of candidate geometry features.
- Input: frozen R3.3 review ZIP or extracted R3.3 run directory.
- Target: `gain_absrel_discounted_per_valid_frame`.
- Feature models and controls are predeclared in `r3_4_incremental_validation.py`.
- Validation split: leave one sequence out; never random split anchors.
- Ridge alpha: 1.0 fixed.
- Prediction decision threshold: gain prediction > 0 means COMMIT, else DISCARD.
- AgeGate comparator: `distance_kf >= 16`.
- No GPU, no model inference, no neural predictor training, no threshold tuning.
- Primary output: out-of-sequence metrics on all diagnostic anchors.
- Secondary output: only the 9 actual policy action anchors.
- Limitation: three development sequences; overlapping H=16 counterfactual windows; local utility is not closed-loop utility.
