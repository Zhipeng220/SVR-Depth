# P2.3 Online Replay v2

This revision is a correctness-focused update to the first 32-frame smoke-test script.

Changes:
1. Clones the packed input/flow payload before each action to prevent mutation leakage across frames or policies.
2. Performs a disposable keyframe + propagation warm-up independently for each policy before timing.
3. Records median, P95, and maximum model-action time; this helps expose outliers hidden by P95 on short clips.
4. Computes online features only for learned policies, and only at candidate frames.
5. Aligns learned-policy candidate decisions to the P2.2 anchor cadence (zero-based frame indices 1,5,9,... by default).
6. Refuses to run a leave-one-sequence-out predictor if the requested held-out sequence is absent from the training CSV.

Important interpretation:
- This is still an exploratory full-frame replay, not a local-refiner implementation.
- `threshold=0` is an initial positive-predicted-gain rule, not a calibrated same-budget threshold.
- The measured model-action time excludes input cloning, preprocessing, optical-flow construction, and metric evaluation; do not report it as end-to-end latency/FPS.
- VeloDepth weights remain frozen.
