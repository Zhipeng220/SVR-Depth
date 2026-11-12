# R3.2 Reset-Aware Drift Gate

This is a new, isolated experiment directory. It does not overwrite the existing R3 scripts or outputs. It adds two causal policies while preserving the original `ResidualGate` as a control.

## Install
Extract the bundle at the VeloDepth project root so this path exists:
`experiments/r_series/r3_2_reset_aware_drift_gate/`

The sibling dependency `experiments/r_series/r2a_commit_enum/` and the audited P1 source files must already exist.

## AutoDL commands
```bash
cd /root/autodl-tmp/SVR-Depth/velodepth
conda activate svrdepth
bash experiments/r_series/r3_2_reset_aware_drift_gate/run_r32.sh start
bash experiments/r_series/r3_2_reset_aware_drift_gate/run_r32.sh preflight
bash experiments/r_series/r3_2_reset_aware_drift_gate/run_r32.sh smoke
bash experiments/r_series/r3_2_reset_aware_drift_gate/run_r32.sh baseline
bash experiments/r_series/r3_2_reset_aware_drift_gate/run_r32.sh sensitivity
bash experiments/r_series/r3_2_reset_aware_drift_gate/run_r32.sh pack
```

`start` creates a new timestamped run root. Run all following stages in the same project checkout. If needed, set `R32_ROOT=/absolute/path/to/run_root` to select a specific run.

## What is tested
- `DriftGate`: `residual >= threshold`.
- `AgeDriftGate`: `distance_kf >= 16 and residual >= threshold`.
- `ResidualGate`: unchanged old direction, `residual <= threshold`, retained as a control.
- Baselines: `AlwaysCommit`, `NeverCommit`, `AgeGate`.

The sensitivity sweep creates 54 closed-loop trajectories (3 sequences × 6 thresholds × 3 residual policies), each from a clean initial state. The smoke test is only an integration check; do not treat it as a result.

The output bundle path is printed by the `pack` stage. Send that ZIP back for result analysis. The script cannot run the GPU experiment in this packaging environment; the actual model runs must execute on the user's AutoDL environment.
