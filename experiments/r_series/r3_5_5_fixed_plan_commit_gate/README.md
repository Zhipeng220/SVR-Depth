# R3.5.5 固定刷新计划下的候选后写回门控验证

本实验是一次有止损条件的最终验证。它复用现有 R3.3/R3.2/R2-A 实现，不覆盖旧实验。

## 本轮固定内容
- 动作锚点：零基帧 `5,17,29,41,49,61`，每条序列 6 个。
- 唯一特征：`raw_rel_median`。
- `AlwaysCommit` 轨迹上采集几何特征和同状态配对反事实收益。
- 按序列留一法训练阈值，阈值规则为 `COMMIT iff raw_rel_median >= threshold`。
- 只有 OOF 局部效用比 AlwaysCommit 在至少两条留出序列上更好，且汇总增量为正，才启动完整门控闭环。
- 闭环比较通过条件：至少两条序列 AbsRel 改善、三序列宏平均改善、任一序列相对退化不超过 2%。
- 候选推理发生在门控前，不声称省算。

## 依赖
要求项目内已有：
- `experiments/r_series/r2a_commit_enum/`
- `experiments/r_series/r3_2_reset_aware_drift_gate/`
- `experiments/r_series/r3_3_geometry_gain_diagnostic/`

把本目录上传到项目根目录即可，不覆盖这些依赖。

## AutoDL 运行
```bash
cd /root/autodl-tmp/SVR-Depth/velodepth
conda activate svrdepth
bash experiments/r_series/r3_5_5_fixed_plan_commit_gate/run_r355.sh start
bash experiments/r_series/r3_5_5_fixed_plan_commit_gate/run_r355.sh preflight
bash experiments/r_series/r3_5_5_fixed_plan_commit_gate/run_r355.sh smoke
```
Smoke 通过后，运行：
```bash
bash experiments/r_series/r3_5_5_fixed_plan_commit_gate/run_r355.sh full
```
`full` 会先运行三条序列的 AlwaysCommit 参考轨迹及配对反事实，然后做 LOSO 离线止损筛选。筛选未通过时，自动停止，不运行门控闭环；通过时，才运行各折冻结阈值的门控轨迹并生成闭环对照。
最后：
```bash
bash experiments/r_series/r3_5_5_fixed_plan_commit_gate/run_r355.sh pack
```

## 输出
- `full/collect/<sequence>/r355_action_geometry_gain.csv`
- `full/collect/<sequence>/r355_online_frame_metrics.csv`
- `full/calibration/r355_loso_thresholds_and_utility.csv`
- `full/calibration/r355_oof_action_decisions.csv`
- `full/calibration/r355_gate_decision.json`
- 若离线筛选通过：`full/gate/<sequence>/r355_online_frame_metrics.csv`
- 若离线筛选通过：`full/comparison/r355_closed_loop_comparison.csv`
- `R3_5_5_离线门控筛选报告.md` 和 `R3_5_5_闭环对照报告.md`

## 解释限制
- H=16 的未来窗口重叠，动作锚点不是独立样本。
- 只有三条开发序列；通过后仍需未参与设计的独立序列。
- 离线局部收益不能替代完整闭环结果。
