# R3.4 候选几何特征增量验证

R3.4 是一个**离线分析实验**：复用 R3.3 已生成的 `r33_geometry_gain_diagnostics.csv`，不运行 GPU、不生成新候选、不改动 R3.3 结果，也不训练神经网络。

## 研究问题

在按序列留一法（LOSO）下，`raw_rel_median` 与 `fit_residual_norm_median` 是否能在 `distance_kf` 的 AgeOnly 基线之外，改善局部写回收益预测、负收益识别和局部反事实效用？

## 固定协议

- 数据：R3.3 的 Crowd2、Balloon2、Crowd3 几何/收益 CSV。
- 验证：每次留出一条完整序列，另两条序列用于拟合；不随机拆分锚点。
- 模型：固定 L2 Ridge，alpha=1.0；训练折内标准化特征与收益，训练折中位数填补缺失值。
- 候选：AgeOnly、RawMedianOnly、FitMedianOnly、AgePlusRawMedian、AgePlusFitMedian、AgePlusBothMedians。
- 对照：MeanOnly、AgePlusRawMeanControl、AgePlusFlowControl。
- 决策阈值：预测收益 > 0 则 COMMIT，否则 DISCARD；不在测试折调阈值。
- 主要评测：全部有效诊断锚点的 out-of-sequence 预测；次要评测：9 个真实策略动作锚点。
- 限制：只有 3 条开发序列，未来 H=16 窗口重叠；离线效用不等于完整闭环收益。

## AutoDL 运行

```bash
cd /root/autodl-tmp/SVR-Depth/velodepth
conda activate svrdepth
bash experiments/r_series/r3_4_geometry_incremental_validation/run_r34.sh preflight
```

先定位 R3.3 结果 ZIP：

```bash
find outputs/r_series/r3_3_geometry_gain_diagnostic -maxdepth 1 -name '*_review.zip' -print
```

然后将实际路径传入分析命令，例如：

```bash
bash experiments/r_series/r3_4_geometry_incremental_validation/run_r34.sh analyze \
  outputs/r_series/r3_3_geometry_gain_diagnostic/run_20261022T042558Z_review.zip
```

若结果 ZIP 不在上述路径，可传绝对路径。分析是 CPU 离线任务，不需要 `start`、GPU 或重新运行 R3.3。

输出在 `outputs/r_series/r3_4_geometry_incremental_validation/run_<UTC时间>/`：
- `r34_loso_predictions.csv`
- `r34_loso_metrics.csv`
- `r34_incremental_comparisons.csv`
- `r34_dataset_audit.csv`
- `R3_4_候选几何特征增量验证报告.md`
- `r34_metadata.json`

打包当前结果：

```bash
bash experiments/r_series/r3_4_geometry_incremental_validation/run_r34.sh pack
```

## 解释规则

- MAE/RMSE 比 AgeOnly 更低，才支持连续收益预测增量。
- 负收益召回率高但错误拒绝大量正收益，也不能单独判为成功。
- AgeGate 比较采用 `distance_kf >= 16` 的显式规则；局部效用只在各诊断锚点上模拟，不是完整在线策略结果。
- 真实动作锚点仅 9 个，只能作方向性检查。
- 若没有稳定增量，停止门控特征堆叠，不进入 R3.5。
