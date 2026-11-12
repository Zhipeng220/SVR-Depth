# R3.3 候选几何差异分解与写回收益诊断

这是一个新增、独立的实验目录，不覆盖 R3.2、R2-A 或 P1。

## 运行顺序（AutoDL）

```bash
cd /root/autodl-tmp/SVR-Depth/velodepth
conda activate svrdepth
bash experiments/r_series/r3_3_geometry_gain_diagnostic/run_r33.sh start
bash experiments/r_series/r3_3_geometry_gain_diagnostic/run_r33.sh preflight
bash experiments/r_series/r3_3_geometry_gain_diagnostic/run_r33.sh smoke
```

Smoke 通过后运行三序列完整诊断：

```bash
bash experiments/r_series/r3_3_geometry_gain_diagnostic/run_r33.sh full
bash experiments/r_series/r3_3_geometry_gain_diagnostic/run_r33.sh pack
```

`start` 只运行一次。后续阶段都复用最新运行根目录。可通过 `R33_ROOT=/absolute/path/to/run_root` 显式指定运行目录。

## 本轮不是策略搜索

R3.3 在实际动作锚点 9、37、65 只执行 AgeGate；其他每 4 帧诊断锚点只额外计算候选、记录几何差异并做同状态未来分支，然后丢弃候选，不改变主轨迹。额外诊断候选与分支会产生额外计算，因此结果不用于声称省算或部署延迟。

## 上传复核

`pack` 会输出 `..._review.zip`。上传该 ZIP，重点分析 `r33_geometry_gain_diagnostics.csv` 和 `r33_feature_gain_association.csv`。
