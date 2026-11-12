# Bonn M01 单独执行器（开发候选，未在目标机器验证）

该入口只运行 A、O_fast、O_multi、P 在 K16、逐帧、raw 的工作点：A 一个、三个可训练方法各三个种子，合计每条录像 10 条轨迹。不执行 M02 曲线、M04 分支或 M05；M01 在科学协议上不依赖 M02/M04/M05 的实验结果。保留原训练与模型源码校验、非空真实权重 SHA、已知 8 份开发清单的 RGB 字节排重、全量 RGB 与固定 GT mask、预测留档、失败记录、3 次旋转计时和场景组验证。不可把已看过的开发录像变成 heldout。时间指标不等于完整能量或训练计算成本；未实现光流 warp 时间指标和分模块计时；本包未自动判定 v2.1 成功/失败。

## 输入与非盲说明

附带 `config.candidate.DRAFT.json` 是基于用户已提供记录生成的草稿，不自动创建在配置目录或改为通过。九个 checkpoint 与五条候选清单路径/哈希需要在目标服务器重新核对；证据文件路径、缩放及统一深度范围、录像组与相机信息、选择证明、开发重叠和冻结标志仍未完成。不要编造信息来通过门槛。

支持两种明确语义：`role=heldout` + `access_history_status=verified_unseen` + `test_scores_unseen=true` 只能用于有证据确认未查看成绩的录像；`role=heldout_history_unverified` + `access_history_status=unknown` + `test_scores_unseen=false` 表示已知参考无重叠但访问历史仍不确定，其报告将标记 `HISTORY_UNVERIFIED_NOT_BLIND_VERIFIED`，不能声称已验证盲测，也不能称完全满足 v2.1 的盲测要求。两种角色均仍须事前冻结工作点、已知开发数据逐字节排重、数据格式及单位/M00 校准和用户确认来源；只换角色不能绕过这些检查。若用户仍希望保留将来作为验证盲测的可能，不应对它们执行质量阶段。

`calibration_evidence`、`independence_evidence`、`selection_evidence` 是已有真实记录的绝对路径；分别应说明当前 Bonn 与模型 z-depth/单位/注册/resize/mask/深度范围、访问与8份参考/预训练及物理场景范围、M01 四工作点在看测试成绩前固定等情况。不得通过测试分数倒选范围、录像、工作点或阈值。物理场景不可证实时，`physical_scene_verified=false`，并在 `physical_scene_group` 填一个明确表示未知的占位（不是声称这些录像独立）；代码不会生成场景组置信区间。若历史始终无法确定，就如实报告该限制。

## 缓存训练证据

冻结时读取九个训练摘要、核对 P/O 的训练步数、头初始化、三个配对清单哈希、RGB 前缀及从种子复原的顺序，并重新计算 72 对周期缓存的实际 SHA-256，与源清单交叉验证和比较。20261025 P/O 引用同一路径，明确记录共享而不是独立复制。缓存匹配能支持训练时使用相同缓存表示，包括 GT 张量；不能单凭它证明历史原始 GT 文件字节。冻结记录 `training_cache_evidence` 会列出这一限制；不伪造 `P_supervision_sha256`。此步骤可能读取数十 GB，耗时且占用磁盘带宽；缓存必须保留。

## 使用阶段

先上传 ZIP 至远端项目根目录并安装，然后在旧 Crowd2 开发前缀执行四方法真实 GPU smoke；不访问五条候选录像。smoke 使用已记录的原 P/O_multi 20261025 及重训 O_fast 20261025，不把其当作原历史 Y11 O_fast。原始数据/代码及 hash 变化会停止。测试/训练/读取的来源配置和日志必须保留。实际设备 GPU smoke 尚未运行：不能用模拟测试假装已通过。

```bash
python -m zipfile -t cvpr_m01_bonn_only_package.zip
python -m zipfile -e cvpr_m01_bonn_only_package.zip .
python cvpr_m01_bonn_only_package/install_cvpr_m01_bonn.py
bash run_cvpr_m01_bonn.sh selftest
```

固定 CUDA_VISIBLE_DEVICES 和 VDEEP_SNAPSHOT 后，执行 `bash run_cvpr_m01_bonn.sh smoke --out-dir <新的outputs目录> --timing`。质量/计时共享 GPU 锁只约束使用同一把锁的程序；若其他作业不持锁，则不能保证无并发。smoke 结果不等于 M01。

后续真实推理必须以用户确认的科学边界和完整、真实的审计材料为前提。准备 `prepare --config <配置> --out <新protocol.json>`，再 `quality --plan ... --out-dir ...`、`timing --plan ... --quality-dir ... --out-dir ...`、`report --plan ... --quality-dir ... --timing-dir ... --out-dir ...`。无恢复实现，失败时保留记录并使用新目录；避免根据成绩反复挑选。正式质量阶段会对测试候选运行预测并读取 GT，记录访问。结果仅 Bonn，尚无第二来源。
