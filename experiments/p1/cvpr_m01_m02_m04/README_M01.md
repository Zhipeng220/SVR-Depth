# M01 独立数据主实验：冻结与汇总组件

本目录依据主实验计划 v2.1 与 N14-A v2 的独立包风格制作。**当前仅实现协议冻结与部分汇总，不含 M01 推理、训练、计时采集器，也不能直接完成 M01。** 不下载数据、不修改模型或旧结果。服务器原仓库、P/O_fast/O_multi 的三个真实 checkpoint、已审计的独立数据、M00 深度及坐标契约和运行环境均未提供；未进行真实 GPU smoke、正式实验或得到任何实验结果。尤其不能以 N14 的 C* 替代 P_multi，也不能把 Y11 开发录像作为独立测试。

## 1. 事前准备

在服务器复制本目录至仓库，复制 `config.example.json` 为 `config.json`。准确填写源文件、骨干模型文件、三个方法各三个 checkpoint 的路径和实际种子，以及所有测试录像的信息；每个训练种子的来源、初始化、监督、训练步骤、选择规则另行留档。`model_files` 应覆盖模型配置及权重；`source_files` 至少覆盖实际流式实现、数据读取/配对、预测后处理、指标与计时代码。仅当人工完成 M00、独立性审计及测试尚未开封时，才设置两个确认字段为 true。`test_unsealed: false` 表示冻结之前尚未查看测试方法成绩；不能通过修改字段把已查看的数据重新当作盲测。各数据源的相机注册、深度单位和合法范围须单独确认；scale 仅留档，本组件不会转换深度。

`sequences` 每项例如（路径为服务器绝对路径）：

```json
{
  "dataset": "Bonn",
  "sequence": "recording_id",
  "physical_scene_group": "verified_scene_id",
  "camera_setup": "documented_setup",
  "sensor_model": "documented_sensor",
  "recording_session": "documented_session",
  "independence_notes": "记录与训练、开发、预训练数据的核查结果；未知须明确写未知",
  "depth_scale_to_meters": 0.0002,
  "manifest": "/absolute/path/pairs.csv"
}
```

每个 `checkpoints` 条目如 `{"seed": "真实种子", "path": "/absolute/path/final.pt"}`；方法各三个不同 seed。每条 manifest 包含 `frame_index_zero_based,rgb_path,gt_match_status,depth_path`，index 从 0 连续，状态为 `MATCHED` 或 `UNMATCHED`。所有 RGB 均推进状态；匹配 GT 只用于评估。路径应为绝对路径。文件逐个 SHA-256，运行前重新核验；哈希可能耗时。脚本不证明用户填报的独立性、标定或权重训练身份，也不验证 P/O_multi 训练匹配；这些必须人工留证。

```bash
cd "$HOME/SVR-Depth/velodepth"
python M01_package/m01.py freeze --config M01_package/config.json --out outputs/m01_frozen/protocol.json
```

冻结输出不覆盖旧文件。先在开发集做模型和深度 smoke，然后冻结正式测试清单；任何修正须新建版本，记录原因、开封状态与受影响的全部运行，不能按测试效果选择或剔除数据。

## 2. 推理接口待接入

正式推理必须对 A、O_fast、O_multi、P 分别以 K16 从每个序列 RGB0 完整运行；适配头采用对应冻结权重，P 每传播帧写回，gain=1，raw 米制 z-depth。继承 Y11 的 points=radius×rays 后处理，不单独插值 radius/rays。统一 GT mask、逐帧记录失败；所有方法对每个 RGB 输出一行。由于当前缺少服务器实际 `y08y09_models.py`、Y11 数据接入和三种子 checkpoint 格式，尚未实现或声称已验证此部分。N14-A v2 的 `Predictor` 仅是参考，其 C* 和 K32 默认不能直接当 M01 执行器。缺少新种子时不得复制权重冒充不同种子。

采集的 `frames.csv` 表头：

```text
sequence,method,seed,index,status,mask_sha256,valid_pixels,AbsRel,RMSE,delta1
```

method 为 `A,O_fast,O_multi,P`；A 的 seed 固定 `native`，其他填协议中对应 seed。每个组合必须覆盖全部 RGB。`status` 为 `ok`、`unmatched`、`invalid_gt` 或 `invalid_prediction`；后两者记录失败，不能悄悄跳过或更换 mask。匹配且 GT 有效的帧需保存相同 mask 的 SHA-256 和有效像素数；`ok` 时提供三个有限数值。原始深度预测、完整逐帧输出和日志另行留存以支持 M03；本组件不生成预测。

`timing.csv` 表头：

```text
sequence,method,seed,repeat,verified,mean_ms
```

每组合三次不同 repeat，轮换顺序；只有经过 GPU 无竞争和质量轨迹核验的计时写 `verified=true`。按计划包含暖文件缓存 RGB 读取/解码、预处理、光流、完整模型、D2H 和同步；排除 GT、指标、写盘、模型加载。完整 p50/p95、显存、调用账本、硬件、暖机、吞吐和失败轮次需要另行记录，本汇总脚本尚未计算它们。

```bash
python M01_package/m01.py report \
  --protocol outputs/m01_frozen/protocol.json \
  --frames outputs/m01_run/frames.csv \
  --timing outputs/m01_run/timing.csv \
  --out outputs/m01_run/analysis
```

输出逐录像、逐物理场景组、来源内场景组等权汇总，以及逐录像 P 对 O_fast 所有种子组合的直接帧级差异；种子分别列示，不把种子或帧当成独立场景。每组均报告匹配和失败数量；风险指标为有效配对帧上相对 A 的 harmed >0、severe >0.01，另外给 P−O_fast >0.01。**存在失败时这些是条件于有效配对帧的描述值，不能忽略失败作完整比较。** 此版本尚无场景组 bootstrap、录像等权敏感性表、时间深度变化、GT 范围/mask 实际实现及主判断区间，故 `summary.json` 固定标记 `PARTIAL_ANALYSIS`，不能宣称满足 M01 全部交付。

## 3. 本地测试

```bash
python -m unittest discover -s M01_package -p 'test_*.py' -v
```

测试仅使用合成数据检查冻结与配对汇总，不代表服务器接口、实际数据、深度单位或 GPU 性能已验证。
