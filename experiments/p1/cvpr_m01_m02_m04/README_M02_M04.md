# M02 / M04：协议冻结与结果检查组件

本包继续使用 M01 的独立数据协议，依照 CVPR 主实验计划 v2.1 定义 M02 预选工作点及 M04 同头连续反馈对照。**本版不是完整实验执行器**：不含模型接入、训练、预测或 GPU 计时，不能单靠本包运行正式 M02/M04。缺少服务器仓库的流式模型、状态读写、权重加载、GT 后处理和新增种子，不能凭 N14-A v2 的 C* 实现替代，也不能把合成测试称作实际实验。未取得实验结果。

## 前置条件

先完成 M00，使用 M01 的冻结脚本生成尚未开封的 `protocol.json`。其中需要经过人工审计的独立数据、深度契约、真实权重和来源代码哈希。开发集的 M02 选点证据、候选及选择规则须在测试开封前留档。三种子需真实训练记录，P/O_multi 的训练匹配须单独核验。脚本只核对声明与文件哈希，不证明独立性、模型语义或训练匹配。M02、M04 必须在测试开封之前分别冻结。

## M02：K16/32/64 固定配置

复制 `M02_config.example.json` 为 `M02_config.json`。每方法声明相同的开发候选预算（1–12；这里按计划保守地要求计数完全相等）及其证据文件；每个方法每个 K 指定恰好一个开发期选出的 `primary`/`selected` 点。P K16 的主点固定 K16/g1/f1/raw；可另加一个 `alternative` 点 K32/g0.5/f1/raw。除 P 主点与可选替代点外，不强制不同家族共享参数名称或含义。`parameter_semantics` 应记录实际配置含义，服务端需附完整参数与实现映射。

示例点（完整清单必须包括 A、O_fast、O_multi、P 各自 K16、32、64，至少 12 点）：

```json
{"id":"P_K16", "family":"P", "k":16, "role":"primary", "gain":1, "frequency":1, "alignment":"raw", "parameter_semantics":"每传播帧 pre_readout 修复并按原生 P 提交状态"}
```

将完成核验的布尔字段设为 true；`test_unsealed: false` 表示测试尚未开封，而不是程序自动证明盲测。`development_selection_evidence` 应包含开发集、每族候选、预算、确定的选择规则和结果，不得包含测试成绩。

```bash
cd "$HOME/SVR-Depth/velodepth"
python M02_M04_package/m02_m04.py freeze \
  --config M02_M04_package/M02_config.json \
  --m01-protocol outputs/m01_frozen/protocol.json \
  --out outputs/m02_frozen/protocol.json
```

M01 的 K16 结果原则上应复用。复用之前须逐项核验 checkpoint、模型源码、输入、状态、配置、预测、计时范围及哈希一致；本工具尚不自动导入或验证 M01 预测，因此不会自动宣称复用。其他点按冻结协议完整流式处理，不在测试集重选前沿，不以跨协议插值宣称同质量加速。

## M04：同头连续轨迹

复制 `M04_config.example.json` 为 `M04_config.json`，经 M00 状态字段审计后填写确认。每个 P 种子以同一 checkpoint 分别完整运行以下 4 个分支；各自保留独立状态历史：

| unit | 当前输出 | 持续状态 |
|---|---|---|
| OFF | 原生 D0 | r0、c0、F0、rays 及其他原生缓存 |
| OUTPUT_ONLY | 修复读出 D1 | 保留全部原生状态，含 r0、c0、F0、rays |
| F_ONLY | 原生 D0 | 仅特征使用 F1；r0、c0、rays 与其他缓存为原生 |
| P_NATIVE | 实际 P 的 D1 | 按真实 pre_readout 路径提交 r1、c1、F1、rays 和其他缓存 |

上表为目标契约，**尚未接入或验证实际模型实现**。必须逐字段检查其他缓存、dtype、读写顺序以及 P_NATIVE 与真实 P 的逐帧回放；OUTPUT_ONLY 和 F_ONLY 所需额外读出计入真实成本。四条闭环的历史会分叉，不能将其差异称为同父状态的单步因果效应；同父状态机制问题属于 M05。F_ONLY 不是直接对既有 P 路径简单地关闭输出。

```bash
python M02_M04_package/m02_m04.py freeze \
  --config M02_M04_package/M04_config.json \
  --m01-protocol outputs/m01_frozen/protocol.json \
  --out outputs/m04_frozen/protocol.json
```

M04 的 P_NATIVE 与 M01 P 若满足完整轨迹、计时和配置一致性，可在服务端校验后复用；本工具不自动验证复用。P/O_multi 的匹配训练为额外比较，应保留 M01 的训练来源与成本，本工具的四臂输出不替代该比较。

## 外部执行器的输入与汇总

M02/M04 每个冻结 `unit`、每个种子、每条录像从 RGB0 流式完整运行，丢失 GT 的 RGB 仍传播；保存逐帧预测用于后续分析。外部执行器需输出 `frames.csv`：

```text
sequence,unit,seed,index,status,mask_sha256,valid_pixels,AbsRel,RMSE,delta1
```

seed 对 A 写 `native`，其他为协议内对应种子；M04 的 OFF 也各自标注 P seed 以保持配对。状态为 `ok`、`unmatched`、`invalid_gt`、`invalid_prediction`；GT 有效时各臂使用完全一致的 mask。任何失败应保留，不得删除不利录像。M02 风险参照同 K 的 A；M04 参照同种子的 OFF。每帧均采用原始米制 z-depth；本工具接收指标，不计算深度、GT mask 或时间指标。

`timing.csv`：

```text
sequence,unit,seed,repeat,verified,mean_ms
```

每个单元须三次不同 repeat，`verified=true` 仅用于经轨迹和无竞争核验的计时。按计划轮换顺序，计入 RGB 读取/解码、预处理、光流、模型、修复、D2H、同步及任何额外读出；保留硬件、峰值显存、关键帧/头/readout 次数、p50/p95、失败与重试日志。本脚本只汇总三次 mean 的中位数，**不**验证计时范围、轨迹、资源竞争或调用数。

```bash
python M02_M04_package/m02_m04.py report \
  --plan outputs/m02_frozen/protocol.json \
  --frames outputs/m02_run/frames.csv \
  --timing outputs/m02_run/timing.csv \
  --out outputs/m02_run/analysis

python M02_M04_package/m02_m04.py report \
  --plan outputs/m04_frozen/protocol.json \
  --frames outputs/m04_run/frames.csv \
  --timing outputs/m04_run/timing.csv \
  --out outputs/m04_run/analysis
```

输出逐录像、逐物理场景组、按来源分开的场景组等权表，逐种子保留；含相对参考的帧级 harmed、severe、AbsRel 差及失败数。每点的估计以该点和参考均有效的帧为分母；不同方法失败时分母可能不同，必须共同报告失败，不应把条件均值误称完整样本结果。M02 固定点的绝对精度、延迟可用于描述；不能将这些表作为重新选择测试前沿的工具。M04 当前报告不能单独识别未来收益；未来质量、相位/链长、置信区间、时序误差、调用数、显存和统计推断仍未实现。状态始终标为 `PARTIAL_ANALYSIS`。

## 合成测试

```bash
python -m unittest discover -s M02_M04_package -p 'test_*.py' -v
```

本地测试检查点配置、完整轨迹及 mask 约束，不能验证真实 GPU、模型状态或科学有效性。
