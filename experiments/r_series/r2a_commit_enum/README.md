# 下一步实验：R0 + R2-A/K3 固定刷新时刻的写回枚举

## 研究问题

固定刷新候选计算时刻以后，选择哪些候选写回持久几何状态，是否比全部写回、全部丢弃更好？
本轮只研究四字段 GEOM_ONLY 写回。主模型权重冻结，不拟合预测器，不开展空间刷新。

## 实验设计

- K=3，枚举 000、001、010、011、100、101、110、111 共 8 条实际递推轨迹。
- 0：使用当前刷新候选输出，丢弃其持久状态，沿传播状态继续。
- 1：使用同一语义的刷新候选输出，把 features_teacher、radius_teacher、confidence_teacher、rays_base 写入传播状态。
- 每个候选从当前轨迹自己的传播前快照计算。只在同一初始状态的 Commit/Discard 对照中要求当前输出一致，不要求已分化的各轨迹输出一直一致。
- 后续帧沿当前轨迹实际生成的状态递推，不拼接离线标签或默认轨迹。
- 几何写回不改变调度字段或 fast-tail，不重置关键帧年龄。
- 刷新候选沿用原生 keyframe_step 的 overlap 语义，不强制设置 first_keyframe=True。这与完整关键帧重置是不同的动作。
- 原生重启优先：传播请求重启时，同帧执行 keyframe；若与固定刷新时刻冲突，该时刻没有额外 Commit/Discard 决策。不把它强行计为一次可选刷新。报告 K_requested、实际可选候选数、冲突数和原生重启数。
- 若各轨迹的原生重启或候选执行数不同，不能宣称等实际计算成本。本轮主要测量状态管理价值。

K=3 的枚举给出的是该固定时刻表、该写回模式和原生重启规则下的最佳已评估轨迹。它不等于所有刷新时刻、所有策略的全局最优，也不可部署。

选择 K=3 是为了用 8 个计划检查真实决策空间，并避免有限 H 贪心门控被误称为完整序列上界。

## 安装与运行

本项目已将 R2-A 安装到 experiments/r_series/r2a_commit_enum/；远程运行前请通过 PyCharm 上传新目录。
R2-A 使用独立入口，但仍依赖 experiments/p1/ 中经过审计的历史辅助代码；不移动或覆盖 P1。

```bash
cd /root/autodl-tmp/SVR-Depth/velodepth
python3 -m zipfile -e r2a_commit_enum_bundle.zip .
bash experiments/r_series/r2a_commit_enum/run_r2a.sh preflight
```

preflight 只使用 CPU 和 Python 标准库，检查历史已审计的四个源码文件哈希与必要 helper 是否存在。哈希允许 CRLF/LF 差异。若源码已改变，会在 GPU 推理前退出，并自动打包当前实际源码供适配，不提供跳过审计的开关。

```bash
bash experiments/r_series/r2a_commit_enum/run_r2a.sh smoke
```

smoke：Crowd2 前 32 帧，零基刷新时刻 5、13、21。先做 R0 真实状态隔离和恢复重放检查，再运行 8 条轨迹。这是接口检查，不作泛化结论。

R0 通过且输出没有异常以后，再执行：

```bash
bash experiments/r_series/r2a_commit_enum/run_r2a.sh full
```

full：Crowd2、Balloon2、Crowd3 各前 86 帧；零基刷新时刻固定为 9、37、65；每序列 8 条轨迹。三个序列均属于开发诊断，不作为全项目从未见过的最终测试集。

脚本优先使用 /root/miniconda3/envs/svrdepth/bin/python。也可显式设置：

```bash
SVR_PYTHON=/实际环境/bin/python bash experiments/r_series/r2a_commit_enum/run_r2a.sh smoke
```

模型加载强制离线，复用现有缓存权重。若缓存不可用，会报错，不会自动下载模型。

## 自动输出

每次在 outputs/r_series/r2a_commit_enum/ 创建独立时间戳目录，保留旧结果。成功或普通运行异常均会生成同级 *_review.zip；命令行会打印其绝对路径。若发生 OOM/进程被系统终止，请保留已写出的目录与日志。

主要文件：

| 文件 | 内容 |
|---|---|
| source_audit.json | 已核对源码版本及匹配状态 |
| r0_runtime_audit.json | 分支无共享张量存储、四字段写回差异、同状态下一帧重放差异 |
| frame_metrics_000.csv 等 | 每条轨迹逐帧质量、三时钟、实际动作、重启、计时 |
| trajectory_summary.csv | 8 条轨迹的完整片段 AbsRel、RMSE、delta1 与实际计数 |
| paired_commit_effects.csv | 仅改变一个写回位的计划对照，未来计划控制相同，H=16 折扣收益及有效帧数 |
| diagnostic_headroom.json | 最佳枚举计划相对更好无条件基线的改善 |
| metadata.json | 运行状态、协议、GT 覆盖、设备和计时边界 |
| failure.txt | 失败原因；仅异常时存在 |

H=16 配对收益使用原始未来步编号的 gamma**k，gamma=0.9；缺失 GT 不填零，不重新编号。32 帧 smoke 尾部窗口会截短，显式报告有效数。完整序列指标是有效帧指标的算术均值，与“所有像素合并”的指标不同。

## 看什么结果

1. R0 必须先通过：同状态恢复重放差异 <=1e-6，Commit/Discard 张量不共享存储。当前输出一致来自共享同一次候选输出，不把这项构造性一致当作精度证据。
2. 三时钟按真实事件更新。Discard 不重置年龄；GEOM_ONLY 只重置几何提交年龄；初始化和原生重启重置两种年龄。
3. 检查所有计划的有效指标帧一致。若 mask 不一致，程序报错，不比较不同分母的均值。
4. 比较 best_enumerated_absrel 与 min(always_commit_absrel, never_commit_absrel)。更好的参照是两个无条件策略中的较好者。
5. 同时检查 RMSE/delta1、原生重启数、实际候选数、冲突数。若主要指标改善但其他指标明显恶化，保留权衡，不能直接宣布全面胜出。
6. 若 8 条计划几乎相同，先检查写回张量差异、重启覆盖和时刻冲突。该固定时刻表没有空间，不等于所有预算和动作定义都没有空间。
7. 若混合写回计划在多个开发序列均有实质改善，下一步再比较廉价可部署门控，不立即扩大预测器。

可在查看结果前采用“相对较好无条件基线的 AbsRel 改善至少 2%，并在至少 2/3 开发序列出现”作为初步工程筛查线。这个 2% 是建议的项目阈值，不是统计显著性标准，也不是投稿标准；最终按任务需求预先确定阈值。

## 验证与限制

- 本包依据已上传 p2_3_v5_23_source_review.zip 中的真实回放/流式接口编写；已执行 CPU 状态机、收益配对、有效帧一致性等测试。
- 当前环境未运行真实 CUDA/VeloDepth 推理，首轮必须从 preflight 和 smoke 开始。
- 保留旧 helper 的 RGB/GT 匹配、深度还原和评测逻辑；其实际源码会随 review ZIP 保存。未审查 helper 的全部历史实现，不能把源码哈希检查等同于完整模型正确性证明。
- 输入与光流预先构建；按轨迹记录的计时包含实验快照与指标计算等诊断开销。不是部署延迟，也不报告端到端 FPS 或加速倍数。
- review ZIP 包含当前相关源码、配置、结果，不包含模型权重、原始 RGB、深度数组或数据集。
