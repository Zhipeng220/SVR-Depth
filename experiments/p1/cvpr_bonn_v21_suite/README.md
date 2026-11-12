# CVPR v2.1：Bonn 单来源实验执行器

本包依据用户提供的《SVR-Depth CVPR主实验执行计划 v2.1》（文件 SHA-256：`769cc8e889a6e6ef5e6cb603ff4fb62da835ef275a69c66e0014cb13a40e3336`）开发，按当前要求先处理 Bonn。**第二数据来源暂缓，整个 v2.1 主实验尚未完成。** 本包未在真实服务器/GPU 上验证；先运行下方开发 smoke，失败时保留日志，不得绕过校验转入正式测试。

## 实现范围与仍缺内容

|部分|本包实现|未自动完成或尚待核验|
|---|---|---|
|M01|Bonn 的 A、O_fast、O_multi、P；K16，三种子输入；完整 RGB 状态传播、GT 固定 mask、逐帧预测和风险|独立 Bonn 测试录像尚未确定；O_fast 权重与额外训练种子尚未确认；不自动训练或证明数据独立|
|M02|预先指定各方法 K16/32/64，固定 P 主点；可选 P-Alt；与 M01 相同的 K16 轨迹物理复用；配对风险和计时|不自动进行开发候选扫描或选点，需提供开封前的选点及等预算证据|
|M04|同一个 P checkpoint 的 OFF、OUTPUT_ONLY、F_ONLY、P_NATIVE 各自连续状态；逐帧预测、age 分层、实际调用计数与端到端计时；P_NATIVE 与 M01 P 全轨迹回放|需先提供实际状态字段审计；现有 smoke 不是所有状态/平台的证明。OUTPUT_ONLY/F_ONLY 诊断实现每传播帧重复完整 fast path，因此其计时包含重复计算，不等于优化部署成本|
|质量/成本/分析|原始米制 z-depth、固定范围与 GT mask、完整预测、非有限值/失败记录、相邻帧未 warp 的 GT 深度变化误差；每单元 3 次轮换同步计时，设备端点检查、预测探针、峰值 allocated 显存；按录像及经确认的场景组汇总、配对场景 bootstrap|不含训练实现/训练计算账本、光流 warp 误差、分模块测时、连续功耗曲线或自动论文成功判定；GPU 端点前后干净也不能排除中途竞争|

现有 Y11 五录像、Crowd2/Crowd3/Balloon2 及已查看的 person_tracking、synchronous 属于开发或复核材料；本包禁止以这些名称或已知路径直接声明 heldout，还会逐字节对比 3 条旧训练/开发录像和 5 条 Y11 录像的 RGB。哈希相同会停止；哈希不相同也**不能证明**未被用于其他训练/预训练或物理场景独立，仍须核验。场景组只有在配置中声明已核实时才用于分组或重采样；组数少于 3 不给区间。全部种子结果分别输出，不把帧、种子或重叠窗口当独立场景。

每条非 A checkpoint 必须有真实 SHA-256 和训练记录。P 与 O_multi 的三个对应种子要求在 `matched_training_evidence` 中分别登记 checkpoint SHA、初始化头、样本顺序、监督帧/GT 清单哈希、优化步数、GT 曝光及固定最终 checkpoint 规则；脚本比较两臂字段，**但不能自行证明提供的证据真实**。O_fast 的训练成本、监督数据及选择规则也需在其训练记录中保留。不得将 recovery 文件或相同字节的权重冒充不同种子。已发现的 `cvpr_m05_pmulti_seed_package.zip` 仅训练两个 P 种子，不会自动训练匹配 O_multi 或 O_fast。

## 1. 安装、自检与真实开发 smoke

将 `cvpr_bonn_v21_package.zip` 放在仓库根目录。安装器验证包内哈希，安装到隔离目录；若目标文件存在且内容不同则停止，不覆盖原模型、代码、权重或结果。

```bash
(
set -euo pipefail
cd "$HOME/SVR-Depth/velodepth"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate svrdepth

python -m zipfile -t cvpr_bonn_v21_package.zip
python -m zipfile -e cvpr_bonn_v21_package.zip .
python cvpr_bonn_v21_package/install_cvpr_bonn_v21.py
bash run_cvpr_bonn_v21.sh selftest

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$HOME/hf_cache"
export VDEEP_SNAPSHOT="$HF_HOME/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c"

RUN="outputs/cvpr_bonn_v21_smoke_$(date +%Y%m%d_%H%M%S)_$$"
LOG="cvpr_bonn_v21_smoke_$(date +%Y%m%d_%H%M%S).log"
nohup bash run_cvpr_bonn_v21.sh smoke --out-dir "$RUN" --timing > "$LOG" 2>&1 < /dev/null &
echo "开发 smoke PID：$!"
echo "结果：$PWD/$RUN"
echo "日志：$PWD/$LOG"
tail -n 80 -F "$LOG"
)
```

smoke 固定为 Crowd2 RGB0..39、历史 P/O_multi 各一份权重、A 及 M04 四分支；O_fast 和额外种子不在 smoke 中。它会计算开发前缀的质量并尝试计时，属于开发诊断而非独立测试。GPU 共享锁在 CUDA 初始化前获取；任何帧状态、回放、输入哈希或计时探针不通过均停止或标记失败。`Ctrl+C` 只退出 tail。检测到 GPU 竞争时拒绝相应计时轮次；端点检测不能排除中途短时竞争。

## 2. 生成并完成配置

```bash
cd "$HOME/SVR-Depth/velodepth"
export HF_HOME="$HOME/hf_cache"
export VDEEP_SNAPSHOT="$HF_HOME/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c"
bash run_cvpr_bonn_v21.sh init-config --out configs/cvpr_bonn_v21.json
```

此命令只从可用的原始 Y04/Y11 文件填入已知开发参考清单和经哈希匹配的历史 P/O_multi 候选；不会设置任何审计通过标志，不会添加测试录像、额外种子或开发选点，也不会覆盖已有配置。必须完成以下字段：

1. `role` 为 `development` 或 `heldout`。只有对独立测试确实尚未查看分方法成绩时，才能对 heldout 将 `test_scores_unseen` 设置为 true；Y11 数据始终不能重标为 heldout。
2. 为每条 Bonn 录像填写 `sequence`、绝对路径 `manifest`、`pairing_audit`、`physical_scene_group`、`physical_scene_verified`、`recording_session`、`sensor_model`、`camera_setup`、`audit_notes` 和 `pretraining_overlap`（`unknown`、`documented_overlap` 或 `documented_no_overlap`）。物理场景未知时设 `physical_scene_verified: false`，不得把未知场景假定独立。
3. manifest 每行含 `frame_index_zero_based,rgb_timestamp,rgb_path,depth_timestamp,depth_path,gt_match_status,gt_reused`；索引从 0 连续、RGB 时间严格递增。匹配 GT 时间差 ≤20ms、不复用，未匹配 RGB 仍保留。`MISSING_DEPTH_FILE` 必须保留原来不存在的绝对 GT 路径；来源确证的缺失及原始一对一配对规则由 `pairing_audit` 留证。工具检查匹配一致性，但不会仅凭配对表证明其是原始完整索引上的全局最近邻。
4. 验证 `depth_encoding` 所列 Bonn RGB 已注册的 uint16 z-depth `/5000`；事先填写所有方法一致的 `depth_min_m`、`depth_max_m`，保留标定、resize、失效像素和单位证据。若实际数据不是此格式，禁止修改除数硬跑，需先适配并验证。
5. 在 `checkpoints` 的 P、O_multi、O_fast 各填写三项 `{"seed":"真实外层种子","path":"/绝对路径/权重.pt","sha256":"64位哈希","training_record":"/绝对路径/训练记录.json"}`。P 与 O_multi 种子名逐一对应。旧 Y04 的嵌套 C3 seed 不是 Y04 外层 seed，应以训练记录和样本顺序核验。
6. `matched_training_evidence` 指向填写完成的 JSON，格式见安装目录 `matched_training_evidence.example.json`；必须有 3 个 `pairs`。P/O 的初始化、样本顺序、监督清单哈希、步数和 GT 曝光分别相同。另填 `selection_evidence`、`calibration_evidence`、`independence_evidence`、`m04_state_audit` 的绝对路径；这些文件会被哈希锁定，但其真实性需人工核查。
7. `points` 逐族逐 K 各有一个预选点，包含 `id,family,k,gain,frequency,alignment,role,semantics`。例如主点为 `{"id":"P16","family":"P","k":16,"gain":1,"frequency":1,"alignment":"raw","role":"primary","semantics":"P pre_readout, full state"}`；其他点为 `selected`。A 固定 gain=0、frequency=1。若在测试开封前纳入 P-Alt，可附加 K32/g0.5/f1/raw、`role=alternative`。每族候选数填入 `candidate_counts`，相同且每族最多 12；全部按开发证据预定，不用测试成绩选点。
8. heldout 的 `known_development_manifests` 至少包含 3 条旧训练/开发清单和 5 条 Y11 清单。`init-config` 在现有路径和协议哈希有效时尝试填入它们；未发现的必须补齐。每项为 `{"sequence":"crowd2","manifest":"/绝对路径.csv"}`。脚本核对标识并做 RGB 字节重叠检查，仍需独立审计此前人工访问与底座预训练。

所有布尔审计字段只有在完成对应核验后才能设 true。脚本不从测试质量中自动推断或修改这些字段。

## 3. 协议冻结、完整轨迹、计时和汇总

配置及证据都完成后，再运行下列命令。输出根目录使用时间戳，冻结前无模型推理；冻结会哈希所有参考 RGB、候选 RGB/GT、代码、模型与证据，可能很耗时。`prepare` 遇到缺失种子、证据、数据或重叠即停止。

```bash
(
set -euo pipefail
cd "$HOME/SVR-Depth/velodepth"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate svrdepth
export CUDA_VISIBLE_DEVICES=0

RUN="outputs/cvpr_bonn_v21_$(date +%Y%m%d_%H%M%S)_$$"
PLAN="$RUN/protocol.json"
QUALITY="$RUN/quality_run"
TIMING="$RUN/timing_run"
REPORT="$RUN/report"

bash run_cvpr_bonn_v21.sh prepare \
  --config configs/cvpr_bonn_v21.json --out "$PLAN"

LOG="cvpr_bonn_v21_$(date +%Y%m%d_%H%M%S).log"
nohup bash -c '
  set -euo pipefail
  bash run_cvpr_bonn_v21.sh quality --plan "$1" --out-dir "$2"
  bash run_cvpr_bonn_v21.sh timing --plan "$1" --quality-dir "$2" --out-dir "$3"
  bash run_cvpr_bonn_v21.sh report --plan "$1" --quality-dir "$2" --timing-dir "$3" --out-dir "$4"
' _ "$PLAN" "$QUALITY" "$TIMING" "$REPORT" > "$LOG" 2>&1 < /dev/null &
echo "PID：$!；结果：$PWD/$RUN；日志：$PWD/$LOG"
tail -n 80 -F "$LOG"
)
```

质量阶段保留每个 RGB 的压缩预测与哈希，磁盘消耗可能很大；检查剩余空间。单元失败会保留 `failure.json` 并继续其他单元，最终标记 `QUALITY_INCOMPLETE`；后台流水线随后停止。如果只有质量结果（包括失败），可手动生成不含计时的部分报告：

```bash
bash run_cvpr_bonn_v21.sh report --plan "$PLAN" --quality-dir "$QUALITY" --out-dir "$RUN/quality_report"
```

注意上面变量在不同终端需要重新设置。当前没有中断恢复：若重跑须使用新目录，保留旧失败日志和实际开销；不得因成绩选择性重跑、删录像或调整工作点。完整质量和计时结束也只说明本包阶段完成，不会自动声称实验成功或完整 v2.1 主实验已完成。
