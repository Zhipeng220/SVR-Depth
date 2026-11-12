# O_multi 与 P_multi 匹配种子训练器（开发候选 1.0）

本包为 Bonn v2.1 的 P/O_multi 匹配训练补齐 O_multi 路径：复用固定 Y04-Fix3 和 Q03/C3 原始训练实现、同一初始头、24 个多录像周期、相同外层种子和 balanced order、1000 次更新、固定最终 checkpoint。不修改原代码和已有权重；不运行开发/测试质量评估，不自动启动 O_fast 或测试推理。构建训练缓存会执行原模型的训练特征提取，不能称为“没有模型计算”。

**状态：本包只在本地完成纯 Python 合成自检和模拟 CPU preflight，尚未在真实服务器运行 O_multi GPU smoke 或正式训练。** 先用历史 seed 做 3 更新 smoke；该 smoke 产生的任何权重均不能作为正式 checkpoint。若预检或 smoke 失败，保留日志，不更改哈希或跳过匹配。

## 匹配检查与边界

- 训练前需提供已完成 P_multi 的权重 SHA、训练 summary；正式种子还需 P 的 `y04_budget_manifest.json`。核验 P 的 Y04 外层 seed、源码和底座、P 训练配对 CSV、24 slots、初始头、balanced order、1000 步和 P 的 GT exposure；O 的实际缓存监督计数和 P budget 不一致则在训练前停止。
- 现有 P 新种子 runner 的嵌套 C3 `seed` 不一定等于 Y04 外层 seed。应使用 `y04_outer_seed` 及 `y04_sample_order_sha256`，不可用嵌套 seed 冒充。历史 20261025 checkpoint 只允许用于本包 preflight/smoke，不重新训练为额外正式种子。
- 脚本逐个哈希当前训练前缀 RGB 与 GT，训练后再核验。P checkpoint 中的配对 CSV 哈希和 exposure 只能证明其登记的帧路径/索引与数量，**不能单独证明历史 P 训练时图像文件的字节与现在相同**。因此新 O 完成后只产生 `matched_pair_evidence_INCOMPLETE.json`，其 `P_supervision_sha256` 留空。必须独立核查 P 训练时的输入字节/不可变存储证据后，才能生成 Bonn v2.1 冻结器所需的完整匹配证据；禁止直接复制 O 的哈希来绕过。
- 训练缓存与模型更新的时间、峰值 PyTorch allocated 显存、GPU 前后端点及尝试日志会记录；端点检查不能排除中途竞争。缓存建立是 O 独立成本，记录在 `attempts.jsonl`；恢复重算和异常均需报告，突然断电/SIGKILL 可能没有结束事件。
- 使用 Y11 GPU UUID 共享锁；不停止其他进程。preflight 在 GPU 锁前读取 P、原训练清单与 RGB/GT 内容，但不访问测试集、不初始化 CUDA、不训练。训练和 smoke 使用 GPU；每个 seed 独立输出目录。

## 1. 安装、自检、历史 P 的 GPU smoke

将 `cvpr_omulti_seed_package.zip` 上传到服务器仓库根目录：

```bash
(
set -euo pipefail
cd "$HOME/SVR-Depth/velodepth"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate svrdepth

python -m zipfile -t cvpr_omulti_seed_package.zip
python -m zipfile -e cvpr_omulti_seed_package.zip .
python cvpr_omulti_seed_package/install_cvpr_omulti_seed.py
bash run_cvpr_omulti_seed.sh selftest

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$HOME/hf_cache"
export VDEEP_SNAPSHOT="$HF_HOME/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c"
P_DIR="$PWD/outputs/gt4a2_y04_validation_fix3_20261025_225902/training/P_multi"
OUT="outputs/cvpr_omulti_seed_20261025_smoke_$(date +%Y%m%d_%H%M%S)_$$"
LOG="cvpr_omulti_seed_smoke_$(date +%Y%m%d_%H%M%S).log"

nohup bash run_cvpr_omulti_seed.sh train \
  --seed 20261025 --smoke-only \
  --p-checkpoint "$P_DIR/gt4a2_c3_C_final_step1000.pt" \
  --p-sha256 77a86cdf523f776608b6b591d6c6552002b90398e85d00d3a3fe29afe14a0a24 \
  --p-summary "$P_DIR/gt4a2_c3_C_training_summary.json" \
  --out-dir "$OUT" > "$LOG" 2>&1 < /dev/null &

echo "Smoke PID：$!；输出：$PWD/$OUT；日志：$PWD/$LOG"
tail -n 80 -F "$LOG"
)
```

最后需出现 `[SMOKE PASS]`。退出 tail 不终止后台任务。smoke 三个周期各 1 个更新，并非 1000 步或正式方法结果；若失败请先检查输出目录 `failure.json` 与日志，不能通过修改 source hash 或删除失败证据强行继续。

## 2. 新种子 preflight 与正式训练

先确保对应 P 种子的正式训练已完成，取得其 checkpoint、实际 SHA、summary 和 budget。下面以 seed 20261101、P 包约定目录为例；路径若不同应按真实产物修改。脚本不会自动创建 P checkpoint，也不会根据表现选 checkpoint。

```bash
(
set -euo pipefail
cd "$HOME/SVR-Depth/velodepth"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate svrdepth
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$HOME/hf_cache"
export VDEEP_SNAPSHOT="$HF_HOME/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c"

SEED=20261101
P_RUN="$PWD/outputs/cvpr_m05_pmulti_seed_${SEED}"
P_CP="$P_RUN/training/P_multi/gt4a2_c3_C_final_step1000.pt"
P_SUMMARY="$P_RUN/training/P_multi/gt4a2_c3_C_training_summary.json"
P_BUDGET="$P_RUN/y04_budget_manifest.json"
P_SHA="$(sha256sum "$P_CP" | awk '{print $1}')"
OUT="outputs/cvpr_omulti_seed_${SEED}"

bash run_cvpr_omulti_seed.sh train \
  --seed "$SEED" --p-checkpoint "$P_CP" --p-sha256 "$P_SHA" \
  --p-summary "$P_SUMMARY" --p-budget "$P_BUDGET" \
  --out-dir "$OUT" --preflight-only

LOG="cvpr_omulti_seed_${SEED}_$(date +%Y%m%d_%H%M%S).log"
nohup bash run_cvpr_omulti_seed.sh train \
  --seed "$SEED" --p-checkpoint "$P_CP" --p-sha256 "$P_SHA" \
  --p-summary "$P_SUMMARY" --p-budget "$P_BUDGET" \
  --out-dir "$OUT" > "$LOG" 2>&1 < /dev/null &
echo "训练 PID：$!；日志：$PWD/$LOG"
tail -n 80 -F "$LOG"
)
```

如 seed 20261102 的 P checkpoint 也已完成，换 seed 后使用独立输出目录顺序运行；不得并行占用同一 GPU。相同 seed 未完成时，可在同样参数和目录上加 `--resume`，并保留所有尝试日志；如果 `y04_summary.json` 已存在，脚本拒绝覆盖完成记录。原始 `y04` 训练器的恢复规则仍适用，未承诺未提交更新零重算。

输出包括 `preflight.json`、`y04_run_identity.json`、`training_input_sha256.json`、`supervision_manifest.csv`、`cache_manifest.json`、`attempts.jsonl`、`training/O_multi/` 及最终 `matched_pair_evidence_INCOMPLETE.json`。完成标记为 `TRAINING_ONLY_COMPLETE_INPUT_PROVENANCE_PENDING`，不能直接填入正式三种子匹配证明。O_fast 三种子训练、独立 Bonn 测试清单和完整 v2.1 主实验仍需另行完成。
