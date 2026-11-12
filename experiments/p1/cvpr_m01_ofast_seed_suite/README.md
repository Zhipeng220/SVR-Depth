# M01 O_fast 多种子训练（开发候选）

复用已审查的 Y09 O_fast RGB/depth adapter、24 个开发周期、原有固定 20261025 训练顺序、3 步 smoke / 1000 步正式训练，以及原有恢复实现；不同的 20261025、20261101、20261102 初始化种子产生不同训练运行。三个 O_fast 种子固定共用原历史训练顺序，并非随机种子共同改变数据顺序；P/O_fast 不声明同初始化或逐帧证明。仅训练 O_fast，不跑 B2，不开封测试。初始化等信息由训练 summary/协议记录。

本包只经过本地模拟测试，尚未在目标机器运行。先运行 smoke，完整 smoke 会建立 3 录像各 8 个训练缓存周期，可能耗时和占用磁盘；3 次优化更新仅作开发验证，产生的 final.pt 不可当正式权重。缺少历史 O_fast 权重时，20261025 是重新训练的种子，不能声称与丢失的历史 Y11 权重相同。此脚本不能自动完成训练成本的汇总审计、匹配原始 P 输入证明、O_fast 开发选点或独立 Bonn 测试。

标准训练使用原始 Y09 preflight 及其 Y04-A 依赖；输入变化、源码变化、恢复身份不符均停止。GPU UUID 共享锁在 CUDA 初始化前获取。每次启动与失败记录写 attempts.jsonl 与 failure_*.json；原 Y09 recovery 保存训练状态并记录可能重算，输入与代码在成功前再校验。共享锁只协调采用同一锁的进程。GT 仅用于训练缓存与损失；先完成开发证据和独立测试冻结再运行正式评价。

安装（ZIP 放在仓库根目录）：

```bash
python -m zipfile -t cvpr_m01_ofast_seed_package.zip
python -m zipfile -e cvpr_m01_ofast_seed_package.zip .
python cvpr_m01_ofast_seed_package/install_cvpr_m01_ofast_seed.py
bash run_cvpr_m01_ofast_seed.sh selftest
```

GPU smoke：先设置 CUDA_VISIBLE_DEVICES、HF_HOME、VDEEP_SNAPSHOT，然后：

```bash
OUT="outputs/cvpr_m01_ofast_20261025_smoke_$(date +%Y%m%d_%H%M%S)_$$"
bash run_cvpr_m01_ofast_seed.sh train --seed 20261025 --smoke-only --out-dir "$OUT"
```

smoke 通过后，使用三个独立新目录逐个训练，不能并行占用同一 GPU：

```bash
bash run_cvpr_m01_ofast_seed.sh train --seed 20261025 --out-dir outputs/cvpr_m01_ofast_seed_20261025
bash run_cvpr_m01_ofast_seed.sh train --seed 20261101 --out-dir outputs/cvpr_m01_ofast_seed_20261101
bash run_cvpr_m01_ofast_seed.sh train --seed 20261102 --out-dir outputs/cvpr_m01_ofast_seed_20261102
```

中断恢复仅对未完成的原输出目录和原种子附加 `--resume`，状态校验不同即停止；保留失败和恢复日志。输出 `training/O_fast/final.pt`、`training/O_fast/summary.json`、`protocol.json`、`m01_ofast_status.json`、`attempts.jsonl` 与缓存成本记录。final.pt 包含 state 字段，属于 O_fast，不可与 O_multi 混用。
