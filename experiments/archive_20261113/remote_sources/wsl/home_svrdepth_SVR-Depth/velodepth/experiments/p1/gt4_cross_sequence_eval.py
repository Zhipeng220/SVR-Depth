#!/usr/bin/env python3
"""GT4: frozen GT2 C/D repair-head cross-sequence evaluation on Bonn sequences.

Uses the exact GT3 inference/preprocessing helpers, but validates checkpoint
provenance against GT2's TRAINING pair manifest while evaluating a distinct
sequence-specific test manifest. No training or GT-conditioned inference.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "p1"))
import gt3_heldout_eval as gt3  # noqa: E402

gt2 = gt3.gt2
DEPTH_SCALE = 5000.0
COHORT = "cross_sequence_holdout"


def read_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    return gt3.sha256_file(path)


def choose_nonoverlapping_valid_clips(by_index: dict[int, dict], max_delta_ms: float):
    """Greedily choose a maximum-cardinality set of disjoint valid 3-frame clips."""
    if not by_index:
        raise RuntimeError("配对清单为空")
    indices = sorted(by_index)
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise RuntimeError("配对清单 RGB 全局索引不连续，停止以免错误地把相邻行当成相邻帧")
    starts = []
    skipped = []
    next_allowed = indices[0]
    for s in range(indices[0], indices[-1] - 1):
        if s < next_allowed:
            continue
        triple = [by_index.get(i) for i in (s, s + 1, s + 2)]
        if any(r is None for r in triple):
            skipped.append({"start": s, "reason": "missing index row"})
            continue
        reason = None
        for row in triple:
            if row.get("gt_match_status") != "MATCHED":
                reason = "one or more frames unmatched to GT"
                break
            if str(row.get("gt_reused", "False")).strip().lower() in {"true", "1", "yes"}:
                reason = "GT reused flag set"
                break
            try:
                dt = float(row["abs_delta_ms"])
            except (TypeError, ValueError):
                reason = "invalid timestamp delta"
                break
            if not np.isfinite(dt) or dt > max_delta_ms:
                reason = f"timestamp delta exceeds {max_delta_ms} ms"
                break
            for col in ("rgb_path", "depth_path"):
                if not row.get(col) or not Path(row[col]).is_file():
                    reason = f"missing file: {col}"
                    break
            if reason:
                break
        if reason is None:
            gt_paths = [str(Path(r["depth_path"]).resolve()) for r in triple]
            rgb_paths = [str(Path(r["rgb_path"]).resolve()) for r in triple]
            if len(set(gt_paths)) != 3 or len(set(rgb_paths)) != 3:
                reason = "duplicate RGB or GT path inside clip"
        if reason is None:
            starts.append(s)
            next_allowed = s + 3
        else:
            skipped.append({"start": s, "reason": reason})
    return starts, skipped


def ensure_disjoint_from_training(eval_rows, train_pair_csv: Path):
    if not train_pair_csv.is_file():
        raise FileNotFoundError(f"GT2 training pair manifest missing: {train_pair_csv}")
    train_rows = read_rows(train_pair_csv)
    train_rgb = {str(Path(r["rgb_path"]).resolve()) for r in train_rows if r.get("rgb_path")}
    train_gt = {str(Path(r["depth_path"]).resolve()) for r in train_rows if r.get("depth_path")}
    eval_rgb = {str(Path(r["rgb_path"]).resolve()) for r in eval_rows if r.get("rgb_path")}
    eval_gt = {str(Path(r["depth_path"]).resolve()) for r in eval_rows if r.get("depth_path")}
    rgb_overlap = sorted(train_rgb & eval_rgb)
    gt_overlap = sorted(train_gt & eval_gt)
    if rgb_overlap or gt_overlap:
        raise RuntimeError(
            f"测试集与 GT2 训练集存在路径重叠：RGB={len(rgb_overlap)}, GT={len(gt_overlap)}；停止。"
        )
    return {"training_rgb_path_overlap": 0, "training_gt_path_overlap": 0,
            "training_pair_csv": str(train_pair_csv.resolve())}


def main():
    ap = argparse.ArgumentParser(description="GT4 frozen C/D cross-sequence evaluation")
    ap.add_argument("--sequence-name", required=True)
    ap.add_argument("--pair-csv", required=True)
    ap.add_argument("--snapshot", default=os.environ.get("VDEEP_SNAPSHOT", ""))
    ap.add_argument("--gt2-dir", default="outputs/gt2_validation")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-time-diff-ms", type=float, default=20.0)
    ap.add_argument("--min-clips", type=int, default=3)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("GT4 需要 CUDA；torch.cuda.is_available() 为 False")
    if not args.snapshot:
        raise RuntimeError("未设置 --snapshot 或 VDEEP_SNAPSHOT")
    if args.max_time_diff_ms <= 0:
        raise ValueError("max-time-diff-ms 必须为正数")

    pair_csv = Path(args.pair_csv).resolve()
    snapshot = Path(args.snapshot).resolve()
    gt2_dir = Path(args.gt2_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not pair_csv.is_file():
        raise FileNotFoundError(f"测试配对清单不存在：{pair_csv}")
    weights_path = snapshot / "model.safetensors"
    config_path = snapshot / "config.json"
    if not weights_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"预训练快照不完整：{snapshot}")
    gt2_summary_path = gt2_dir / "gt2_summary.json"
    if not gt2_summary_path.is_file():
        raise FileNotFoundError(f"GT2 汇总不存在：{gt2_summary_path}")
    gt2_summary = json.loads(gt2_summary_path.read_text(encoding="utf-8"))
    if gt2_summary.get("engineering_status") != "PASS":
        raise RuntimeError("GT2 engineering_status 不是 PASS")

    expected_sources = gt2_summary.get("source_sha256", {})
    decoder_path = ROOT / "velodepth/models/decoder_velo.py"
    head_path = ROOT / "velodepth/models/persistent_feature_repair.py"
    train_script_path = ROOT / "experiments/p1/gt2_fixed_clip_repair_train.py"
    for path, key in [
        (decoder_path, "velodepth/models/decoder_velo.py"),
        (head_path, "velodepth/models/persistent_feature_repair.py"),
        (train_script_path, str(Path("/home/svrdepth/SVR-Depth/velodepth/experiments/p1/gt2_fixed_clip_repair_train.py"))),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"必要源码缺失：{path}")
        expected = expected_sources.get(key)
        # The GT2 summary stores the training script using its absolute path;
        # tolerate a moved checkout by finding the matching basename key.
        if expected is None and path == train_script_path:
            expected = next((v for k, v in expected_sources.items() if Path(k).name == path.name), None)
        if not expected:
            raise RuntimeError(f"GT2 summary 缺少源码哈希：{key}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"源码哈希不匹配，拒绝使用旧检查点：{path}\nexpected={expected}\nactual={actual}")

    weights_sha = sha256_file(weights_path)
    if gt2_summary.get("pretrained_weights_sha256") != weights_sha:
        raise RuntimeError("预训练权重哈希与 GT2 记录不一致")
    train_pair_csv = Path(gt2_summary.get("pair_csv", "")).resolve()
    train_pair_sha = str(gt2_summary.get("pair_csv_sha256", ""))
    if not train_pair_sha:
        raise RuntimeError("GT2 summary 缺少训练配对清单哈希")
    if not train_pair_csv.is_file() or sha256_file(train_pair_csv) != train_pair_sha:
        raise RuntimeError("GT2 训练配对清单缺失或哈希不匹配；不能验证检查点来源")

    c_path = gt2_dir / "gt2_C_repair_head.pt"
    d_path = gt2_dir / "gt2_D_repair_head.pt"
    c_ckpt = gt3.validate_checkpoint(c_path, "C", weights_sha, train_pair_sha)
    d_ckpt = gt3.validate_checkpoint(d_path, "D", weights_sha, train_pair_sha)
    if c_ckpt.get("initial_head_state_sha256") != d_ckpt.get("initial_head_state_sha256"):
        raise RuntimeError("C/D 检查点并非来自同一个初始修复头")

    eval_pair_sha = sha256_file(pair_csv)
    pair_summary_path = pair_csv.parent / "gt4_pair_summary.json"
    pair_summary = json.loads(pair_summary_path.read_text(encoding="utf-8")) if pair_summary_path.is_file() else {}
    eval_rows = read_rows(pair_csv)
    disjoint_audit = ensure_disjoint_from_training(eval_rows, train_pair_csv)
    by_index = gt3.read_pair_index(pair_csv)
    starts, skipped = choose_nonoverlapping_valid_clips(by_index, args.max_time_diff_ms)
    if len(starts) < args.min_clips:
        raise RuntimeError(
            f"只有 {len(starts)} 个完整、不重叠的三帧片段，少于 --min-clips={args.min_clips}。"
            "请检查配对清单；不要降低时间差阈值或重复使用 GT 来凑样本。"
        )

    clips = []
    for s in starts:
        indices = [s, s + 1, s + 2]
        name = f"{args.sequence_name}_clip_{s:04d}"
        selected = gt3.select_records(by_index, indices, args.max_time_diff_ms, COHORT, name)
        clips.append({"cohort": COHORT, "clip_name": name, "indices": indices,
                      "primary_indices": [s + 1, s + 2], "selected": selected})
    clip_sets = [set(c["indices"]) for c in clips]
    for i in range(len(clip_sets)):
        for j in range(i + 1, len(clip_sets)):
            if clip_sets[i] & clip_sets[j]:
                raise RuntimeError("自动选择的 GT4 片段存在重叠")

    torch.set_num_threads(4)
    torch.manual_seed(20261024)
    np.random.seed(20261024)
    torch.cuda.manual_seed_all(20261024)
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    config = {
        "experiment": "GT4 frozen-checkpoint cross-sequence evaluation",
        "sequence_name": args.sequence_name,
        "pair_csv": str(pair_csv),
        "evaluation_pair_csv_sha256": eval_pair_sha,
        "pair_summary": pair_summary,
        "snapshot": str(snapshot),
        "pretrained_weights_sha256": weights_sha,
        "gt2_summary": str(gt2_summary_path),
        "gt2_training_pair_csv": str(train_pair_csv),
        "gt2_training_pair_csv_sha256": train_pair_sha,
        "training_sequence": gt2_summary.get("sequence", "Bonn Crowd2"),
        "evaluation_training_path_disjointness": disjoint_audit,
        "gt2_C_checkpoint_sha256": sha256_file(c_path),
        "gt2_D_checkpoint_sha256": sha256_file(d_path),
        "decoder_sha256": sha256_file(decoder_path),
        "repair_head_source_sha256": sha256_file(head_path),
        "gt2_training_script_sha256": sha256_file(train_script_path),
        "gt4_pair_builder_sha256": sha256_file(Path(__file__).with_name("gt4_build_sequence_pairs.py")),
        "gt4_evaluator_sha256": sha256_file(Path(__file__)),
        "depth_scale": DEPTH_SCALE,
        "max_time_diff_ms": args.max_time_diff_ms,
        "clip_length": 3,
        "selected_clip_starts": starts,
        "selected_clip_indices": [c["indices"] for c in clips],
        "primary_scored_positions": [1, 2],
        "skipped_candidate_starts": skipped,
        "gradient_updates": False,
        "ground_truth_used_in_forward": False,
        "modes": ["A:repair_off", "C:pre_readout trained checkpoint", "D:post_readout trained checkpoint"],
        "gpu": torch.cuda.get_device_name(0),
    }
    write_json(out_dir / "gt4_config.json", config)
    (out_dir / "gt4_pair_csv_used.csv").write_bytes(pair_csv.read_bytes())

    metric_rows = []
    pred_cache = {}
    run_log = []
    for mode, ckpt_path in [("A", None), ("C", c_path), ("D", d_path)]:
        print(f"\n========== GT4 {args.sequence_name}: mode {mode} ==========", flush=True)
        t0 = time.perf_counter()
        model = gt3.load_eval_model(snapshot, mode, device, ckpt_path, weights_sha, train_pair_sha)
        for clip in clips:
            print(f"[{mode}] {clip['clip_name']} indices={clip['indices']}", flush=True)
            context = gt3.build_context(model, clip["selected"], device)
            preds = gt3.forward_clip(model, context)
            if len(preds) != len(clip["selected"]):
                raise RuntimeError(f"{clip['clip_name']}: 预测帧数与输入帧数不匹配")
            for pos, (frame, pred) in enumerate(zip(context["frames"], preds)):
                metrics = gt2.calculate_metrics(frame["gt_np"], pred)
                pred_t = torch.from_numpy(pred).to(device)
                with torch.no_grad():
                    loss_t, _ = gt2.masked_log_depth_loss(pred_t, frame["gt"])
                primary = frame["index"] in clip["primary_indices"]
                metric_rows.append({
                    "sequence_name": args.sequence_name,
                    "cohort": COHORT,
                    "clip_name": clip["clip_name"],
                    "mode": mode,
                    "sequence_position": pos,
                    "frame_index_zero_based": frame["index"],
                    "primary_eval": str(bool(primary)).lower(),
                    "rgb_file": frame["rgb_path"].name,
                    "gt_file": frame["gt_path"].name,
                    "abs_timestamp_diff_ms": frame["abs_delta_ms"],
                    **metrics,
                    "log_depth_loss": float(loss_t.item()),
                })
                pred_cache[(clip["clip_name"], mode, frame["index"])] = pred
            del context, preds
            torch.cuda.empty_cache()
        elapsed = time.perf_counter() - t0
        run_log.append({"mode": mode, "elapsed_sec": elapsed,
                        "checkpoint": str(ckpt_path) if ckpt_path else None})
        print(f"[{mode}] complete in {elapsed:.1f}s", flush=True)
        del model
        torch.cuda.empty_cache()

    branch_checks = []
    max_keyframe_diff = 0.0
    max_d_first_propagated_diff = 0.0
    for clip in clips:
        first = clip["indices"][0]
        a0 = pred_cache[(clip["clip_name"], "A", first)]
        for mode in ("C", "D"):
            candidate = pred_cache[(clip["clip_name"], mode, first)]
            diff = float(np.max(np.abs(a0.astype(np.float32) - candidate.astype(np.float32))))
            max_keyframe_diff = max(max_keyframe_diff, diff)
            branch_checks.append({"sequence_name": args.sequence_name, "clip_name": clip["clip_name"],
                                  "check": f"{mode}_keyframe_vs_A", "frame_index": first,
                                  "max_abs_depth_diff_m": diff, "pass": diff <= 1e-4})
        second = clip["indices"][1]
        a1 = pred_cache[(clip["clip_name"], "A", second)]
        d1 = pred_cache[(clip["clip_name"], "D", second)]
        diff = float(np.max(np.abs(a1.astype(np.float32) - d1.astype(np.float32))))
        max_d_first_propagated_diff = max(max_d_first_propagated_diff, diff)
        branch_checks.append({"sequence_name": args.sequence_name, "clip_name": clip["clip_name"],
                              "check": "D_first_propagated_frame_vs_A", "frame_index": second,
                              "max_abs_depth_diff_m": diff, "pass": diff <= 1e-4})

    branch_pass = all(r["pass"] for r in branch_checks)
    agg = gt3.aggregate(metric_rows, COHORT)
    summary = {
        "experiment": "GT4 frozen-checkpoint cross-sequence evaluation",
        "sequence_name": args.sequence_name,
        "status": "PASS" if branch_pass else "BRANCH_INVARIANT_FAIL",
        "scope": "Cross-sequence screening only; no training, no GT input to forward, metrics scored on propagation positions 1 and 2 of disjoint three-frame clips.",
        "pair_summary": pair_summary,
        "clip_count": len(clips),
        "scored_frame_count_per_mode": agg["primary_frame_count_per_mode"],
        "cross_sequence_holdout": agg,
        "branch_invariants": {"max_keyframe_abs_diff_m_across_A_C_D": max_keyframe_diff,
                              "max_D_first_propagated_abs_diff_m_vs_A": max_d_first_propagated_diff,
                              "all_checks_pass": branch_pass, "checks": branch_checks},
        "run": run_log,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "important_limitations": [
            "A sequence is a separate domain but this remains a small Bonn-only cross-sequence screen, not broad dataset generalization.",
            "The three-frame protocol measures short-horizon propagation from a fresh keyframe; it does not establish long-horizon state stability.",
            "A lower mean metric on these clips is descriptive and not a statistical significance claim.",
            "Checkpoint source hashes are verified against GT2 training metadata; the evaluation pair manifest is tracked separately.",
        ],
    }
    write_json(out_dir / "gt4_summary.json", summary)
    gt3.write_csv(out_dir / "gt4_frame_metrics.csv", metric_rows)
    gt3.write_csv(out_dir / "gt4_branch_checks.csv", branch_checks)

    means = agg["mean_metrics_by_mode"]
    lines = [
        f"# GT4 实验报告：{args.sequence_name} 跨序列验证", "",
        "## 1. 实验目的", "",
        "在不更新任何参数的条件下，将 GT2 在 Bonn Crowd2 上训练的 C/D 修复头直接迁移到本序列；比较 A/C/D 的三帧短时传播表现。", "",
        "## 2. 数据与协议", "",
        f"- 序列：{args.sequence_name}",
        f"- RGB 索引记录：{pair_summary.get('rgb_index_records', 'N/A')}",
        f"- 一对一匹配 GT：{pair_summary.get('matched_pairs', 'N/A')}",
        f"- 未匹配 RGB：{pair_summary.get('unmatched_rgb_frames', 'N/A')}",
        f"- 选择的不重叠三帧片段数：{len(clips)}；起点：{starts}",
        f"- 主要评分帧：每段第 2、3 帧；总计每模式 {agg['primary_frame_count_per_mode']['A']} 帧",
        f"- RGB-GT 最大时间差阈值：{args.max_time_diff_ms:.3f} ms",
        f"- GT 单位：uint16 / {DEPTH_SCALE:.0f} 米；零值无效",
        "- A 为原始模型；C/D 使用 GT2 检查点；全部冻结，不反向传播，不将 GT 输入模型前向。",
        "- 测试配对清单与 GT2 训练配对清单分别记录哈希；测试数据路径与训练 RGB/GT 路径无交集。", "",
        "## 3. 主要指标", "",
        "| 模式 | AbsRel ↓ | RMSE ↓ (m) | delta1 ↑ | log-depth loss ↓ | 相对 A 的 AbsRel 下降 | 相对 A 的 RMSE 下降 | 片段 AbsRel 胜出数 | 判定 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for mode in ("A", "C", "D"):
        m = means[mode]
        if mode == "A":
            rel_a = rel_r = "—"; wins = "—"; verdict = "基线"
        else:
            comp = agg["comparisons_vs_A"][mode]
            rel_a = "N/A" if comp["relative_absrel_reduction_pct"] is None else f"{comp['relative_absrel_reduction_pct']:.2f}%"
            rel_r = "N/A" if comp["relative_rmse_reduction_pct"] is None else f"{comp['relative_rmse_reduction_pct']:.2f}%"
            wins = f"{comp['clips_absrel_better_than_A']}/{comp['clips_evaluated']}"
            verdict = comp["screening_verdict"]
        lines.append(f"| {mode} | {m['AbsRel']:.6f} | {m['RMSE']:.6f} | {m['delta1']:.6f} | {m['log_depth_loss']:.6f} | {rel_a} | {rel_r} | {wins} | {verdict} |")
    lines += ["", "### 逐片段 AbsRel", "", "| 片段 | A | C | D |", "|---|---:|---:|---:|"]
    for name, modes in agg["per_clip_mean_metrics"].items():
        lines.append(f"| {name} | {modes['A']['AbsRel']:.6f} | {modes['C']['AbsRel']:.6f} | {modes['D']['AbsRel']:.6f} |")
    lines += ["", "## 4. 分支不变量", "",
              f"- A/C/D keyframe 最大绝对深度差：{max_keyframe_diff:.8g} m",
              f"- D 首个传播帧与 A 最大绝对深度差：{max_d_first_propagated_diff:.8g} m",
              f"- 分支检查：{'PASS' if branch_pass else 'FAIL'}", "",
              "## 5. 结论边界", "",
              "- 本实验只能说明 Crowd2 训练检查点在当前 Bonn 目标序列上的跨序列筛查结果；不能据此声称泛化到任意场景或数据集。",
              "- 三帧协议不回答长时状态累积稳定性；GT3 连续状态 delta1 退化仍需单独保留和诊断。",
              "- 本报告为描述性筛查，不是统计显著性结论。若 C/D 出现 AbsRel 改善但 delta1 下降，应将其报告为混合结果，而非整体成功。", "",
              "## 6. 输出文件", "",
              "- `gt4_pair_summary.json` / `gt4_pairs.csv`：配对审计与清单",
              "- `gt4_frame_metrics.csv`：逐帧 A/C/D 指标",
              "- `gt4_branch_checks.csv`：分支不变量",
              "- `gt4_config.json` / `gt4_summary.json`：协议、哈希和汇总",
              "- `gt4_run.log`：运行日志（由 tee 保存）", ""]
    (out_dir / "GT4_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n========== GT4 SUMMARY ==========")
    print("Report:", out_dir / "GT4_report.md", flush=True)
    print("Summary:", out_dir / "gt4_summary.json", flush=True)
    print("Frames:", out_dir / "gt4_frame_metrics.csv", flush=True)
    print("Clips:", len(clips), "scored frames/mode:", agg["primary_frame_count_per_mode"]["A"], flush=True)
    print("Branch invariants:", "PASS" if branch_pass else "FAIL", flush=True)
    print("Comparisons:", json.dumps(agg["comparisons_vs_A"], ensure_ascii=False), flush=True)
    if not branch_pass:
        raise RuntimeError("GT4 分支不变量失败；保留输出，停止判定")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"GT4 FAILED: {exc!r}", file=sys.stderr, flush=True)
        raise
