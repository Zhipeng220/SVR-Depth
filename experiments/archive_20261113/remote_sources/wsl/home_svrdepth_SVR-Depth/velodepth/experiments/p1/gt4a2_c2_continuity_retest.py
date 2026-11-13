#!/usr/bin/env python3
"""GT4-A2-C2: no-retraining continuous replay on Crowd2, RGB indices 0..767."""
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

import gt2_fixed_clip_repair_train as gt2  # noqa: E402
import gt4a2_exp1_train_eval as exp1  # noqa: E402

END_INDEX_EXCLUSIVE = 768
REFRESH_INTERVAL = 32
N_CYCLES = 24
EXPECTED_TARGET_COUNT = 96
MODES = {"A": "off", "C": "pre_readout", "D": "post_readout"}
OUT_DEFAULT = "outputs/gt4a2_crowd2_continuity"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_checkpoint_cpu(path: Path, label: str, mode: str, ckpt_dir: Path,
                        snapshot: Path, pair_csv: Path, anchor_csv: Path,
                        exp1_config: dict) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {label} 第1000步检查点：{path}")
    summary_path = ckpt_dir / f"gt4a2_exp1_{label}_training_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"缺少训练摘要：{summary_path}")
    train_summary = load_json(summary_path)
    actual_ckpt_sha = sha256_file(path)
    expected_ckpt_sha = train_summary.get("checkpoint_sha256")
    if expected_ckpt_sha and actual_ckpt_sha != expected_ckpt_sha:
        raise RuntimeError(f"{label} 检查点 SHA-256 与训练摘要不一致")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("mode_label") != label or ckpt.get("mode") != mode:
        raise RuntimeError(f"{label} 检查点模式不匹配：{ckpt.get('mode_label')}/{ckpt.get('mode')}")
    if int(ckpt.get("step_count", -1)) != 1000:
        raise RuntimeError(f"{label} 不是第1000步最终检查点")
    if ckpt.get("anchor_manifest_sha256") != sha256_file(anchor_csv):
        raise RuntimeError(f"{label} 检查点对应的锚点清单哈希不一致")
    if ckpt.get("training_pair_csv_sha256") != sha256_file(pair_csv):
        raise RuntimeError(f"{label} 检查点对应的 Crowd2 配对清单哈希不一致")
    if ckpt.get("pretrained_weights_sha256") != sha256_file(snapshot / "model.safetensors"):
        raise RuntimeError(f"{label} 检查点对应的预训练权重哈希不一致")
    recorded_summary_sha = train_summary.get("final_head_state_sha256")
    actual_head_sha = exp1.state_sha256(ckpt["repair_head_state_dict"])
    if recorded_summary_sha and actual_head_sha != recorded_summary_sha:
        raise RuntimeError(f"{label} 修复头状态哈希与训练摘要不一致")
    ckpt["_actual_sha256"] = actual_ckpt_sha
    ckpt["_training_summary"] = train_summary
    return ckpt


def preflight(args) -> dict:
    snapshot = Path(args.snapshot).resolve()
    pair_csv = Path(args.crowd2_pair_csv).resolve()
    ckpt_dir = Path(args.checkpoint_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    anchor_csv = ckpt_dir / "gt4a2_exp1_anchor_manifest.csv"
    exp1_config_path = ckpt_dir / "gt4a2_exp1_config.json"
    exp1_summary_path = ckpt_dir / "gt4a2_exp1_summary.json"

    required = [
        snapshot / "config.json",
        snapshot / "model.safetensors",
        pair_csv,
        anchor_csv,
        exp1_config_path,
        exp1_summary_path,
        ROOT / "experiments/p1/gt4a2_exp1_train_eval.py",
        ROOT / "experiments/p1/gt2_fixed_clip_repair_train.py",
        ROOT / "velodepth/models/decoder_velo.py",
        ROOT / "velodepth/models/persistent_feature_repair.py",
    ]
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(f"前置文件缺失：{p}")

    exp1_config = load_json(exp1_config_path)
    expected_sources = exp1_config.get("source_hashes", {})
    source_check = {}
    for rel, expected in expected_sources.items():
        p = ROOT / rel
        if not p.is_file():
            raise FileNotFoundError(f"实验1记录的源码文件缺失：{p}")
        actual = sha256_file(p)
        source_check[rel] = {"expected": expected, "actual": actual, "pass": actual == expected}
        if actual != expected:
            raise RuntimeError(
                f"源码哈希与 GT4-A2 实验1不一致：{rel}。"
                "为保证复测只改变状态历史，停止运行。"
            )

    rows_all = exp1.read_manifest(pair_csv, training=False)
    if len(rows_all) < END_INDEX_EXCLUSIVE:
        raise RuntimeError(f"Crowd2 RGB 不足 768 帧：实际 {len(rows_all)}")
    rows = rows_all[:END_INDEX_EXCLUSIVE]
    if [int(r["frame_index_zero_based"]) for r in rows] != list(range(END_INDEX_EXCLUSIVE)):
        raise RuntimeError("Crowd2 前 768 个 RGB 索引不连续")

    anchors = exp1.build_anchor_manifest(rows)
    existing_anchors = list(csv.DictReader(anchor_csv.open("r", newline="", encoding="utf-8-sig")))
    if len(existing_anchors) != N_CYCLES:
        raise RuntimeError(f"现有训练锚点清单行数不是24：{len(existing_anchors)}")
    key_fields = ("sample_id", "refresh_cycle", "cycle_start_index", "cycle_end_index",
                  "anchor_offset_after_keyframe", "anchor_index", "cached_state_index",
                  "supervised_future_indices")
    for fresh, old in zip(anchors, existing_anchors):
        for key in key_fields:
            if str(fresh[key]) != str(old[key]):
                raise RuntimeError(f"重新生成的锚点清单与训练清单不一致：sample={fresh['sample_id']} field={key}")

    target_indices = []
    target_paths = set()
    for anchor in anchors:
        for idx in map(int, anchor["supervised_future_indices"].split(",")):
            row = rows[idx]
            if str(row.get("gt_match_status", "")).strip().upper() != "MATCHED":
                raise RuntimeError(f"96个训练目标中存在未匹配 GT：index={idx}")
            p = str(Path(row["depth_path"]).resolve())
            if p in target_paths:
                raise RuntimeError(f"96个训练目标 GT 路径重复：index={idx}")
            target_paths.add(p)
            target_indices.append(idx)
    target_indices = sorted(target_indices)
    if len(target_indices) != EXPECTED_TARGET_COUNT:
        raise RuntimeError(f"目标帧数量不是96：{len(target_indices)}")

    checkpoints = {}
    for label in ("C", "D"):
        checkpoints[label] = load_checkpoint_cpu(
            ckpt_dir / f"gt4a2_exp1_{label}_final_step1000.pt",
            label, MODES[label], ckpt_dir, snapshot, pair_csv, anchor_csv, exp1_config
        )
    c_init = checkpoints["C"].get("initial_head_state_sha256")
    d_init = checkpoints["D"].get("initial_head_state_sha256")
    if c_init != d_init:
        raise RuntimeError("C/D 检查点记录的初始修复头哈希不一致")

    result = {
        "status": "PASS",
        "experiment": "GT4-A2-C2 Crowd2 continuous-state replay (no retraining)",
        "snapshot": str(snapshot),
        "pretrained_weights_sha256": sha256_file(snapshot / "model.safetensors"),
        "crowd2_pair_csv": str(pair_csv),
        "crowd2_pair_csv_sha256": sha256_file(pair_csv),
        "checkpoint_dir": str(ckpt_dir),
        "checkpoint_sha256": {k: v["_actual_sha256"] for k, v in checkpoints.items()},
        "checkpoint_final_head_sha256": {
            k: exp1.state_sha256(v["repair_head_state_dict"]) for k, v in checkpoints.items()
        },
        "recorded_source_hash_check": source_check,
        "rgb_indices": [0, END_INDEX_EXCLUSIVE - 1],
        "rgb_frames": END_INDEX_EXCLUSIVE,
        "refresh_interval": REFRESH_INTERVAL,
        "keyframe_indices": list(range(0, END_INDEX_EXCLUSIVE, REFRESH_INTERVAL)),
        "anchor_count": len(anchors),
        "target_frame_count": len(target_indices),
        "target_indices": target_indices,
        "all_96_targets_matched_and_unique": True,
        "cuda_available": torch.cuda.is_available(),
        "fixed_checkpoint_policy": "C/D final step 1000 only; no retraining and no checkpoint selection",
    }
    if not args.preflight_only and not torch.cuda.is_available():
        raise RuntimeError("连续复测需要 CUDA；torch.cuda.is_available() 为 False")
    return result


def load_fixed_model(label: str, snapshot: Path, ckpt_dir: Path, device: torch.device):
    mode = MODES[label]
    model = gt2.load_model(snapshot, mode, device)
    if label in ("C", "D"):
        ckpt = torch.load(
            ckpt_dir / f"gt4a2_exp1_{label}_final_step1000.pt",
            map_location="cpu", weights_only=False
        )
        model.pixel_decoder.persistent_feature_repair.load_state_dict(
            ckpt["repair_head_state_dict"], strict=True
        )
        model.pixel_decoder.persistent_feature_repair.eval()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def evaluate_short_windows(rows: list[dict], anchors: list[dict], snapshot: Path,
                           ckpt_dir: Path, device: torch.device, out_dir: Path):
    print("\n========== 1. Rebuild A-state caches and exact 24 short windows =========", flush=True)
    model_a = gt2.load_model(snapshot, "off", device)
    shape = exp1.shape_info_for(model_a, Path(rows[0]["rgb_path"]), device)
    state_cache = exp1.precompute_a_states(model_a, rows, anchors, shape, device)
    contexts = exp1.build_training_contexts(model_a, rows, anchors, state_cache, shape, device)
    del model_a, state_cache
    torch.cuda.empty_cache()

    target_losses = {}
    short_rows = []
    a_short_preds = {}
    mode_window_means = {}
    for label in ("A", "C", "D"):
        print(f"[short-window] replay mode={label}", flush=True)
        model = load_fixed_model(label, snapshot, ckpt_dir, device)

        # IMPORTANT: forward_fast uses RadialModule.original_shapes, which is
        # initialized by Decoder.set_shapes during a keyframe forward. Each
        # newly loaded model instance must be warmed up before short-window
        # propagation. Match the warm-up used by GT4-A2 experiment-1 training.
        warm_context = contexts[0]
        warm_frame = warm_context["frames"][0]
        warm_shape = {
            "validity_mask": torch.ones(
                (1, 1, *warm_context["network_shape"]),
                device=device,
                dtype=torch.bool,
            ),
            "network_shape": warm_context["network_shape"],
        }
        exp1.keyframe_state(model, warm_frame, warm_shape)
        model.eval()
        if model.pixel_decoder.persistent_feature_repair is not None:
            model.pixel_decoder.persistent_feature_repair.eval()

        window_losses = []
        with torch.no_grad():
            for context in contexts:
                radius, confidence, features = exp1.move_initial_state(context, device)
                per_frame_losses = []
                per_frame_counts = []
                for t in range(1, 6):
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                        radius, confidence, features = exp1.one_fast_step(
                            model, context["frames"][t - 1], context["frames"][t],
                            context["fast_static"][t - 1], radius, confidence, features
                        )
                        if t >= 2:
                            pred = gt2.make_z_depth(
                                radius, context["rays_map"], context["padded_shape"],
                                context["paddings"], model.interpolation_mode
                            )
                    if t >= 2:
                        target_frame = context["frames"][t]
                        loss, count = gt2.masked_log_depth_loss(pred, target_frame["gt"])
                        loss_value = float(loss.detach().item())
                        target_idx = int(target_frame["index"])
                        target_losses[(label, target_idx)] = {
                            "mode": label,
                            "target_index": target_idx,
                            "refresh_cycle": target_idx // REFRESH_INTERVAL,
                            "frame_offset_in_cycle": target_idx % REFRESH_INTERVAL,
                            "anchor_index": int(context["anchor_index"]),
                            "short_window_log_depth_loss": loss_value,
                            "short_window_valid_pixels": count,
                            "gt_file": Path(target_frame["gt_path"]).name,
                            "abs_timestamp_diff_ms": float(target_frame.get("abs_delta_ms") or 0.0),
                        }
                        per_frame_losses.append(loss_value)
                        per_frame_counts.append(count)
                        if label == "A":
                            a_short_preds[target_idx] = pred.detach().float().cpu().numpy().copy()
                if len(per_frame_losses) != 4:
                    raise RuntimeError(f"短窗口目标数不是4：anchor={context['anchor_index']}")
                window_mean = float(np.mean(per_frame_losses))
                window_losses.append(window_mean)
                short_rows.append({
                    "mode": label,
                    "sample_id": int(context["sample_id"]),
                    "anchor_index": int(context["anchor_index"]),
                    "target_indices": ",".join(str(int(context["anchor_index"]) + k) for k in range(1, 5)),
                    "short_window_mean_log_depth_loss": window_mean,
                    "mean_valid_pixels_per_target": float(np.mean(per_frame_counts)),
                })
                if len(window_losses) == 1 or len(window_losses) % 6 == 0 or len(window_losses) == len(contexts):
                    print(f"[short-window/{label}] {len(window_losses)}/{len(contexts)} windows", flush=True)
        mode_window_means[label] = float(np.mean(window_losses))
        if label in ("C", "D"):
            summary = load_json(ckpt_dir / f"gt4a2_exp1_{label}_training_summary.json")
            recorded = float(summary["final_mean_training_objective"])
            mode_window_means[label + "_recorded_final_objective"] = recorded
            mode_window_means[label + "_replay_abs_diff_vs_recorded"] = abs(mode_window_means[label] - recorded)
            if abs(mode_window_means[label] - recorded) > 5e-4:
                raise RuntimeError(
                    f"{label} 固定短窗口重放与已记录最终目标差异过大："
                    f"{mode_window_means[label]:.8f} vs {recorded:.8f}"
                )
        del model
        torch.cuda.empty_cache()

    if len(a_short_preds) != EXPECTED_TARGET_COUNT:
        raise RuntimeError(f"A 短窗口预测缓存不是96帧：{len(a_short_preds)}")
    write_csv(out_dir / "gt4a2_c2_short_window_cycle_means.csv", short_rows)
    write_csv(out_dir / "gt4a2_c2_short_window_target_losses.csv",
              sorted(target_losses.values(), key=lambda r: (r["mode"], r["target_index"])))
    return target_losses, a_short_preds, mode_window_means, contexts


def run_continuous_mode(label: str, rows: list[dict], target_indices: set[int],
                        snapshot: Path, ckpt_dir: Path, device: torch.device,
                        a_keyframes: dict | None = None, a_short_preds: dict | None = None):
    print(f"\n========== 2. Continuous replay: mode={label}, frames=0..767 =========", flush=True)
    model = load_fixed_model(label, snapshot, ckpt_dir, device)
    shape = exp1.shape_info_for(model, Path(rows[0]["rgb_path"]), device)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    prev_frame = None
    radius = confidence = features = rays_map = None
    frame_rows = []
    target_continuous = {}
    keyframe_preds = {}
    keyframe_max_diff = 0.0
    a_prediction_max_diff = 0.0
    start = time.perf_counter()

    with torch.no_grad():
        for idx, row in enumerate(rows):
            matched = str(row.get("gt_match_status", "")).strip().upper() == "MATCHED"
            frame = exp1.load_frame(row, shape, device, need_gt=matched)
            is_keyframe = idx % REFRESH_INTERVAL == 0
            if is_keyframe:
                rays_map, radius, confidence, features = exp1.keyframe_state(model, frame, shape)
            else:
                static = exp1.transition_static(model, prev_frame, frame, shape, dis)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    radius, confidence, features = exp1.one_fast_step(
                        model, prev_frame, frame, static, radius, confidence, features
                    )
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                pred_t = gt2.make_z_depth(
                    radius, rays_map, shape["padded_shape"], shape["paddings"],
                    model.interpolation_mode
                )
            pred_np = pred_t.detach().float().cpu().numpy()
            if is_keyframe:
                keyframe_preds[idx] = pred_np.copy()
                if label != "A":
                    if a_keyframes is None or idx not in a_keyframes:
                        raise RuntimeError(f"缺少 A 关键帧预测：index={idx}")
                    keyframe_max_diff = max(
                        keyframe_max_diff,
                        float(np.max(np.abs(pred_np.astype(np.float32) -
                                            a_keyframes[idx].astype(np.float32))))
                    )
            if idx in target_indices:
                if not matched:
                    raise RuntimeError(f"96目标帧在连续运行时没有GT：index={idx}")
                loss, count = gt2.masked_log_depth_loss(pred_t, frame["gt"])
                target_continuous[idx] = {
                    "mode": label,
                    "target_index": idx,
                    "continuous_log_depth_loss": float(loss.item()),
                    "continuous_valid_pixels": count,
                }
                if label == "A":
                    if a_short_preds is None or idx not in a_short_preds:
                        raise RuntimeError(f"缺少 A 短窗口预测用于一致性检查：index={idx}")
                    a_prediction_max_diff = max(
                        a_prediction_max_diff,
                        float(np.max(np.abs(pred_np.astype(np.float32) -
                                            a_short_preds[idx].astype(np.float32))))
                    )
            if matched:
                metrics = gt2.calculate_metrics(frame["gt_np"], pred_np)
                frame_rows.append({
                    "mode": label,
                    "frame_index_zero_based": idx,
                    "refresh_cycle": idx // REFRESH_INTERVAL,
                    "frame_offset_in_cycle": idx % REFRESH_INTERVAL,
                    "is_keyframe": str(is_keyframe).lower(),
                    "rgb_file": frame["rgb_file"],
                    "gt_file": Path(row["depth_path"]).name,
                    "abs_timestamp_diff_ms": float(row.get("abs_delta_ms") or 0.0),
                    **metrics,
                })
            prev_frame = frame
            if idx == 0 or (idx + 1) % 100 == 0 or idx == len(rows) - 1:
                print(f"[continuous/{label}] {idx+1}/{len(rows)} frames", flush=True)

    elapsed = time.perf_counter() - start
    if len(target_continuous) != EXPECTED_TARGET_COUNT:
        raise RuntimeError(f"{label} 连续运行的96目标帧数不符：{len(target_continuous)}")
    if not frame_rows:
        raise RuntimeError(f"{label} 没有可评分的 Crowd2 帧")
    metrics_summary = {
        metric: float(np.mean([float(r[metric]) for r in frame_rows]))
        for metric in ("AbsRel", "RMSE", "delta1")
    }
    prop_rows = [r for r in frame_rows if r["is_keyframe"] == "false"]
    prop_summary = {
        metric: float(np.mean([float(r[metric]) for r in prop_rows]))
        for metric in ("AbsRel", "RMSE", "delta1")
    }
    summary = {
        "mode": label,
        "rgb_frames_processed": len(rows),
        "matched_gt_frames_scored": len(frame_rows),
        "propagated_matched_frames_scored": len(prop_rows),
        "all_matched_frame_macro_metrics": metrics_summary,
        "propagated_frame_macro_metrics": prop_summary,
        "elapsed_sec": elapsed,
        "keyframe_max_abs_depth_diff_vs_A_m": keyframe_max_diff if label != "A" else 0.0,
        "A_short_window_vs_continuous_max_abs_depth_diff_on_96_targets_m":
            a_prediction_max_diff if label == "A" else None,
    }
    print(
        f"[continuous/{label}] done: AbsRel={metrics_summary['AbsRel']:.6f}, "
        f"RMSE={metrics_summary['RMSE']:.6f}, delta1={metrics_summary['delta1']:.6f}, "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()
    return frame_rows, target_continuous, keyframe_preds, summary


def summarize_results(frame_rows: list[dict], target_rows: list[dict], mode_summaries: list[dict],
                      short_means: dict, out_dir: Path, pre: dict) -> dict:
    sequence_rows = []
    offset_rows = []
    metrics_by_mode = {}
    for label in ("A", "C", "D"):
        rr = [r for r in frame_rows if r["mode"] == label]
        metrics_by_mode[label] = {
            m: float(np.mean([float(r[m]) for r in rr])) for m in ("AbsRel", "RMSE", "delta1")
        }
        prop = [r for r in rr if r["is_keyframe"] == "false"]
        metrics_by_mode[label]["propagated_only"] = {
            m: float(np.mean([float(r[m]) for r in prop])) for m in ("AbsRel", "RMSE", "delta1")
        }
        sequence_rows.append({
            "mode": label,
            "scored_frames": len(rr),
            "AbsRel_all": metrics_by_mode[label]["AbsRel"],
            "RMSE_all": metrics_by_mode[label]["RMSE"],
            "delta1_all": metrics_by_mode[label]["delta1"],
            "AbsRel_propagated_only": metrics_by_mode[label]["propagated_only"]["AbsRel"],
            "RMSE_propagated_only": metrics_by_mode[label]["propagated_only"]["RMSE"],
            "delta1_propagated_only": metrics_by_mode[label]["propagated_only"]["delta1"],
        })
        for offset in range(REFRESH_INTERVAL):
            selected = [r for r in rr if int(r["frame_offset_in_cycle"]) == offset]
            offset_rows.append({
                "mode": label,
                "frame_offset_in_cycle": offset,
                "is_keyframe_offset": offset == 0,
                "matched_frames": len(selected),
                "AbsRel": float(np.mean([float(r["AbsRel"]) for r in selected])) if selected else "",
                "RMSE": float(np.mean([float(r["RMSE"]) for r in selected])) if selected else "",
                "delta1": float(np.mean([float(r["delta1"]) for r in selected])) if selected else "",
            })
    write_csv(out_dir / "gt4a2_c2_continuous_frame_metrics.csv", frame_rows)
    write_csv(out_dir / "gt4a2_c2_sequence_summary.csv", sequence_rows)
    write_csv(out_dir / "gt4a2_c2_frame_offset_summary.csv", offset_rows)
    write_csv(out_dir / "gt4a2_c2_target_loss_comparison.csv", target_rows)

    short_by_mode = {}
    continuous_by_mode = {}
    for label in ("A", "C", "D"):
        sr = [r for r in target_rows if r["mode"] == label]
        short_by_mode[label] = float(np.mean([float(r["short_window_log_depth_loss"]) for r in sr]))
        continuous_by_mode[label] = float(np.mean([float(r["continuous_log_depth_loss"]) for r in sr]))
    target_summary = {}
    for label in ("A", "C", "D"):
        short_loss = short_by_mode[label]
        cont_loss = continuous_by_mode[label]
        short_improvement = (short_by_mode["A"] - short_loss) / max(short_by_mode["A"], 1e-12) * 100.0
        cont_improvement = (continuous_by_mode["A"] - cont_loss) / max(continuous_by_mode["A"], 1e-12) * 100.0
        target_summary[label] = {
            "short_window_mean_loss_96_targets": short_loss,
            "continuous_mean_loss_same_96_targets": cont_loss,
            "continuous_minus_short_abs_loss": cont_loss - short_loss,
            "continuous_loss_increase_pct_vs_short": (cont_loss - short_loss) / max(short_loss, 1e-12) * 100.0,
            "short_window_improvement_pct_vs_A": short_improvement,
            "continuous_improvement_pct_vs_A_on_same_targets": cont_improvement,
        }
    a_consistency = next(s for s in mode_summaries if s["mode"] == "A")
    max_a_loss_diff = max(
        abs(float(r["continuous_log_depth_loss"]) - float(r["short_window_log_depth_loss"]))
        for r in target_rows if r["mode"] == "A"
    )
    max_a_pred_diff = float(a_consistency["A_short_window_vs_continuous_max_abs_depth_diff_on_96_targets_m"])
    keyframe_max = {
        s["mode"]: s["keyframe_max_abs_depth_diff_vs_A_m"] for s in mode_summaries
    }
    a_consistency_pass = max_a_pred_diff <= 1e-4 and max_a_loss_diff <= 1e-4
    offset_map = {(r["mode"], int(r["frame_offset_in_cycle"])): r for r in offset_rows}
    route_signals = {}
    for label in ("C", "D"):
        short_benefit = target_summary[label]["short_window_improvement_pct_vs_A"] > 0
        continuous_target_benefit = target_summary[label]["continuous_improvement_pct_vs_A_on_same_targets"] > 0
        full = metrics_by_mode[label]
        base = metrics_by_mode["A"]
        full_stable = (
            full["AbsRel"] < base["AbsRel"]
            and full["RMSE"] <= base["RMSE"]
            and full["delta1"] >= base["delta1"] - 0.005
        )
        early_cand = [float(offset_map[(label, i)]["AbsRel"]) for i in range(1, 9)
                      if offset_map[(label, i)]["AbsRel"] != ""]
        early_a = [float(offset_map[("A", i)]["AbsRel"]) for i in range(1, 9)
                   if offset_map[("A", i)]["AbsRel"] != ""]
        late_cand = [float(offset_map[(label, i)]["AbsRel"]) for i in range(25, 32)
                     if offset_map[(label, i)]["AbsRel"] != ""]
        late_a = [float(offset_map[("A", i)]["AbsRel"]) for i in range(25, 32)
                  if offset_map[("A", i)]["AbsRel"] != ""]
        early_better = bool(early_cand and early_a and np.mean(early_cand) < np.mean(early_a))
        late_worse = bool(late_cand and late_a and np.mean(late_cand) > np.mean(late_a))
        route_signals[label] = {
            "short_window_target_loss_better_than_A": short_benefit,
            "continuous_same_target_loss_better_than_A": continuous_target_benefit,
            "full_sequence_stable_improvement_vs_A": bool(full_stable),
            "offset_1_8_better_than_A": early_better,
            "offset_25_31_worse_than_A": late_worse,
        }

    if not a_consistency_pass:
        route = "NO_DECISION_A_CONSISTENCY_CHECK_FAILED"
        route_text = "A短窗口与连续运行不一致；先排查状态缓存、帧相位或预处理，不选择修改路线。"
    elif any(v["full_sequence_stable_improvement_vs_A"] for v in route_signals.values()):
        route = "SCENE_COVERAGE_CANDIDATE"
        route_text = "至少一个修复方式在 Crowd2 全序列的 AbsRel/RMSE/δ1 上稳定改善；结合此前 Crowd3/Balloon2 失败，优先考虑训练场景覆盖。"
    elif any(v["short_window_target_loss_better_than_A"] and
             not v["continuous_same_target_loss_better_than_A"] for v in route_signals.values()):
        route = "TRAINING_CLOSURE_CANDIDATE"
        route_text = "至少一种方式的短窗口目标优于 A，但连续运行同一批96帧不再优于 A；优先改训练闭环与跨窗口状态传递。"
    elif any(v["continuous_same_target_loss_better_than_A"] and
             v["offset_25_31_worse_than_A"] for v in route_signals.values()):
        route = "POSITION_COVERAGE_CANDIDATE"
        route_text = "同一批监督帧仍有收益，但刷新周期后段退化；优先补齐传播位置与连续修复次数覆盖。"
    else:
        route = "MIXED_EVIDENCE_REVIEW_REQUIRED"
        route_text = "结果未满足单一自动分支；依据同一批目标损失与逐位置 AbsRel 表人工判读，不追加实验。"

    summary = {
        "experiment": "GT4-A2-C2 Crowd2 continuous-state replay; no retraining",
        "status": "COMPLETE",
        "protocol_validity": {
            "A_short_window_vs_continuous_consistency_pass": bool(a_consistency_pass),
            "max_abs_A_prediction_difference_on_96_targets_m": max_a_pred_diff,
            "max_abs_A_log_loss_difference_on_96_targets": max_a_loss_diff,
            "max_keyframe_prediction_difference_vs_A_m": keyframe_max,
        },
        "short_vs_continuous_target_loss": target_summary,
        "full_sequence_metrics": metrics_by_mode,
        "route_signals": route_signals,
        "candidate_route": route,
        "candidate_route_explanation": route_text,
        "short_window_replay_means": short_means,
        "preflight": pre,
        "limitations": [
            "No weights were trained or updated in this experiment.",
            "The 96 target frames are the same supervised RGB-GT pairs used in the 24 short windows.",
            "The first 768 RGB indices are processed continuously; unmatched-GT RGB frames remain in the state trajectory and are omitted only from metric scoring.",
            "The 96 target frames cover selected offsets, while the offset summary reports every position 0..31.",
            "Route signals are deterministic development diagnostics, not statistical significance tests.",
        ],
    }
    write_json(out_dir / "gt4a2_c2_continuity_summary.json", summary)
    write_report(out_dir, summary, sequence_rows, offset_rows, target_rows, mode_summaries)
    return summary


def write_report(out_dir: Path, summary: dict, sequence_rows: list[dict],
                 offset_rows: list[dict], target_rows: list[dict], mode_summaries: list[dict]):
    lines = [
        "# GT4-A2-C2：Crowd2 连续状态复测报告",
        "",
        "**实验性质：不重新训练；仅重放现有 C/D 第1000步检查点。**",
        "",
        "## 1. 实验问题",
        "",
        "检验短窗口训练得到的 C/D 修复头，在 Crowd2 前768帧从序列开头连续运行、每32帧刷新且每个传播帧持续修复时，能否在原来的96个监督目标帧上保留短窗口收益。",
        "",
        "## 2. 协议与完整性",
        "",
        f"- RGB 索引：0–767，共 {summary['preflight']['rgb_frames']} 帧。",
        f"- 刷新：每 {REFRESH_INTERVAL} 帧；关键帧索引为 0,32,...,736。",
        f"- 训练窗口：{N_CYCLES} 个，监督目标 {EXPECTED_TARGET_COUNT} 帧；同一目标集合分别以 A/C/D 短窗口重放和连续运行。",
        "- 权重：C/D 现有第1000步最终检查点；本实验没有训练或更新任何参数。",
        f"- A 短窗口与连续运行的预测最大绝对差：{summary['protocol_validity']['max_abs_A_prediction_difference_on_96_targets_m']:.8g} m。",
        f"- A 同一96帧 log-depth loss 最大绝对差：{summary['protocol_validity']['max_abs_A_log_loss_difference_on_96_targets']:.8g}。",
        f"- A 一致性检查：{'PASS' if summary['protocol_validity']['A_short_window_vs_continuous_consistency_pass'] else 'FAIL'}。",
        "",
        "## 3. 同一96个目标帧：短窗口与连续运行的 log-depth loss",
        "",
        "| 模式 | 短窗口平均 loss | 连续运行同帧平均 loss | 连续相对短窗口变化 | 短窗口相对A改善 | 连续同帧相对A改善 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in ("A", "C", "D"):
        r = summary["short_vs_continuous_target_loss"][label]
        lines.append(
            f"| {label} | {r['short_window_mean_loss_96_targets']:.6f} | "
            f"{r['continuous_mean_loss_same_96_targets']:.6f} | "
            f"{r['continuous_loss_increase_pct_vs_short']:+.2f}% | "
            f"{r['short_window_improvement_pct_vs_A']:+.2f}% | "
            f"{r['continuous_improvement_pct_vs_A_on_same_targets']:+.2f}% |"
        )
    lines += [
        "",
        "正的“相对A改善”表示 loss 低于 A；正的“连续相对短窗口变化”表示连续运行 loss 上升。A 的一致性检查失败时，不应对 C/D 的历史效应作因果解释。",
        "",
        "## 4. Crowd2 前768帧完整连续序列指标",
        "",
        "| 模式 | 有效GT帧数 | AbsRel ↓ | RMSE ↓ (m) | δ1 ↑ | 仅传播帧 AbsRel | 仅传播帧 RMSE | 仅传播帧 δ1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sequence_rows:
        lines.append(
            f"| {r['mode']} | {r['scored_frames']} | {r['AbsRel_all']:.6f} | "
            f"{r['RMSE_all']:.6f} | {r['delta1_all']:.6f} | "
            f"{r['AbsRel_propagated_only']:.6f} | {r['RMSE_propagated_only']:.6f} | "
            f"{r['delta1_propagated_only']:.6f} |"
        )
    lines += [
        "",
        "## 5. 按刷新周期内位置统计 AbsRel",
        "",
        "以下每个 offset 汇总所有周期中相同传播位置的逐帧 AbsRel 均值。offset 0 是关键帧；offset 1–31 是传播帧。完整表也写入 `gt4a2_c2_frame_offset_summary.csv`。",
        "",
        "| offset | A | C | D |",
        "|---:|---:|---:|---:|",
    ]
    off = {(r["mode"], int(r["frame_offset_in_cycle"])): r for r in offset_rows}
    for i in range(REFRESH_INTERVAL):
        vals = []
        for label in ("A", "C", "D"):
            v = off[(label, i)]["AbsRel"]
            vals.append("NA" if v == "" else f"{float(v):.6f}")
        lines.append(f"| {i} | {vals[0]} | {vals[1]} | {vals[2]} |")
    lines += [
        "",
        "## 6. 路线判断信号",
        "",
        f"**候选路线：`{summary['candidate_route']}`**",
        "",
        summary["candidate_route_explanation"],
        "",
        "| 模式 | 短窗口目标优于A | 连续同帧目标优于A | 全序列三指标稳定改善 | offset 1–8 优于A | offset 25–31 劣于A |",
        "|---|---|---|---|---|---|",
    ]
    for label in ("C", "D"):
        r = summary["route_signals"][label]
        lines.append(
            f"| {label} | {'是' if r['short_window_target_loss_better_than_A'] else '否'} | "
            f"{'是' if r['continuous_same_target_loss_better_than_A'] else '否'} | "
            f"{'是' if r['full_sequence_stable_improvement_vs_A'] else '否'} | "
            f"{'是' if r['offset_1_8_better_than_A'] else '否'} | "
            f"{'是' if r['offset_25_31_worse_than_A'] else '否'} |"
        )
    lines += [
        "",
        "## 7. 解释边界",
        "",
        "- 本实验不做统计显著性检验；报告连续/短窗口的绝对与相对变化，不把“明显”伪装成统计显著。",
        "- 同一96帧的目标损失直接检验历史状态差异；逐位置 AbsRel 用于判断训练监督位置之外是否退化。",
        "- Crowd2 上的复测不能单独证明跨场景泛化；Crowd3/Balloon2 的既有结果仅作为背景，不在本实验重跑。",
        "- 本轮不新增误差图、深度分桶、刷新间隔扫描、随机种子或训练预算。",
        "",
        "## 8. 输出文件",
        "",
        "- `gt4a2_c2_short_window_target_losses.csv`：A/C/D 短窗口下同一96帧的 loss。",
        "- `gt4a2_c2_target_loss_comparison.csv`：短窗口与连续运行逐帧配对对照。",
        "- `gt4a2_c2_continuous_frame_metrics.csv`：A/C/D 前768帧逐帧指标。",
        "- `gt4a2_c2_sequence_summary.csv`：全序列及仅传播帧指标。",
        "- `gt4a2_c2_frame_offset_summary.csv`：offset 0–31 的指标。",
        "- `gt4a2_c2_continuity_summary.json`：完整性检查、损失变化与路线信号。",
        "",
    ]
    (out_dir / "GT4_A2_Crowd2_连续状态复测报告.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default=os.environ.get("VDEEP_SNAPSHOT", ""))
    parser.add_argument(
        "--crowd2-pair-csv",
        default=os.environ.get(
            "GT4A2_CROWD2_PAIR_CSV",
            "outputs/gt4a2_exp1/gt4a2_exp1_crowd2_pairs_used.csv"
        ),
    )
    parser.add_argument("--checkpoint-dir", default="outputs/gt4a2_exp1")
    parser.add_argument("--out-dir", default=OUT_DEFAULT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not args.snapshot:
        raise RuntimeError("未设置 VDEEP_SNAPSHOT")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.preflight_only and (out_dir / "gt4a2_c2_continuity_summary.json").exists():
        raise RuntimeError(f"已有完成标记，拒绝覆盖：{out_dir / 'gt4a2_c2_continuity_summary.json'}")

    pre = preflight(args)
    write_json(out_dir / "gt4a2_c2_preflight.json", pre)
    print(
        f"[PASS] preflight: RGB=0..767, anchors={pre['anchor_count']}, "
        f"same target frames={pre['target_frame_count']}, CUDA={pre['cuda_available']}",
        flush=True,
    )
    if args.preflight_only:
        print("GT4-A2-C2 preflight PASS; no model inference started.", flush=True)
        return 0

    snapshot = Path(args.snapshot).resolve()
    pair_csv = Path(args.crowd2_pair_csv).resolve()
    ckpt_dir = Path(args.checkpoint_dir).resolve()
    rows_all = exp1.read_manifest(pair_csv, training=False)
    rows = rows_all[:END_INDEX_EXCLUSIVE]
    anchors = exp1.build_anchor_manifest(rows)
    target_indices = sorted({
        idx
        for a in anchors
        for idx in map(int, a["supervised_future_indices"].split(","))
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "gt4a2_c2_protocol.json", {
        "experiment": "GT4-A2-C2 Crowd2 continuous-state replay",
        "no_retraining": True,
        "rgb_indices": [0, 767],
        "refresh_interval": REFRESH_INTERVAL,
        "anchors": anchors,
        "target_indices": target_indices,
        "target_count": len(target_indices),
        "pair_csv_sha256": sha256_file(pair_csv),
        "checkpoint_sha256": pre["checkpoint_sha256"],
        "source_hashes": {rel: info["actual"] for rel, info in pre["recorded_source_hash_check"].items()},
        "pretrained_weights_sha256": pre["pretrained_weights_sha256"],
    })

    torch.set_num_threads(4)
    torch.manual_seed(20261024)
    np.random.seed(20261024)
    torch.cuda.manual_seed_all(20261024)
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    target_losses, a_short_preds, short_means, contexts = evaluate_short_windows(
        rows, anchors, snapshot, ckpt_dir, device, out_dir
    )
    del contexts
    torch.cuda.empty_cache()

    frame_rows = []
    continuous_target_losses = {}
    mode_summaries = []
    a_keyframes = None
    target_set = set(target_indices)
    for label in ("A", "C", "D"):
        fr, target_cont, keyframes, mode_summary = run_continuous_mode(
            label, rows, target_set, snapshot, ckpt_dir, device,
            a_keyframes=a_keyframes,
            a_short_preds=a_short_preds if label == "A" else None,
        )
        frame_rows.extend(fr)
        for idx, item in target_cont.items():
            target_losses[(label, idx)].update(item)
        mode_summaries.append(mode_summary)
        if label == "A":
            a_keyframes = keyframes
        else:
            mode_summary["keyframe_check_pass"] = mode_summary["keyframe_max_abs_depth_diff_vs_A_m"] <= 1e-4

    target_rows = []
    for label in ("A", "C", "D"):
        for idx in target_indices:
            r = dict(target_losses[(label, idx)])
            short_loss = float(r["short_window_log_depth_loss"])
            cont_loss = float(r["continuous_log_depth_loss"])
            r["continuous_minus_short_abs"] = cont_loss - short_loss
            r["continuous_change_pct_vs_short"] = (cont_loss - short_loss) / max(short_loss, 1e-12) * 100.0
            r["short_valid_pixels"] = r.pop("short_window_valid_pixels")
            r["continuous_valid_pixels"] = r.pop("continuous_valid_pixels")
            r["short_window_log_depth_loss"] = short_loss
            r["continuous_log_depth_loss"] = cont_loss
            target_rows.append(r)

    sequence_rows = []
    summary = summarize_results(frame_rows, target_rows, mode_summaries, short_means, out_dir, pre)
    summary["runtime_peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
    write_json(out_dir / "gt4a2_c2_continuity_summary.json", summary)
    # Rewrite report after peak-memory metadata is finalized; report content remains deterministic.
    write_report(out_dir, summary, sequence_rows_from_csv(out_dir / "gt4a2_c2_sequence_summary.csv"),
                 rows_from_csv(out_dir / "gt4a2_c2_frame_offset_summary.csv"),
                 target_rows, mode_summaries)
    print("\n========== GT4-A2-C2 COMPLETE =========", flush=True)
    print("A consistency:", summary["protocol_validity"]["A_short_window_vs_continuous_consistency_pass"])
    print("Candidate route:", summary["candidate_route"])
    print("Report:", out_dir / "GT4_A2_Crowd2_连续状态复测报告.md")
    return 0


def rows_from_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("frame_offset_in_cycle", "matched_frames", "AbsRel", "RMSE", "delta1"):
            if k in r and r[k] != "":
                try:
                    r[k] = int(r[k]) if k in ("frame_offset_in_cycle", "matched_frames") else float(r[k])
                except ValueError:
                    pass
    return rows


def sequence_rows_from_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("scored_frames", "AbsRel_all", "RMSE_all", "delta1_all",
                  "AbsRel_propagated_only", "RMSE_propagated_only", "delta1_propagated_only"):
            if k in r and r[k] != "":
                try:
                    r[k] = int(r[k]) if k == "scored_frames" else float(r[k])
                except ValueError:
                    pass
    return rows


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
