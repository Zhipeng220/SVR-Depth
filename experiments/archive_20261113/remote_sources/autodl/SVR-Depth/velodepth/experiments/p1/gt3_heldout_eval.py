#!/usr/bin/env python3
"""GT3: frozen-checkpoint held-out evaluation for E2-S8 C/D repair heads.

Cohorts:
  1) Independent three-frame held-out clips (default starts: 11,100,300,600,850).
     Primary scores are propagated frames (positions 1 and 2), not the keyframe.
  2) Continuation from indices 8..13. Only indices 11..13 are primary scores.
     This is a causal state-continuation test, NOT an independent generalization test.

No gradient updates are performed. Uses GT2's audited preprocessing/inference helpers.
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
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "p1"))

import gt2_fixed_clip_repair_train as gt2  # noqa: E402

DEPTH_SCALE = 5000.0
TRAIN_INDICES = {8, 9, 10}
DEFAULT_INDEPENDENT_STARTS = [11, 100, 300, 600, 850]
EXPECTED_HEAD_PARAMS = 170496


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_pair_index(pair_csv: Path) -> dict[int, dict]:
    if not pair_csv.is_file():
        raise FileNotFoundError(f"GT1-A1 配对清单不存在：{pair_csv}")
    out: dict[int, dict] = {}
    with pair_csv.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            idx = int(row["frame_index_zero_based"])
            if idx in out:
                raise RuntimeError(f"配对清单包含重复 RGB 全局索引：{idx}")
            out[idx] = row
    if not out:
        raise RuntimeError(f"配对清单为空：{pair_csv}")
    return out


def select_records(
    by_index: dict[int, dict],
    indices: list[int],
    max_delta_ms: float,
    cohort: str,
    clip_name: str,
) -> list[dict]:
    if not indices:
        raise ValueError(f"{clip_name}: 空索引列表")
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError(f"{clip_name}: 索引必须连续递增：{indices}")

    selected = []
    for idx in indices:
        if idx not in by_index:
            raise RuntimeError(f"{clip_name}: 配对清单缺少索引 {idx}")
        row = by_index[idx]
        if row.get("gt_match_status") != "MATCHED":
            raise RuntimeError(f"{clip_name}: 索引 {idx} 未匹配 GT：{row}")
        rgb_path = Path(row["rgb_path"])
        gt_path = Path(row["depth_path"])
        if not rgb_path.is_file():
            raise FileNotFoundError(f"{clip_name}: RGB 不存在：{rgb_path}")
        if not gt_path.is_file():
            raise FileNotFoundError(f"{clip_name}: GT 不存在：{gt_path}")
        dt = float(row["abs_delta_ms"])
        if not np.isfinite(dt) or dt > max_delta_ms:
            raise RuntimeError(
                f"{clip_name}: 索引 {idx} 的 RGB-GT 时间差 {dt:.3f} ms "
                f"超过阈值 {max_delta_ms:.3f} ms"
            )
        reused = str(row.get("gt_reused", "False")).strip().lower() in {
            "true", "1", "yes"
        }
        if reused:
            raise RuntimeError(f"{clip_name}: 索引 {idx} 重用了 GT：{gt_path}")

        selected.append({
            "index": idx,
            "rgb_timestamp": float(row["rgb_timestamp"]),
            "gt_timestamp": float(row["depth_timestamp"]),
            "rgb_path": rgb_path.resolve(),
            "gt_path": gt_path.resolve(),
            "abs_delta_ms": dt,
            "cohort": cohort,
            "clip_name": clip_name,
        })

    if len({str(x["rgb_path"]) for x in selected}) != len(selected):
        raise RuntimeError(f"{clip_name}: RGB 路径重复")
    if len({str(x["gt_path"]) for x in selected}) != len(selected):
        raise RuntimeError(f"{clip_name}: GT 路径重复")
    if any(
        selected[i]["rgb_timestamp"] <= selected[i - 1]["rgb_timestamp"]
        for i in range(1, len(selected))
    ):
        raise RuntimeError(f"{clip_name}: RGB 时间戳非严格递增")
    return selected


def build_context(model, selected: list[dict], device: torch.device) -> dict:
    """Generalized GT2 context builder for any >=2 consecutive frames."""
    if len(selected) < 2:
        raise ValueError("GT3 clip 至少需要 2 帧")

    root_img = gt2.load_rgb(selected[0]["rgb_path"])
    original_h, original_w = root_img.shape[:2]
    for rec in selected:
        with Image.open(rec["rgb_path"]) as im:
            if im.size != (original_w, original_h):
                raise RuntimeError(
                    f"RGB 尺寸不一致：{rec['rgb_path']} size={im.size}"
                )

    ratio_bounds = model.shape_constraints["ratio_bounds"]
    pixels_min = float(model.shape_constraints["pixels_min"])
    pixels_max = float(model.shape_constraints["pixels_max"])
    interval = (pixels_max - pixels_min) / 10.0
    pixels_bounds = (pixels_min, pixels_min + interval)
    paddings, (padded_h, padded_w) = gt2.get_paddings(
        (original_h, original_w), ratio_bounds
    )
    _resize_factor, (new_h, new_w) = gt2.get_resize_factor(
        (padded_h, padded_w), pixels_bounds
    )

    mean = torch.tensor(
        gt2.IMAGENET_DATASET_MEAN, device=device
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        gt2.IMAGENET_DATASET_STD, device=device
    ).view(1, 3, 1, 1)

    frames = []
    for rec in selected:
        rgb_np = gt2.load_rgb(rec["rgb_path"])
        img, frame_u8 = gt2.preprocess_rgb(
            rgb_np, device, paddings, (new_h, new_w), mean, std
        )
        gt_raw = cv2.imread(str(rec["gt_path"]), cv2.IMREAD_UNCHANGED)
        if gt_raw is None:
            raise RuntimeError(f"无法读取 GT：{rec['gt_path']}")
        if gt_raw.dtype != np.uint16 or gt_raw.ndim != 2:
            raise RuntimeError(
                f"GT 格式错误：{rec['gt_path']} dtype={gt_raw.dtype}, "
                f"shape={gt_raw.shape}"
            )
        if gt_raw.shape != (original_h, original_w):
            raise RuntimeError(
                f"GT/RGB 尺寸不一致：GT={gt_raw.shape}, "
                f"RGB={(original_h, original_w)}, path={rec['gt_path']}"
            )
        gt_np = gt_raw.astype(np.float32) / DEPTH_SCALE
        frames.append({
            **rec,
            "img": img,
            "frame_u8": frame_u8,
            "gt_np": gt_np,
            "gt": torch.from_numpy(gt_np).to(device),
        })

    validity_mask = torch.ones(
        (1, 1, new_h, new_w), device=device, dtype=torch.bool
    )
    image_paddings = torch.zeros((1, 4), device=device, dtype=torch.long)
    depth_paddings = torch.zeros((1, 4), device=device, dtype=torch.long)
    fast_static = []
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    for t in range(1, len(frames)):
        curr_rgb = frames[t]["frame_u8"].permute(1, 2, 0).numpy()
        prev_rgb = frames[t - 1]["frame_u8"].permute(1, 2, 0).numpy()
        flow_np = dis.calc(
            cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY),
            None,
        )
        flow_px = (
            torch.from_numpy(flow_np)
            .to(device=device)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        flow_norm, grid_sample = model._normalize_and_grid_from_flow(
            flow_px, frames[t]["img"], image_paddings, depth_paddings
        )
        image_residual = (
            gt2.rgb2luma(frames[t]["img"]).unsqueeze(1)
            - F.grid_sample(
                gt2.rgb2luma(frames[t - 1]["img"]).unsqueeze(1),
                grid_sample,
                mode="bilinear",
            )
        )
        fast_static.append({
            "flow_norm": flow_norm,
            "grid_sample": grid_sample,
            "image_residual": image_residual,
            "validity_mask": validity_mask,
        })
        del flow_np, flow_px, curr_rgb, prev_rgb

    base_inputs = {
        "image": frames[0]["img"],
        "image_original": frames[0]["img"],
        "validity_mask": validity_mask,
    }
    base_metas = [{"T": 1, "B": 1}]
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=True
    ):
        base_features, base_tokens = model.pixel_encoder(base_inputs["image"])
        base_features = [
            model.stacking_fn(base_features[i:j]).contiguous()
            for i, j in model.slices_encoder_range
        ]
        base_tokens = [
            model.stacking_fn(base_tokens[i:j]).contiguous()
            for i, j in model.slices_encoder_range
        ]
        base_inputs["features"], base_inputs["tokens"] = base_features, base_tokens
        model.pixel_decoder.set_shapes(base_inputs, base_metas)
        rays_base, radius_state, confidence_state, features_state = (
            model.pixel_decoder.forward_keyframe(base_inputs, base_metas)
        )

    rays_map = (
        rays_base.reshape(1, new_h, new_w, 3)
        .permute(0, 3, 1, 2)
        .contiguous()
        .detach()
    )
    context = {
        "frames": frames,
        "fast_static": fast_static,
        "rays_map": rays_map,
        "keyframe_radius": radius_state.detach(),
        "keyframe_confidence": confidence_state.detach(),
        "keyframe_features": [x.detach() for x in features_state],
        "paddings": paddings,
        "padded_shape": (padded_h, padded_w),
        "original_shape": (original_h, original_w),
        "network_shape": (new_h, new_w),
        "validity_mask": validity_mask,
        "image_paddings": image_paddings,
        "depth_paddings": depth_paddings,
    }
    del (
        base_inputs, base_metas, base_features, base_tokens,
        rays_base, radius_state, confidence_state, features_state
    )
    return context


def forward_clip(model, context: dict) -> list[np.ndarray]:
    radius_state = context["keyframe_radius"]
    confidence_state = context["keyframe_confidence"]
    features_state = context["keyframe_features"]
    preds = []

    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=True
    ):
        preds.append(gt2.make_z_depth(
            radius_state, context["rays_map"], context["padded_shape"],
            context["paddings"], model.interpolation_mode
        ).detach().float().cpu().numpy())

        for t in range(1, len(context["frames"])):
            radius_state, confidence_state, features_state = gt2.fast_step(
                model, context, t, radius_state, features_state
            )
            pred = gt2.make_z_depth(
                radius_state, context["rays_map"], context["padded_shape"],
                context["paddings"], model.interpolation_mode
            )
            preds.append(pred.detach().float().cpu().numpy())

    return preds


def validate_checkpoint(
    checkpoint_path: Path,
    expected_mode: str,
    expected_weights_sha: str,
    expected_pair_sha: str,
) -> dict:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"GT2 检查点不存在：{checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if ckpt.get("mode") != expected_mode:
        raise RuntimeError(
            f"{checkpoint_path.name}: mode={ckpt.get('mode')}, "
            f"预期 {expected_mode}"
        )
    if ckpt.get("clip_indices") != [8, 9, 10] or ckpt.get("target_index") != 10:
        raise RuntimeError(f"{checkpoint_path.name}: GT2 训练片段元数据不符")
    if float(ckpt.get("depth_scale", -1)) != DEPTH_SCALE:
        raise RuntimeError(f"{checkpoint_path.name}: depth_scale 不符")
    if ckpt.get("pretrained_weights_sha256") != expected_weights_sha:
        raise RuntimeError(f"{checkpoint_path.name}: 预训练权重哈希不符")
    if ckpt.get("pair_csv_sha256") != expected_pair_sha:
        raise RuntimeError(f"{checkpoint_path.name}: 配对清单哈希不符")
    state = ckpt.get("repair_head_state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"{checkpoint_path.name}: 缺少 repair_head_state_dict")
    return ckpt


def load_eval_model(
    snapshot: Path,
    mode: str,
    device: torch.device,
    checkpoint_path: Path | None,
    expected_weights_sha: str,
    expected_pair_sha: str,
) -> torch.nn.Module:
    if mode == "A":
        model = gt2.load_model(snapshot, "off", device)
    elif mode in {"C", "D"}:
        model_mode = "pre_readout" if mode == "C" else "post_readout"
        ckpt = validate_checkpoint(
            checkpoint_path, mode, expected_weights_sha, expected_pair_sha
        )
        model = gt2.load_model(snapshot, model_mode, device)
        head = model.pixel_decoder.persistent_feature_repair
        if head is None:
            raise RuntimeError(f"{mode}: 修复头没有注册")
        if sum(p.numel() for p in head.parameters()) != EXPECTED_HEAD_PARAMS:
            raise RuntimeError(f"{mode}: 修复头参数量不符")
        head.load_state_dict(ckpt["repair_head_state_dict"], strict=True)
        configured = getattr(model.pixel_decoder, "feature_repair_mode", None)
        if configured != model_mode:
            raise RuntimeError(
                f"{mode}: checkpoint 与分支不一致，configured={configured}"
            )
    else:
        raise ValueError(mode)

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    if getattr(model.pixel_decoder, "persistent_feature_repair", None) is not None:
        model.pixel_decoder.persistent_feature_repair.eval()
    return model


def aggregate(rows: list[dict], cohort: str) -> dict:
    selected = [
        r for r in rows
        if r["cohort"] == cohort and r["primary_eval"] == "true"
    ]
    modes = ["A", "C", "D"]
    metrics = ["AbsRel", "RMSE", "delta1", "valid_ratio", "log_depth_loss"]
    means = {}
    for mode in modes:
        mr = [r for r in selected if r["mode"] == mode]
        means[mode] = {
            metric: (
                float(np.mean([float(r[metric]) for r in mr]))
                if mr else None
            )
            for metric in metrics
        }

    clip_names = sorted({
        r["clip_name"] for r in selected if r["mode"] == "A"
    })
    per_clip = {}
    for clip_name in clip_names:
        per_clip[clip_name] = {}
        for mode in modes:
            cr = [
                r for r in selected
                if r["clip_name"] == clip_name and r["mode"] == mode
            ]
            per_clip[clip_name][mode] = {
                metric: (
                    float(np.mean([float(r[metric]) for r in cr]))
                    if cr else None
                )
                for metric in metrics
            }

    comparisons = {}
    base = means["A"]
    for mode in ["C", "D"]:
        candidate = means[mode]
        if base["AbsRel"] is None or candidate["AbsRel"] is None:
            rel_abs = None
            rel_rmse = None
        else:
            rel_abs = (
                (base["AbsRel"] - candidate["AbsRel"])
                / max(base["AbsRel"], 1e-12) * 100.0
            )
            rel_rmse = (
                (base["RMSE"] - candidate["RMSE"])
                / max(base["RMSE"], 1e-12) * 100.0
            )
        wins = 0
        valid_clip_count = 0
        for clip_name in clip_names:
            a = per_clip[clip_name]["A"]["AbsRel"]
            c = per_clip[clip_name][mode]["AbsRel"]
            if a is not None and c is not None:
                valid_clip_count += 1
                wins += int(c < a)
        if (
            rel_abs is not None
            and rel_rmse is not None
            and rel_abs > 0
            and rel_rmse >= 0
            and wins >= 3
        ):
            verdict = "PRELIMINARY_POSITIVE"
        elif rel_abs is not None and rel_abs <= 0 and wins < 3:
            verdict = "NOT_SUPPORTED"
        else:
            verdict = "MIXED"
        comparisons[mode] = {
            "relative_absrel_reduction_pct": rel_abs,
            "relative_rmse_reduction_pct": rel_rmse,
            "clips_absrel_better_than_A": wins,
            "clips_evaluated": valid_clip_count,
            "screening_verdict": verdict,
        }
    return {
        "cohort": cohort,
        "primary_frame_count_per_mode": {
            mode: sum(1 for r in selected if r["mode"] == mode)
            for mode in modes
        },
        "mean_metrics_by_mode": means,
        "per_clip_mean_metrics": per_clip,
        "comparisons_vs_A": comparisons,
        "interpretation_limit": (
            "Descriptive small-sample screening only; not a statistical "
            "significance claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GT3 frozen C/D evaluation on held-out Bonn Crowd2 clips"
    )
    parser.add_argument(
        "--pair-csv",
        default="outputs/gt1a1_validation/gt1a1_pairs.csv",
    )
    parser.add_argument(
        "--snapshot",
        default=os.environ.get("VDEEP_SNAPSHOT", ""),
    )
    parser.add_argument(
        "--gt2-dir",
        default="outputs/gt2_validation",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/gt3_validation",
    )
    parser.add_argument(
        "--independent-starts",
        default=",".join(str(x) for x in DEFAULT_INDEPENDENT_STARTS),
    )
    parser.add_argument("--continuation-start", type=int, default=8)
    parser.add_argument("--continuation-end", type=int, default=13)
    parser.add_argument("--max-time-diff-ms", type=float, default=20.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("GT3 需要 CUDA；当前 torch.cuda.is_available() 为 False")
    if not args.snapshot:
        raise RuntimeError("未设置 --snapshot 或 VDEEP_SNAPSHOT")

    pair_csv = Path(args.pair_csv).resolve()
    snapshot = Path(args.snapshot).resolve()
    gt2_dir = Path(args.gt2_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = snapshot / "model.safetensors"
    config_path = snapshot / "config.json"
    if not weights_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"预训练快照不完整：{snapshot}")
    if not pair_csv.is_file():
        raise FileNotFoundError(f"GT1-A1 配对清单不存在：{pair_csv}")

    gt2_summary_path = gt2_dir / "gt2_summary.json"
    if not gt2_summary_path.is_file():
        raise FileNotFoundError(
            f"GT2 汇总不存在：{gt2_summary_path}；请先确认 GT2_review.zip "
            "对应的结果已解压到 outputs/gt2_validation。"
        )
    gt2_summary = json.loads(gt2_summary_path.read_text(encoding="utf-8"))
    if gt2_summary.get("engineering_status") != "PASS":
        raise RuntimeError("GT2 engineering_status 不是 PASS")

    expected_source_hashes = gt2_summary.get("source_sha256", {})
    decoder_path = ROOT / "velodepth/models/decoder_velo.py"
    head_path = ROOT / "velodepth/models/persistent_feature_repair.py"
    expected_decoder_hash = expected_source_hashes.get(
        "velodepth/models/decoder_velo.py"
    )
    expected_head_hash = expected_source_hashes.get(
        "velodepth/models/persistent_feature_repair.py"
    )
    if not expected_decoder_hash or not expected_head_hash:
        raise RuntimeError("GT2 summary 缺少源码哈希，停止")
    if sha256_file(decoder_path) != expected_decoder_hash:
        raise RuntimeError(
            "decoder_velo.py 与 GT2 记录的源码哈希不一致；"
            "不要在不同源码上评估旧检查点。"
        )
    if sha256_file(head_path) != expected_head_hash:
        raise RuntimeError(
            "persistent_feature_repair.py 与 GT2 记录的源码哈希不一致。"
        )

    weights_sha = sha256_file(weights_path)
    pair_sha = sha256_file(pair_csv)
    if gt2_summary.get("pretrained_weights_sha256") != weights_sha:
        raise RuntimeError("预训练权重哈希与 GT2 记录不一致")
    if gt2_summary.get("pair_csv_sha256") != pair_sha:
        raise RuntimeError("GT1-A1 配对清单哈希与 GT2 记录不一致")

    c_path = gt2_dir / "gt2_C_repair_head.pt"
    d_path = gt2_dir / "gt2_D_repair_head.pt"
    c_ckpt = validate_checkpoint(c_path, "C", weights_sha, pair_sha)
    d_ckpt = validate_checkpoint(d_path, "D", weights_sha, pair_sha)
    if c_ckpt.get("initial_head_state_sha256") != d_ckpt.get(
        "initial_head_state_sha256"
    ):
        raise RuntimeError("C/D 检查点不是从同一个初始修复头出发")

    starts = [
        int(x.strip()) for x in args.independent_starts.split(",") if x.strip()
    ]
    if len(starts) < 3:
        raise ValueError("GT3 至少需要 3 个独立留出片段")
    if len(set(starts)) != len(starts):
        raise ValueError("independent-starts 存在重复起点")
    if any(s < 0 for s in starts):
        raise ValueError("片段起点不能为负数")
    continuation_indices = list(
        range(args.continuation_start, args.continuation_end + 1)
    )
    if len(continuation_indices) < 4:
        raise ValueError("连续状态测试至少需要 4 帧")
    if continuation_indices[:3] != [8, 9, 10]:
        raise ValueError(
            "当前 GT3-B 协议要求连续上下文从 GT2 训练片段 8,9,10 开始"
        )
    if continuation_indices[-1] < 11:
        raise ValueError("GT3-B 至少需要包含帧 11")

    independent_index_sets = [
        set(range(s, s + 3)) for s in starts
    ]
    for idx_set in independent_index_sets:
        if idx_set & TRAIN_INDICES:
            raise ValueError(
                f"独立留出片段与 GT2 训练索引重叠：{sorted(idx_set & TRAIN_INDICES)}"
            )
    for i in range(len(independent_index_sets)):
        for j in range(i + 1, len(independent_index_sets)):
            if independent_index_sets[i] & independent_index_sets[j]:
                raise ValueError("独立留出片段彼此重叠")

    by_index = read_pair_index(pair_csv)
    clips = []
    for start in starts:
        name = f"holdout_start_{start:04d}"
        selected = select_records(
            by_index, list(range(start, start + 3)),
            args.max_time_diff_ms, "independent_holdout", name
        )
        clips.append({
            "cohort": "independent_holdout",
            "clip_name": name,
            "indices": list(range(start, start + 3)),
            "primary_indices": [start + 1, start + 2],
            "selected": selected,
        })

    continuation_name = (
        f"continuation_{args.continuation_start:04d}_"
        f"{args.continuation_end:04d}"
    )
    continuation_selected = select_records(
        by_index, continuation_indices, args.max_time_diff_ms,
        "continuation_after_train_anchor", continuation_name
    )
    clips.append({
        "cohort": "continuation_after_train_anchor",
        "clip_name": continuation_name,
        "indices": continuation_indices,
        "primary_indices": [i for i in continuation_indices if i >= 11],
        "selected": continuation_selected,
    })

    torch.set_num_threads(4)
    torch.manual_seed(20261024)
    np.random.seed(20261024)
    torch.cuda.manual_seed_all(20261024)
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    config = {
        "experiment": "GT3 frozen-checkpoint held-out evaluation",
        "pair_csv": str(pair_csv),
        "pair_csv_sha256": pair_sha,
        "snapshot": str(snapshot),
        "pretrained_weights_sha256": weights_sha,
        "gt2_summary": str(gt2_summary_path),
        "gt2_C_checkpoint_sha256": sha256_file(c_path),
        "gt2_D_checkpoint_sha256": sha256_file(d_path),
        "decoder_sha256": sha256_file(decoder_path),
        "repair_head_source_sha256": sha256_file(head_path),
        "depth_scale": DEPTH_SCALE,
        "max_time_diff_ms": args.max_time_diff_ms,
        "independent_starts": starts,
        "independent_clip_indices": [
            list(range(s, s + 3)) for s in starts
        ],
        "continuation_indices": continuation_indices,
        "continuation_primary_indices": [
            i for i in continuation_indices if i >= 11
        ],
        "training_indices": sorted(TRAIN_INDICES),
        "primary_metrics": (
            "independent clips: positions 1 and 2 only; continuation: "
            "indices >=11 only"
        ),
        "gradient_updates": False,
        "modes": ["A:repair_off", "C:pre_readout trained checkpoint",
                  "D:post_readout trained checkpoint"],
        "gpu": torch.cuda.get_device_name(0),
    }
    (out_dir / "gt3_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "gt3_pair_csv_used.csv").write_bytes(pair_csv.read_bytes())

    metric_rows: list[dict] = []
    prediction_cache: dict[tuple[str, str, str, int], np.ndarray] = {}
    run_log = []

    model_specs = [
        ("A", None),
        ("C", c_path),
        ("D", d_path),
    ]
    for mode, checkpoint_path in model_specs:
        print(f"\n========== GT3 mode {mode} ==========", flush=True)
        start_time = time.perf_counter()
        model = load_eval_model(
            snapshot, mode, device, checkpoint_path, weights_sha, pair_sha
        )
        print(
            f"[{mode}] model loaded; checkpoint="
            f"{checkpoint_path.name if checkpoint_path else 'original weights'}",
            flush=True,
        )
        for clip in clips:
            print(
                f"[{mode}] evaluating {clip['clip_name']} "
                f"indices={clip['indices']}",
                flush=True,
            )
            context = build_context(model, clip["selected"], device)
            preds = forward_clip(model, context)
            if len(preds) != len(clip["selected"]):
                raise RuntimeError(f"{clip['clip_name']}: 预测帧数不匹配")

            for pos, (frame, pred) in enumerate(zip(context["frames"], preds)):
                metrics = gt2.calculate_metrics(frame["gt_np"], pred)
                pred_t = torch.from_numpy(pred).to(device)
                with torch.no_grad():
                    loss_t, _ = gt2.masked_log_depth_loss(pred_t, frame["gt"])
                is_primary = frame["index"] in clip["primary_indices"]
                row = {
                    "cohort": clip["cohort"],
                    "clip_name": clip["clip_name"],
                    "mode": mode,
                    "sequence_position": pos,
                    "frame_index_zero_based": frame["index"],
                    "primary_eval": str(bool(is_primary)).lower(),
                    "rgb_file": frame["rgb_path"].name,
                    "gt_file": frame["gt_path"].name,
                    "abs_timestamp_diff_ms": frame["abs_delta_ms"],
                    **metrics,
                    "log_depth_loss": float(loss_t.item()),
                }
                metric_rows.append(row)
                prediction_cache[
                    (clip["cohort"], clip["clip_name"], mode, frame["index"])
                ] = pred

            del context, preds
            torch.cuda.empty_cache()

        elapsed = time.perf_counter() - start_time
        run_log.append({
            "mode": mode,
            "elapsed_sec": elapsed,
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        })
        print(f"[{mode}] finished in {elapsed:.1f} sec", flush=True)
        del model
        torch.cuda.empty_cache()

    # Branch invariants: D must preserve the current readout before the repaired
    # state can affect the next frame. Keyframe output must also remain unchanged.
    branch_checks = []
    max_keyframe_diff = 0.0
    max_d_first_propagated_diff = 0.0
    for clip in clips:
        first_idx = clip["indices"][0]
        a0 = prediction_cache[(clip["cohort"], clip["clip_name"], "A", first_idx)]
        for mode in ("C", "D"):
            m0 = prediction_cache[(clip["cohort"], clip["clip_name"], mode, first_idx)]
            diff = float(np.max(np.abs(a0.astype(np.float32) - m0.astype(np.float32))))
            max_keyframe_diff = max(max_keyframe_diff, diff)
            branch_checks.append({
                "clip_name": clip["clip_name"],
                "check": f"{mode}_keyframe_vs_A",
                "frame_index": first_idx,
                "max_abs_depth_diff_m": diff,
                "pass": diff <= 1e-4,
            })

        # For independent clips, D's first propagated output should match A:
        # D repairs state only after readout, so its effect starts on the next frame.
        if clip["cohort"] == "independent_holdout":
            idx = clip["indices"][1]
            a1 = prediction_cache[(clip["cohort"], clip["clip_name"], "A", idx)]
            d1 = prediction_cache[(clip["cohort"], clip["clip_name"], "D", idx)]
            diff = float(np.max(np.abs(a1.astype(np.float32) - d1.astype(np.float32))))
            max_d_first_propagated_diff = max(max_d_first_propagated_diff, diff)
            branch_checks.append({
                "clip_name": clip["clip_name"],
                "check": "D_first_propagated_frame_vs_A",
                "frame_index": idx,
                "max_abs_depth_diff_m": diff,
                "pass": diff <= 1e-4,
            })

    branch_pass = all(x["pass"] for x in branch_checks)
    independent_summary = aggregate(metric_rows, "independent_holdout")
    continuation_summary = aggregate(
        metric_rows, "continuation_after_train_anchor"
    )

    summary = {
        "experiment": "GT3 frozen-checkpoint held-out evaluation",
        "status": "PASS" if branch_pass else "BRANCH_INVARIANT_FAIL",
        "scope": (
            "Zero-gradient held-out evaluation. Independent clips are the "
            "generalization screen; continuation scores are a separate causal "
            "state-continuation check."
        ),
        "independent_holdout": independent_summary,
        "continuation_after_train_anchor": continuation_summary,
        "branch_invariants": {
            "max_keyframe_abs_diff_m_across_A_C_D": max_keyframe_diff,
            "max_D_first_propagated_abs_diff_m_vs_A": max_d_first_propagated_diff,
            "all_checks_pass": branch_pass,
            "checks": branch_checks,
        },
        "run": run_log,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "important_limitations": [
            "The independent cohort contains five short clips from one sequence; it is a preliminary screen, not broad cross-sequence evidence.",
            "The continuation cohort uses RGB context from GT2 training indices 8,9,10; only later indices 11 onward are scored, and this cohort is not independent generalization.",
            "No GT values are fed into model forward passes and no gradient updates occur.",
            "A lower metric on a small number of frames is descriptive and does not establish statistical significance.",
        ],
    }
    (out_dir / "gt3_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(out_dir / "gt3_frame_metrics.csv", metric_rows)
    write_csv(out_dir / "gt3_branch_checks.csv", branch_checks)

    report = [
        "# GT3 实验报告：冻结检查点的留出片段与连续状态验证",
        "",
        "## 1. 实验目的",
        "",
        "不更新任何模型参数，评估 GT2 训练后的 C/D 修复头在未参与训练的 Bonn Crowd2 帧上的表现，并单独检查从 GT2 训练片段延续到后续帧时的状态传递。",
        "",
        "## 2. 固定协议",
        "",
        f"- 独立留出片段起点：{starts}",
        f"- 独立留出片段：{[list(range(s, s + 3)) for s in starts]}",
        f"- 连续状态片段：{continuation_indices}",
        f"- 连续状态主要评分索引：{[i for i in continuation_indices if i >= 11]}",
        f"- GT 单位：uint16 / {DEPTH_SCALE:.0f} 米；零值无效",
        f"- RGB-GT 最大时间差：{args.max_time_diff_ms:.3f} ms",
        "- 模式：A 原始基线；C 读出前修复；D 读出后修复持久状态",
        "- 所有检查点冻结；不反向传播；不使用 GT 更新模型",
        "- 独立片段主要指标只计算第 2、3 帧（传播帧），不把 keyframe 计入修复收益",
        "- 连续状态测试只计算索引 11 及之后的帧；索引 8、9、10 只作为上下文",
        "",
        "## 3. 独立留出片段结果",
        "",
        "| 模式 | 平均 AbsRel ↓ | 平均 RMSE ↓ (m) | 平均 delta1 ↑ | 相对 A 的 AbsRel 下降 | 优于 A 的片段数 | 初步判定 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for mode in ("A", "C", "D"):
        m = independent_summary["mean_metrics_by_mode"][mode]
        if mode == "A":
            verdict = "基线"
            rel = "—"
            wins = "—"
        else:
            c = independent_summary["comparisons_vs_A"][mode]
            verdict = c["screening_verdict"]
            rel = (
                "N/A" if c["relative_absrel_reduction_pct"] is None
                else f"{c['relative_absrel_reduction_pct']:.2f}%"
            )
            wins = f"{c['clips_absrel_better_than_A']}/{c['clips_evaluated']}"
        report.append(
            f"| {mode} | {m['AbsRel']:.6f} | {m['RMSE']:.6f} | "
            f"{m['delta1']:.6f} | {rel} | {wins} | {verdict} |"
        )
    report += [
        "",
        "### 逐片段 AbsRel",
        "",
        "| 片段 | A | C | D |",
        "|---|---:|---:|---:|",
    ]
    for clip_name, modes in independent_summary["per_clip_mean_metrics"].items():
        report.append(
            f"| {clip_name} | {modes['A']['AbsRel']:.6f} | "
            f"{modes['C']['AbsRel']:.6f} | {modes['D']['AbsRel']:.6f} |"
        )
    cont_means = continuation_summary["mean_metrics_by_mode"]
    report += [
        "",
        "## 4. 连续状态延续结果（与独立留出分开判读）",
        "",
        "| 模式 | 帧 11–13 平均 AbsRel ↓ | 平均 RMSE ↓ (m) | 平均 delta1 ↑ |",
        "|---|---:|---:|---:|",
    ]
    for mode in ("A", "C", "D"):
        m = cont_means[mode]
        report.append(
            f"| {mode} | {m['AbsRel']:.6f} | {m['RMSE']:.6f} | "
            f"{m['delta1']:.6f} |"
        )
    report += [
        "",
        "## 5. 分支语义检查",
        "",
        f"- A/C/D keyframe 最大绝对深度差：{max_keyframe_diff:.8g} m",
        f"- 独立片段中 D 首个传播帧与 A 最大绝对深度差：{max_d_first_propagated_diff:.8g} m",
        f"- 分支不变量：{'PASS' if branch_pass else 'FAIL'}",
        "- D 首个传播帧应与 A 相同，因为 D 在当前帧读出之后才修复状态；其影响应从后续帧开始体现。",
        "",
        "## 6. 结论边界与下一步",
        "",
        "- 本实验是冻结检查点评估，不进行任何训练。",
        "- 五个独立片段仍来自同一个 Bonn Crowd2 序列，属于初步泛化筛查，不代表跨序列泛化。",
        "- 连续状态测试使用 GT2 训练片段的 RGB 作为上下文，因此只用于检查状态影响是否延续，不能与独立留出结果混为一谈。",
        "- 如果某模式在多数独立片段上改善 AbsRel 且 RMSE 不恶化，可以进入跨序列 GT4；若结果混合，应先扩大留出帧覆盖或分析失败片段，不应立即增加训练步数。",
        "- 本报告的 screening verdict 是描述性筛查，不是统计显著性结论。",
        "",
        "## 7. 文件",
        "",
        "- `gt3_frame_metrics.csv`：逐帧 A/C/D 指标",
        "- `gt3_branch_checks.csv`：keyframe 与 D 首个传播帧不变量",
        "- `gt3_summary.json`：协议、汇总、检查点与源码哈希",
        "- `gt3_config.json`：固定测试片段与运行配置",
        "- `gt3_run.log`：执行日志",
        "",
    ]
    (out_dir / "GT3_report.md").write_text("\n".join(report), encoding="utf-8")

    # Keep the review bundle self-contained enough to reproduce checkpoint use.
    archive = out_dir / "GT3_review.zip"
    bundle_files = [
        out_dir / "GT3_report.md",
        out_dir / "gt3_config.json",
        out_dir / "gt3_summary.json",
        out_dir / "gt3_frame_metrics.csv",
        out_dir / "gt3_branch_checks.csv",
        out_dir / "gt3_pair_csv_used.csv",
        out_dir / "gt3_run.log",
        Path(__file__).resolve(),
        c_path,
        d_path,
        gt2_summary_path,
    ]
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in bundle_files:
            if p.is_file():
                if p.parent == out_dir:
                    arcname = f"outputs/gt3_validation/{p.name}"
                elif p.parent == gt2_dir:
                    arcname = f"outputs/gt2_validation/{p.name}"
                else:
                    arcname = f"experiments/p1/{p.name}"
                z.write(p, arcname=arcname)

    print("\n========== GT3 SUMMARY ==========", flush=True)
    print("Report:", out_dir / "GT3_report.md", flush=True)
    print("Frame metrics:", out_dir / "gt3_frame_metrics.csv", flush=True)
    print("Branch checks:", out_dir / "gt3_branch_checks.csv", flush=True)
    print("Review ZIP:", archive, flush=True)
    print("Independent cohort:", json.dumps(
        independent_summary["comparisons_vs_A"], ensure_ascii=False
    ), flush=True)
    print("Continuation cohort:", json.dumps(
        continuation_summary["comparisons_vs_A"], ensure_ascii=False
    ), flush=True)
    print("Branch invariants:", "PASS" if branch_pass else "FAIL", flush=True)
    return 0 if branch_pass else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
