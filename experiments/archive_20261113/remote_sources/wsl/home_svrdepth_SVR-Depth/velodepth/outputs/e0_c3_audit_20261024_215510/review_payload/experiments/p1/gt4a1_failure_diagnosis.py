#!/usr/bin/env python3
"""GT4-A1: frozen A/C/D failure-mode diagnosis on selected GT4 clips.

Reads the existing GT4 reports/metrics, chooses clips by pre-declared
metric-based rules, reruns frozen A/C/D inference, and exports depth/error
maps plus depth-bin and spatial-grid error tables. No training is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
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
MODES = ("A", "C", "D")
DEPTH_EDGES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, float("inf"))
DEPTH_LABELS = ("0-0.5m", "0.5-1m", "1-1.5m", "1.5-2m",
                "2-3m", "3-5m", ">5m")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def verify_inputs(seq_dir: Path, snapshot: Path, gt2_dir: Path):
    cfg_path = seq_dir / "gt4_config.json"
    metric_path = seq_dir / "gt4_frame_metrics.csv"
    summary_path = seq_dir / "gt4_summary.json"
    for p in (cfg_path, metric_path, summary_path):
        if not p.is_file():
            raise FileNotFoundError(f"GT4 输入文件缺失：{p}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not cfg.get("evaluation_pair_csv_sha256"):
        raise RuntimeError(f"{cfg_path}: 缺少测试配对清单哈希")
    pair_csv = Path(cfg["pair_csv"])
    if not pair_csv.is_file():
        # Fallback to the audited copy in the same sequence output folder.
        pair_csv = seq_dir / "gt4_pairs.csv"
    if not pair_csv.is_file():
        pair_csv = seq_dir / "gt4_pair_csv_used.csv"
    if not pair_csv.is_file():
        raise FileNotFoundError(f"GT4 测试配对清单不存在：{pair_csv}")
    if sha256_file(pair_csv) != cfg["evaluation_pair_csv_sha256"]:
        raise RuntimeError(f"GT4 测试配对清单哈希不匹配：{pair_csv}")

    if sha256_file(snapshot / "model.safetensors") != cfg["pretrained_weights_sha256"]:
        raise RuntimeError("预训练权重 SHA-256 与 GT4 配置不一致")
    checks = [
        (ROOT / "velodepth/models/decoder_velo.py", cfg["decoder_sha256"], "decoder_velo.py"),
        (ROOT / "velodepth/models/persistent_feature_repair.py",
         cfg["repair_head_source_sha256"], "persistent_feature_repair.py"),
        (ROOT / "experiments/p1/gt2_fixed_clip_repair_train.py",
         cfg["gt2_training_script_sha256"], "gt2_fixed_clip_repair_train.py"),
        (ROOT / "experiments/p1/gt4_cross_sequence_eval.py",
         cfg["gt4_evaluator_sha256"], "gt4_cross_sequence_eval.py"),
        (ROOT / "experiments/p1/gt4_build_sequence_pairs.py",
         cfg["gt4_pair_builder_sha256"], "gt4_build_sequence_pairs.py"),
    ]
    for p, expected, label in checks:
        if not p.is_file():
            raise FileNotFoundError(f"关键源码缺失：{p}")
        actual = sha256_file(p)
        if actual != expected:
            raise RuntimeError(
                f"{label} 哈希不匹配，停止以避免混用源码版本：\n"
                f"expected={expected}\nactual={actual}"
            )

    gt2_summary_path = gt2_dir / "gt2_summary.json"
    if not gt2_summary_path.is_file():
        raise FileNotFoundError(f"GT2 汇总不存在：{gt2_summary_path}")
    gt2_summary = json.loads(gt2_summary_path.read_text(encoding="utf-8"))
    if gt2_summary.get("engineering_status") != "PASS":
        raise RuntimeError("GT2 engineering_status 不是 PASS")
    train_pair_csv = Path(gt2_summary.get("pair_csv", ""))
    train_pair_sha = str(gt2_summary.get("pair_csv_sha256", ""))
    if not train_pair_csv.is_file() or not train_pair_sha:
        raise RuntimeError("GT2 训练配对清单不存在或缺少哈希")
    if sha256_file(train_pair_csv) != train_pair_sha:
        raise RuntimeError("GT2 训练配对清单哈希不匹配")

    ckpt_paths = {"C": gt2_dir / "gt2_C_repair_head.pt",
                  "D": gt2_dir / "gt2_D_repair_head.pt"}
    for mode, p in ckpt_paths.items():
        if not p.is_file():
            raise FileNotFoundError(f"{mode} 检查点不存在：{p}")
        expected = cfg[f"gt2_{mode}_checkpoint_sha256"]
        if sha256_file(p) != expected:
            raise RuntimeError(f"{mode} 检查点 SHA-256 与 GT4 配置不一致")

    if summary.get("status") != "PASS":
        raise RuntimeError(f"{summary_path}: GT4 status 不是 PASS")
    return cfg, summary, pair_csv, train_pair_csv, train_pair_sha, ckpt_paths, metric_path


def select_clips(metric_path: Path, max_per_group: int = 3):
    rows = read_csv(metric_path)
    primary = [r for r in rows if str(r.get("primary_eval", "")).lower() == "true"]
    by_clip_mode = {}
    for r in primary:
        key = (r["clip_name"], r["mode"])
        by_clip_mode.setdefault(key, []).append(r)

    clips = sorted({r["clip_name"] for r in primary if r["mode"] == "A"})
    stats = {}
    for clip in clips:
        stats[clip] = {}
        for mode in MODES:
            rr = by_clip_mode.get((clip, mode), [])
            if not rr:
                raise RuntimeError(f"GT4 指标缺少 clip/mode：{clip}/{mode}")
            stats[clip][mode] = {
                k: float(np.mean([float(x[k]) for x in rr]))
                for k in ("AbsRel", "RMSE", "delta1", "log_depth_loss")
            }
        a = stats[clip]["A"]["AbsRel"]
        c = stats[clip]["C"]["AbsRel"]
        d = stats[clip]["D"]["AbsRel"]
        stats[clip]["C_gain"] = (a - c) / max(a, 1e-12)
        stats[clip]["D_gain"] = (a - d) / max(a, 1e-12)
        stats[clip]["CD_disagreement"] = abs(stats[clip]["C_gain"] - stats[clip]["D_gain"])

    chosen = {}
    roles = {}
    def add_group(group, ordered, count):
        for clip in ordered[:count]:
            chosen[clip] = True
            roles.setdefault(clip, []).append(group)

    add_group("C_best", sorted(clips, key=lambda x: stats[x]["C_gain"], reverse=True), 3)
    add_group("C_worst", sorted(clips, key=lambda x: stats[x]["C_gain"]), 5)
    add_group("D_best", sorted(clips, key=lambda x: stats[x]["D_gain"], reverse=True), 2)
    add_group("D_worst", sorted(clips, key=lambda x: stats[x]["D_gain"]), 3)
    add_group("C_D_disagreement",
              sorted(clips, key=lambda x: stats[x]["CD_disagreement"], reverse=True), 2)

    selected_rows = []
    for clip in sorted(chosen):
        selected_rows.append({
            "clip_name": clip,
            "selection_roles": ";".join(roles[clip]),
            "A_AbsRel": stats[clip]["A"]["AbsRel"],
            "C_AbsRel": stats[clip]["C"]["AbsRel"],
            "D_AbsRel": stats[clip]["D"]["AbsRel"],
            "C_relative_AbsRel_gain": stats[clip]["C_gain"],
            "D_relative_AbsRel_gain": stats[clip]["D_gain"],
            "C_RMSE": stats[clip]["C"]["RMSE"],
            "D_RMSE": stats[clip]["D"]["RMSE"],
            "C_delta1": stats[clip]["C"]["delta1"],
            "D_delta1": stats[clip]["D"]["delta1"],
        })
    return selected_rows


def colorize(values, vmin, vmax, valid_mask=None, signed=False):
    arr = np.asarray(values, dtype=np.float32)
    norm = np.clip((arr - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
    gray = np.uint8(np.round(norm * 255.0))
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    if valid_mask is not None:
        colored[~valid_mask] = (0, 0, 0)
    return colored


def metrics_for(gt, pred):
    valid = (gt > 0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
    if not valid.any():
        return {"valid_pixels": 0, "pixel_fraction": 0.0,
                "AbsRel": None, "RMSE": None, "delta1": None}
    g = gt[valid].astype(np.float64)
    p = pred[valid].astype(np.float64)
    ratio = np.maximum(g / p, p / g)
    return {
        "valid_pixels": int(valid.sum()),
        "pixel_fraction": float(valid.mean()),
        "AbsRel": float(np.mean(np.abs(g - p) / g)),
        "RMSE": float(np.sqrt(np.mean((g - p) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
    }


def make_montage(frame_dir: Path):
    """Create a compact visual panel: RGB, GT, A/C/D depth, errors and signed deltas."""
    names = [
        ("RGB", "rgb.png"),
        ("GT depth", "gt_depth_vis_0to5m.png"),
        ("A depth", "pred_depth_A_vis_0to5m.png"),
        ("C depth", "pred_depth_C_vis_0to5m.png"),
        ("D depth", "pred_depth_D_vis_0to5m.png"),
        ("A rel.err", "relative_error_A_0to1.png"),
        ("C rel.err", "relative_error_C_0to1.png"),
        ("D rel.err", "relative_error_D_0to1.png"),
        ("C-A rel.err delta", "relative_error_delta_C_minus_A.png"),
        ("D-A rel.err delta", "relative_error_delta_D_minus_A.png"),
    ]
    tiles = []
    tile_w, tile_h, label_h = 320, 240, 28
    for label, filename in names:
        p = frame_dir / filename
        if not p.is_file():
            continue
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            continue
        im = cv2.resize(im, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        tile = np.zeros((tile_h + label_h, tile_w, 3), dtype=np.uint8)
        tile[label_h:, :, :] = im
        cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return
    cols = 5
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.zeros((rows * (tile_h + label_h), cols * tile_w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        y = (i // cols) * (tile_h + label_h)
        x = (i % cols) * tile_w
        canvas[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
    cv2.imwrite(str(frame_dir / "diagnostic_montage.png"), canvas)


def export_frame(frame_dir: Path, frame, predictions: dict[str, np.ndarray], sequence: str):
    frame_dir.mkdir(parents=True, exist_ok=True)
    rgb = gt2.load_rgb(frame["rgb_path"])
    gt = frame["gt_np"].astype(np.float32)
    if rgb.shape[:2] != gt.shape:
        raise RuntimeError(f"RGB/GT 尺寸不一致：{frame['rgb_path']}")
    cv2.imwrite(str(frame_dir / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(frame_dir / "gt_depth_m.npy", gt)
    valid_gt = np.isfinite(gt) & (gt > 0)
    cv2.imwrite(str(frame_dir / "gt_depth_vis_0to5m.png"),
                colorize(gt, 0.0, 5.0, valid_gt))
    write_json(frame_dir / "frame_metadata.json", {
        "sequence": sequence,
        "frame_index_zero_based": int(frame["index"]),
        "rgb_file": str(frame["rgb_path"]),
        "gt_file": str(frame["gt_path"]),
        "rgb_gt_abs_delta_ms": float(frame["abs_delta_ms"]),
        "depth_scale": DEPTH_SCALE,
        "depth_visualization_range_m": [0.0, 5.0],
        "invalid_gt_definition": "GT depth <= 0 or non-finite",
    })

    relative_errors = {}
    abs_errors = {}
    for mode, pred in predictions.items():
        pred = np.asarray(pred, dtype=np.float32).squeeze()
        if pred.shape != gt.shape:
            raise RuntimeError(
                f"{sequence} index={frame['index']} {mode}: pred shape {pred.shape} != GT {gt.shape}"
            )
        np.save(frame_dir / f"pred_depth_{mode}_m.npy", pred)
        valid = valid_gt & np.isfinite(pred) & (pred > 0)
        cv2.imwrite(str(frame_dir / f"pred_depth_{mode}_vis_0to5m.png"),
                    colorize(pred, 0.0, 5.0, valid))
        rel = np.full(gt.shape, np.nan, dtype=np.float32)
        ab = np.full(gt.shape, np.nan, dtype=np.float32)
        rel[valid] = np.abs(pred[valid] - gt[valid]) / gt[valid]
        ab[valid] = np.abs(pred[valid] - gt[valid])
        relative_errors[mode] = rel
        abs_errors[mode] = ab
        cv2.imwrite(str(frame_dir / f"relative_error_{mode}_0to1.png"),
                    colorize(np.nan_to_num(rel, nan=0.0), 0.0, 1.0, valid))
        cv2.imwrite(str(frame_dir / f"absolute_error_{mode}_0to1m.png"),
                    colorize(np.nan_to_num(ab, nan=0.0), 0.0, 1.0, valid))

    common = valid_gt.copy()
    for mode in MODES:
        p = predictions[mode]
        common &= np.isfinite(p) & (p > 0)
    if common.any():
        deltas = [relative_errors[m][common] for m in ("C", "D")]
        scale = float(np.percentile(np.abs(np.concatenate(deltas)), 98))
        scale = min(max(scale, 0.05), 1.0)
        for mode in ("C", "D"):
            delta = relative_errors[mode] - relative_errors["A"]
            vis = colorize(np.nan_to_num(delta, nan=0.0), -scale, scale, common)
            cv2.imwrite(str(frame_dir / f"relative_error_delta_{mode}_minus_A.png"), vis)
            np.save(frame_dir / f"relative_error_delta_{mode}_minus_A.npy", delta)
        write_json(frame_dir / "error_delta_visualization.json", {
            "meaning": "delta = relative_error(candidate) - relative_error(A); negative means candidate improves; positive means candidate worsens",
            "symmetric_color_scale": [-scale, scale],
            "percentile_used": 98,
        })
    make_montage(frame_dir)
    return relative_errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", default="crowd3,balloon2")
    ap.add_argument("--gt4-dir", default="outputs/gt4_validation")
    ap.add_argument("--gt2-dir", default="outputs/gt2_validation")
    ap.add_argument("--snapshot", default=os.environ.get("VDEEP_SNAPSHOT", ""))
    ap.add_argument("--out-dir", default="outputs/gt4a1_diagnostics")
    ap.add_argument("--max-time-diff-ms", type=float, default=20.0)
    ap.add_argument("--grid-size", type=int, default=4)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("GT4-A1 需要 CUDA；torch.cuda.is_available() 为 False")
    if not args.snapshot:
        raise RuntimeError("未设置 --snapshot 或 VDEEP_SNAPSHOT")
    snapshot = Path(args.snapshot).resolve()
    gt4_dir = Path(args.gt4_dir).resolve()
    gt2_dir = Path(args.gt2_dir).resolve()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20261024)
    np.random.seed(20261024)
    torch.cuda.manual_seed_all(20261024)
    device = torch.device("cuda")

    all_selection, all_frame_metrics, all_depth_bins, all_grid = [], [], [], []
    run_manifest = {
        "experiment": "GT4-A1 cross-sequence failure-mode diagnosis",
        "training_updates": 0,
        "ground_truth_used_for_model_forward": False,
        "modes": list(MODES),
        "selection_rule": {
            "C_best": 3, "C_worst": 5, "D_best": 2,
            "D_worst": 3, "C_D_disagreement": 2,
            "basis": "existing GT4 primary-frame per-clip AbsRel; no new test-time tuning",
        },
        "depth_bins_m": list(DEPTH_EDGES),
        "spatial_grid": [args.grid_size, args.grid_size],
        "sequences": {},
    }

    for seq in [x.strip() for x in args.sequences.split(",") if x.strip()]:
        if seq not in {"crowd3", "balloon2"}:
            raise ValueError(f"不支持的序列名：{seq}")
        seq_dir = gt4_dir / seq
        cfg, old_summary, pair_csv, train_pair_csv, train_pair_sha, ckpt_paths, metric_path = verify_inputs(
            seq_dir, snapshot, gt2_dir
        )
        selected_clip_rows = select_clips(metric_path)
        for row in selected_clip_rows:
            row["sequence"] = seq
        all_selection.extend(selected_clip_rows)
        seq_out = out_root / seq
        seq_out.mkdir(parents=True, exist_ok=True)
        write_csv(seq_out / "gt4a1_selected_clips.csv", selected_clip_rows)

        by_index = gt3.read_pair_index(pair_csv)
        clip_records = {}
        for row in selected_clip_rows:
            clip_name = row["clip_name"]
            original_rows = [r for r in read_csv(metric_path)
                             if r["clip_name"] == clip_name and r["mode"] == "A"]
            original_rows.sort(key=lambda r: int(r["sequence_position"]))
            indices = [int(r["frame_index_zero_based"]) for r in original_rows]
            if len(indices) != 3 or indices != list(range(indices[0], indices[0] + 3)):
                raise RuntimeError(f"{clip_name}: GT4 原始记录不是连续三帧：{indices}")
            selected = gt3.select_records(
                by_index, indices, args.max_time_diff_ms,
                "GT4_A1_diagnostic", clip_name
            )
            clip_records[clip_name] = selected

        run_manifest["sequences"][seq] = {
            "gt4_config": str((seq_dir / "gt4_config.json").resolve()),
            "gt4_metrics_sha256": sha256_file(metric_path),
            "pair_csv_sha256": sha256_file(pair_csv),
            "selected_clip_count": len(selected_clip_rows),
            "selected_clips": selected_clip_rows,
        }

        # Store all predictions before creating signed error-difference maps.
        predictions_by_frame = {}
        frame_meta_by_key = {}
        for mode in MODES:
            print(f"\n========== GT4-A1 sequence={seq} mode={mode} ==========", flush=True)
            ckpt_path = ckpt_paths.get(mode) if mode in {"C", "D"} else None
            model = gt3.load_eval_model(
                snapshot, mode, device, ckpt_path,
                cfg["pretrained_weights_sha256"], train_pair_sha
            )
            for clip_name, selected in clip_records.items():
                print(f"[{seq}/{mode}] {clip_name} indices={[r['index'] for r in selected]}", flush=True)
                context = gt3.build_context(model, selected, device)
                preds = gt3.forward_clip(model, context)
                if len(preds) != 3:
                    raise RuntimeError(f"{clip_name}/{mode}: 预期 3 帧预测，实际 {len(preds)}")
                for pos, (frame, pred) in enumerate(zip(context["frames"], preds)):
                    idx = int(frame["index"])
                    key = (clip_name, idx)
                    predictions_by_frame.setdefault(key, {})[mode] = np.asarray(pred).squeeze().astype(np.float32)
                    # Retain only lightweight metadata/GT arrays, not GPU tensors
                    # or preprocessed inputs from every clip.
                    frame_meta_by_key[key] = {
                        "index": frame["index"],
                        "rgb_path": frame["rgb_path"],
                        "gt_path": frame["gt_path"],
                        "gt_np": frame["gt_np"],
                        "abs_delta_ms": frame["abs_delta_ms"],
                        "sequence_position": pos,
                        "primary_eval": pos in (1, 2),
                    }
                del context, preds
                torch.cuda.empty_cache()
            del model
            torch.cuda.empty_cache()

        for (clip_name, idx), pred_map in sorted(predictions_by_frame.items()):
            frame = frame_meta_by_key[(clip_name, idx)]
            frame_dir = seq_out / "frames" / clip_name / f"frame_{idx:04d}"
            relative_errors = export_frame(frame_dir, frame, pred_map, seq)
            gt = frame["gt_np"].astype(np.float32)
            primary = bool(frame["primary_eval"])
            for mode in MODES:
                pred = pred_map[mode]
                m = gt2.calculate_metrics(gt, pred)
                valid = (gt > 0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
                row = {
                    "sequence": seq, "clip_name": clip_name, "frame_index_zero_based": idx,
                    "sequence_position": frame["sequence_position"], "primary_eval": str(primary).lower(),
                    "mode": mode, "rgb_file": Path(frame["rgb_path"]).name,
                    "gt_file": Path(frame["gt_path"]).name, **m,
                }
                all_frame_metrics.append(row)
                if primary:
                    for b, label in enumerate(DEPTH_LABELS):
                        lo, hi = DEPTH_EDGES[b], DEPTH_EDGES[b + 1]
                        mask = valid & (gt >= lo) & (gt < hi)
                        if mask.any():
                            sub = metrics_for(gt[mask], pred[mask])
                            all_depth_bins.append({
                                "sequence": seq, "clip_name": clip_name, "frame_index_zero_based": idx,
                                "mode": mode, "depth_bin": label, "depth_min_m": lo,
                                "depth_max_m": "inf" if np.isinf(hi) else hi,
                                **sub,
                            })
                    h, w = gt.shape
                    gs = args.grid_size
                    for gy in range(gs):
                        y0, y1 = round(gy*h/gs), round((gy+1)*h/gs)
                        for gx in range(gs):
                            x0, x1 = round(gx*w/gs), round((gx+1)*w/gs)
                            sub = metrics_for(gt[y0:y1, x0:x1], pred[y0:y1, x0:x1])
                            all_grid.append({
                                "sequence": seq, "clip_name": clip_name, "frame_index_zero_based": idx,
                                "mode": mode, "grid_rows": gs, "grid_cols": gs,
                                "grid_y": gy, "grid_x": gx, **sub,
                            })

        # Add selected-frame A/C/D metric summary to per-sequence report.
        seq_rows = [r for r in all_frame_metrics if r["sequence"] == seq and r["primary_eval"] == "true"]
        seq_summary = {}
        for mode in MODES:
            rr = [r for r in seq_rows if r["mode"] == mode]
            seq_summary[mode] = {
                metric: float(np.mean([float(r[metric]) for r in rr]))
                for metric in ("AbsRel", "RMSE", "delta1")
            }
        run_manifest["sequences"][seq]["rerun_metrics"] = seq_summary
        write_json(seq_out / "gt4a1_summary.json", run_manifest["sequences"][seq])
        lines = [
            f"# GT4-A1 失败模式诊断：{seq}", "",
            f"- 选定片段数：{len(selected_clip_rows)}；每段 3 帧。",
            "- A/C/D 检查点冻结；不更新参数；GT 不进入模型前向。",
            "- 每个片段输出 RGB、GT、A/C/D 预测深度、相对误差、绝对误差和相对误差差分图。",
            "- 深度图可视化使用统一 0–5 m 色标；误差差分图中负值表示候选分支优于 A，正值表示劣于 A。",
            "", "## 选定片段", "",
            "|片段|选择原因|A AbsRel|C AbsRel|D AbsRel|C 相对收益|D 相对收益|",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for r in selected_clip_rows:
            lines.append(
                f"|{r['clip_name']}|{r['selection_roles']}|{float(r['A_AbsRel']):.5f}|"
                f"{float(r['C_AbsRel']):.5f}|{float(r['D_AbsRel']):.5f}|"
                f"{100*float(r['C_relative_AbsRel_gain']):+.1f}%|"
                f"{100*float(r['D_relative_AbsRel_gain']):+.1f}%|"
            )
        lines += ["", "## 重新推理指标（仅所选片段的评分帧）", "",
                  "|模式|AbsRel ↓|RMSE ↓ (m)|delta1 ↑|评分帧数|",
                  "|---|---:|---:|---:|---:|"]
        for mode in MODES:
            rr = [r for r in seq_rows if r["mode"] == mode]
            lines.append(
                f"|{mode}|{seq_summary[mode]['AbsRel']:.6f}|{seq_summary[mode]['RMSE']:.6f}|"
                f"{seq_summary[mode]['delta1']:.6f}|{len(rr)}|"
            )
        lines += [
            "", "## 结果解读边界", "",
            "片段由既有 GT4 的 AbsRel 结果按预先规定的 best/worst/disagreement 规则筛选，因此这些片段用于失败分析，不能当作无偏的新测试集平均性能。",
            "深度区间和空间网格统计用于定位误差来源，不用于选择测试时分支或调整模型。",
        ]
        (seq_out / "GT4_A1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_csv(out_root / "gt4a1_selected_clips_all.csv", all_selection)
    write_csv(out_root / "gt4a1_frame_metrics.csv", all_frame_metrics)
    write_csv(out_root / "gt4a1_depth_bin_metrics.csv", all_depth_bins)
    write_csv(out_root / "gt4a1_spatial_grid_metrics.csv", all_grid)

    # Descriptive summaries by GT-depth bin and spatial cell. These summarize
    # diagnostic clips only and are not unbiased test-set performance estimates.
    def summarize(rows, group_fields):
        grouped = {}
        for r in rows:
            key = tuple(r[k] for k in group_fields)
            grouped.setdefault(key, []).append(r)
        out = []
        for key, group in sorted(grouped.items(), key=lambda x: tuple(str(v) for v in x[0])):
            valid = [r for r in group if r.get("AbsRel") is not None]
            row = {k: v for k, v in zip(group_fields, key)}
            row["frames_or_cells"] = len(group)
            row["mean_valid_pixels"] = float(np.mean([int(r["valid_pixels"]) for r in group]))
            for metric in ("AbsRel", "RMSE", "delta1"):
                vals = [float(r[metric]) for r in valid if r.get(metric) is not None]
                row[f"mean_{metric}"] = float(np.mean(vals)) if vals else None
            out.append(row)
        return out

    write_csv(out_root / "gt4a1_depth_bin_summary.csv",
              summarize(all_depth_bins, ["sequence", "mode", "depth_bin"]))
    write_csv(out_root / "gt4a1_spatial_grid_summary.csv",
              summarize(all_grid, ["sequence", "mode", "grid_y", "grid_x"]))
    write_json(out_root / "gt4a1_manifest.json", run_manifest)
    print("\nGT4-A1 diagnostic inference complete.", flush=True)
    print("Output:", out_root, flush=True)
    print("Selected clips:", len(all_selection), flush=True)
    print("Frame metric rows:", len(all_frame_metrics), flush=True)
    print("Depth-bin rows:", len(all_depth_bins), flush=True)
    print("Spatial-grid rows:", len(all_grid), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
