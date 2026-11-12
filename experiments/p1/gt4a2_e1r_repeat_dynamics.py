#!/usr/bin/env python3
"""GT4-A2 E1-R: repeat-repair dynamics diagnostic. No training.

Requires gt4a2_e1_state_causal.py in the same experiments/p1 directory.
Policies use geometry-consistent state commits for both C and D checkpoint
weights so that repair frequency, rather than C/D readout semantics, is varied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "p1"))

import gt4a2_e1_state_causal as e1  # noqa: E402
from velodepth.models.persistent_feature_repair import PersistentFeatureRepairHead  # noqa: E402

exp1 = e1.exp1
gt2 = e1.gt2
REFRESH_INTERVAL = e1.REFRESH_INTERVAL
POLICIES = ("R0_none", "R1_once", "R4_every4", "R1_every")
HORIZONS = tuple(range(1, REFRESH_INTERVAL))  # h=1..31; h=31 is reset control


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    ap = argparse.ArgumentParser(description="E1-R repeat-repair dynamics; no training.")
    ap.add_argument("--snapshot", default=os.environ.get("VDEEP_SNAPSHOT", ""))
    ap.add_argument("--crowd2-pair-csv", default="outputs/gt4a2_exp1/gt4a2_exp1_crowd2_pairs_used.csv")
    ap.add_argument("--crowd3-pair-csv", default="outputs/gt4a2_exp1/gt4a2_exp1_crowd3_pairs_used.csv")
    ap.add_argument("--balloon2-pair-csv", default="outputs/gt4a2_exp1/gt4a2_exp1_balloon2_pairs_used.csv")
    ap.add_argument("--c-checkpoint", default="outputs/gt4a2_c3/gt4a2_c3_C_final_step1000.pt")
    ap.add_argument("--d-checkpoint", default="outputs/gt4a2_c3/gt4a2_c3_D_final_step1000.pt")
    ap.add_argument("--checkpoint", choices=("C", "D", "both"), default="both")
    ap.add_argument("--max-snapshots-per-sequence", type=int, default=0,
                    help="0=all eligible snapshots; otherwise evenly sample per sequence.")
    ap.add_argument("--static-repeats", default="1,2,4",
                    help="Comma-separated static repeated-application counts; default 1,2,4.")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--out-dir", default="outputs/gt4a2_e1r_validation")
    return ap.parse_args()


def parse_repeat_counts(raw: str) -> tuple[int, ...]:
    vals = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not vals or vals[0] < 1 or any(x > 16 for x in vals):
        raise ValueError("--static-repeats must contain integers in [1,16], e.g. 1,2,4")
    return tuple(vals)


def build_preflight(args, static_repeats):
    if not args.snapshot:
        raise RuntimeError("VDEEP_SNAPSHOT is unset; pass --snapshot or export it.")
    snapshot = Path(args.snapshot).resolve()
    if not (snapshot / "config.json").is_file() or not (snapshot / "model.safetensors").is_file():
        raise FileNotFoundError(f"Pretrained snapshot incomplete: {snapshot}")
    source_hashes = e1.verify_source_hashes()
    e1_script = Path(e1.__file__).resolve()
    this_script = Path(__file__).resolve()
    pair_paths = {
        "crowd2": Path(args.crowd2_pair_csv).resolve(),
        "crowd3": Path(args.crowd3_pair_csv).resolve(),
        "balloon2": Path(args.balloon2_pair_csv).resolve(),
    }
    manifests = {}
    for name, path in pair_paths.items():
        rows = exp1.read_manifest(path, training=False)
        targets = e1.build_snapshots(rows, args.max_snapshots_per_sequence)
        manifests[name] = {
            "path": str(path), "sha256": sha256_file(path), "rgb_rows": len(rows),
            "eligible_snapshots": len(targets), "snapshot_indices": [x["snapshot_index"] for x in targets],
        }
    ckpt_paths = {"C": Path(args.c_checkpoint).resolve(), "D": Path(args.d_checkpoint).resolve()}
    selected = ("C", "D") if args.checkpoint == "both" else (args.checkpoint,)
    ckpts = {label: e1.load_checkpoint_metadata(ckpt_paths[label], label) for label in selected}
    return {
        "status": "PREFLIGHT_PASS", "root": str(ROOT), "snapshot": str(snapshot),
        "pretrained_weights_sha256": sha256_file(snapshot / "model.safetensors"),
        "source_hashes": source_hashes,
        "e1_helper_script": str(e1_script), "e1_helper_script_sha256": sha256_file(e1_script),
        "e1r_script": str(this_script), "e1r_script_sha256": sha256_file(this_script),
        "manifests": manifests, "checkpoints": ckpts,
        "protocol": {
            "refresh_interval": REFRESH_INTERVAL,
            "snapshot": "native A state immediately after propagation at cycle_start+1",
            "policies": {
                "R0_none": "no repair at snapshot or during future replay",
                "R1_once": "repair once at snapshot, then disable repair",
                "R4_every4": "repair at snapshot and after future propagation h=4,8,...,28",
                "R1_every": "repair at snapshot and after every future propagation h=1..30",
            },
            "state_commit": "geometry-consistent: repair F and recompute radius/confidence from repaired features",
            "static_probe_repeats": list(static_repeats),
            "future_metrics": "all h=1..31; h=31 is the scheduled keyframe reset control",
            "training_performed": False,
        },
    }


def clone_state(state, device):
    return e1.state_to_device(state, device)


def apply_repair(features, head, model):
    """Apply R_theta once, then read out a geometry-consistent radius/confidence."""
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        repaired = [x.detach().clone() for x in head(features)]
        radius, confidence = e1.decode_radius_from_state_features(model, repaired)
    return radius.detach(), confidence.detach(), [x.detach() for x in repaired]


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float()
    bf = b.detach().float()
    return float((af - bf).norm().item() / max(af.norm().item(), 1e-12))


def feature_rel_l2(features_a, features_b) -> float:
    num = 0.0
    den = 0.0
    for a, b in zip(features_a, features_b):
        num += float((a.detach().float() - b.detach().float()).square().sum().item())
        den += float(a.detach().float().square().sum().item())
    return float(np.sqrt(num) / max(np.sqrt(den), 1e-12))


def static_repeat_probe(seq, checkpoint_label, cycle, t, frame, base_state,
                        head, model, shape, repeat_counts, rows_out):
    """Repeatedly apply R to the same feature state without any P propagation."""
    base_features = [x.detach().clone() for x in base_state["features"]]
    prev_features = [x.detach().clone() for x in base_features]
    prev_radius = base_state["radius"].detach().clone()
    max_repeat = max(repeat_counts)
    wanted = {0, *repeat_counts}
    for k in range(0, max_repeat + 1):
        if k == 0:
            features = [x.detach().clone() for x in base_features]
            radius = base_state["radius"].detach().clone()
            confidence = base_state["confidence"].detach().clone()
        else:
            radius, confidence, features = apply_repair(prev_features, head, model)
        if k in wanted:
            pred = e1.pred_numpy(radius, base_state["rays_map"], shape, model)
            metric = e1.score_prediction(frame, pred)
            fbase = e1.feature_delta_summary(base_features, features)
            rows_out.append({
                "sequence": seq, "checkpoint": checkpoint_label, "snapshot_cycle": cycle,
                "snapshot_index": t, "repeat_count": k,
                "feature_delta_rel_l2_vs_base": fbase["feature_delta_rel_l2"],
                "feature_delta_max_abs_vs_base": fbase["feature_delta_max_abs"],
                "feature_delta_rel_l2_vs_previous": 0.0 if k == 0 else feature_rel_l2(prev_features, features),
                "radius_delta_rel_l2_vs_base": relative_l2(base_state["radius"], radius),
                "radius_delta_max_abs_vs_base_m": e1.max_abs_tensor(base_state["radius"], radius),
                "radius_delta_rel_l2_vs_previous": 0.0 if k == 0 else relative_l2(prev_radius, radius),
                **metric,
            })
        prev_features = [x.detach().clone() for x in features]
        prev_radius = radius.detach().clone()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def replay_policy(model, head, rows, frames, static_by_index, shape, state_index,
                  initial_state, sequence, checkpoint_label, cycle, policy):
    radius = initial_state["radius"].detach().clone()
    confidence = initial_state["confidence"].detach().clone()
    features = [x.detach().clone() for x in initial_state["features"]]
    rays_map = initial_state["rays_map"].detach().clone()
    prev_frame = frames[state_index]
    repair_count = 0
    # R1/R4/every all begin with the same one-time repair at the snapshot.
    if policy != "R0_none":
        radius, confidence, features = apply_repair(features, head, model)
        repair_count += 1
    rows_out = []
    predictions = {}
    with torch.no_grad():
        for h in range(1, REFRESH_INTERVAL):
            idx = state_index + h
            frame = frames[idx]
            is_keyframe = (idx % REFRESH_INTERVAL == 0)
            repair_this_step = False
            feat_delta_event = ""
            radius_delta_event = ""
            radius_max_event = ""
            if is_keyframe:
                # Fixed scheduled refresh: all branches reset identically; no repair at reset.
                rays_map, radius, confidence, features = exp1.keyframe_state(model, frame, shape)
            else:
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    radius, confidence, features = exp1.one_fast_step(
                        model, prev_frame, frame, static_by_index[idx], radius, confidence, features
                    )
                if policy == "R1_every":
                    repair_this_step = True
                elif policy == "R4_every4" and h % 4 == 0:
                    repair_this_step = True
                if repair_this_step:
                    before_features = [x.detach().clone() for x in features]
                    before_radius = radius.detach().clone()
                    radius, confidence, features = apply_repair(features, head, model)
                    repair_count += 1
                    feat_delta_event = feature_rel_l2(before_features, features)
                    radius_delta_event = relative_l2(before_radius, radius)
                    radius_max_event = e1.max_abs_tensor(before_radius, radius)
            pred = e1.pred_numpy(radius, rays_map, shape, model)
            predictions[h] = pred
            metric = e1.score_prediction(frame, pred)
            rows_out.append({
                "sequence": sequence, "checkpoint": checkpoint_label, "snapshot_cycle": cycle,
                "snapshot_index": state_index, "policy": policy, "horizon": h,
                "target_index": idx, "is_scheduled_keyframe": is_keyframe,
                "repair_applied_this_step": repair_this_step,
                "repair_count_total_including_snapshot": repair_count,
                "repair_interval": {"R0_none": 0, "R1_once": 0, "R4_every4": 4, "R1_every": 1}[policy],
                "feature_delta_rel_l2_at_event": feat_delta_event,
                "radius_delta_rel_l2_at_event": radius_delta_event,
                "radius_delta_max_abs_at_event_m": radius_max_event,
                **metric,
            })
            prev_frame = frame
    return rows_out, predictions


def summarize_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("gt_matched") and row.get("AbsRel") != "":
            key = (row["sequence"], row["checkpoint"], row["policy"], int(row["horizon"]))
            groups[key].append(row)
    summary = []
    for (seq, ckpt, policy, h), rr in sorted(groups.items()):
        summary.append({
            "sequence": seq, "checkpoint": ckpt, "policy": policy, "horizon": h,
            "n_snapshot_target_pairs": len(rr),
            "mean_AbsRel": float(np.mean([float(x["AbsRel"]) for x in rr])),
            "median_AbsRel": float(np.median([float(x["AbsRel"]) for x in rr])),
            "mean_RMSE": float(np.mean([float(x["RMSE"]) for x in rr])),
            "mean_delta1": float(np.mean([float(x["delta1"]) for x in rr])),
            "note": "Overlapping snapshots are correlated; descriptive diagnostic only.",
        })
    paired = []
    by_key = defaultdict(dict)
    for row in rows:
        if row.get("gt_matched") and row.get("AbsRel") != "":
            key = (row["sequence"], row["checkpoint"], int(row["snapshot_cycle"]), int(row["horizon"]))
            by_key[key][row["policy"]] = row
    for (seq, ckpt, cycle, h), branches in sorted(by_key.items()):
        if "R0_none" not in branches:
            continue
        base = branches["R0_none"]
        for policy in ("R1_once", "R4_every4", "R1_every"):
            if policy not in branches:
                continue
            cur = branches[policy]
            a = float(base["AbsRel"])
            b = float(cur["AbsRel"])
            paired.append({
                "sequence": seq, "checkpoint": ckpt, "snapshot_cycle": cycle,
                "horizon": h, "policy": policy,
                "R0_AbsRel": a, "policy_AbsRel": b, "delta_AbsRel": b - a,
                "relative_AbsRel_reduction_pct": (a - b) / max(a, 1e-12) * 100.0,
                "delta_RMSE": float(cur["RMSE"]) - float(base["RMSE"]),
                "delta_delta1": float(cur["delta1"]) - float(base["delta1"]),
            })
    return summary, paired


def main():
    args = parse_args()
    static_repeats = parse_repeat_counts(args.static_repeats)
    out_dir = Path(args.out_dir).resolve()
    if (out_dir / "e1r_summary.json").exists() and not args.preflight_only:
        raise RuntimeError(f"Refusing to overwrite existing result: {out_dir / 'e1r_summary.json'}")
    pre = build_preflight(args, static_repeats)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "e1r_preflight.json", pre)
    print(f"[PASS] E1-R preflight: sequences={len(pre['manifests'])}, checkpoints={list(pre['checkpoints'])}", flush=True)
    if args.preflight_only:
        print("E1-R preflight complete; no inference or training performed.", flush=True)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("E1-R inference requires CUDA; no training is performed.")

    device = torch.device("cuda")
    torch.set_num_threads(4)
    torch.manual_seed(20261024)
    np.random.seed(20261024)
    torch.cuda.manual_seed_all(20261024)
    pair_paths = {
        "crowd2": Path(args.crowd2_pair_csv).resolve(),
        "crowd3": Path(args.crowd3_pair_csv).resolve(),
        "balloon2": Path(args.balloon2_pair_csv).resolve(),
    }
    checkpoint_paths = {"C": Path(args.c_checkpoint).resolve(), "D": Path(args.d_checkpoint).resolve()}
    selected = ("C", "D") if args.checkpoint == "both" else (args.checkpoint,)
    dynamic_rows, static_rows, invariant_rows = [], [], []
    start_all = time.perf_counter()

    for seq, pair_path in pair_paths.items():
        rows = exp1.read_manifest(pair_path, training=False)
        targets = e1.build_snapshots(rows, args.max_snapshots_per_sequence)
        model = gt2.load_model(Path(args.snapshot).resolve(), "off", device)
        model.eval()
        shape = exp1.shape_info_for(model, Path(rows[0]["rgb_path"]), device)
        anchors = [{"anchor_index": int(x["anchor_index"]), "refresh_cycle": int(x["cycle"])} for x in targets]
        print(f"[{seq}] caching A states: {len(rows)} frames; snapshots={len(targets)}", flush=True)
        state_cache = exp1.precompute_a_states(model, rows, anchors, shape, device)

        for checkpoint_label in selected:
            checkpoint_path = checkpoint_paths[checkpoint_label]
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            head = PersistentFeatureRepairHead(
                channels=int(model.pixel_decoder.radial_module.hidden_dim), bottleneck=32, levels=4
            ).to(device)
            head.load_state_dict(ckpt["repair_head_state_dict"], strict=True)
            head.eval()
            del ckpt
            print(f"[{seq}/{checkpoint_label}] loaded repair weights; static probes + repeat schedules", flush=True)

            for target in targets:
                cycle = int(target["cycle"])
                t = int(target["snapshot_index"])
                cached = state_cache[int(target["anchor_index"])]
                base_state = clone_state(cached, device)
                matched_t = str(rows[t].get("gt_match_status", "")).strip().upper() == "MATCHED"
                frame_t = exp1.load_frame(rows[t], shape, device, need_gt=matched_t)

                # Static R^k probe: no image transition / no P between applications.
                static_repeat_probe(seq, checkpoint_label, cycle, t, frame_t, base_state,
                                    head, model, shape, static_repeats, static_rows)

                frames = e1.make_window_frames(rows, t, shape, device)
                frames[t] = frame_t
                static_by_index = e1.precompute_window_static(model, frames, t, shape)
                pred_by_policy = {}
                rows_by_policy = {}
                for policy in POLICIES:
                    rr, pp = replay_policy(
                        model, head, rows, frames, static_by_index, shape, t,
                        base_state, seq, checkpoint_label, cycle, policy
                    )
                    rows_by_policy[policy] = rr
                    pred_by_policy[policy] = pp
                    dynamic_rows.extend(rr)

                # Fixed refresh is a hard reset: all policies must converge at h=31.
                h_reset = REFRESH_INTERVAL - 1
                p0 = pred_by_policy["R0_none"][h_reset].astype(np.float32)
                max_reset_diff = max(float(np.max(np.abs(p.astype(np.float32) - p0)))
                                     for p in pred_by_policy.values())
                invariant_rows.append({
                    "sequence": seq, "checkpoint": checkpoint_label,
                    "snapshot_cycle": cycle, "snapshot_index": t,
                    "invariant": "all_policies_equal_at_scheduled_keyframe_h31",
                    "observed_deviation": max_reset_diff,
                    "threshold": 1e-3, "unit": "m", "passed": max_reset_diff <= 1e-3,
                })
                # Verify event counts are exactly as pre-registered before reset.
                expected_counts = {"R0_none": 0, "R1_once": 1, "R4_every4": 8, "R1_every": 31}
                for policy, expected_count in expected_counts.items():
                    actual_count = int(rows_by_policy[policy][-1]["repair_count_total_including_snapshot"])
                    # h31 resets state but the cumulative event count is retained in the trace.
                    invariant_rows.append({
                        "sequence": seq, "checkpoint": checkpoint_label,
                        "snapshot_cycle": cycle, "snapshot_index": t,
                        "invariant": f"repair_count_{policy}",
                        "observed_deviation": abs(actual_count - expected_count),
                        "threshold": 0, "unit": "count", "passed": actual_count == expected_count,
                    })
                del frames, static_by_index, pred_by_policy, rows_by_policy, base_state, frame_t
                torch.cuda.empty_cache()
                print(f"[{seq}/{checkpoint_label}] cycle={cycle} complete", flush=True)
            del head
            torch.cuda.empty_cache()
        del model, state_cache
        torch.cuda.empty_cache()

    summary_rows, paired_rows = summarize_rows(dynamic_rows)
    write_csv(out_dir / "e1r_static_repeat_probe.csv", static_rows)
    write_csv(out_dir / "e1r_future_frame_metrics.csv", dynamic_rows)
    write_csv(out_dir / "e1r_future_summary.csv", summary_rows)
    write_csv(out_dir / "e1r_paired_effects.csv", paired_rows)
    write_csv(out_dir / "e1r_invariants.csv", invariant_rows)

    all_pass = all(bool(x["passed"]) for x in invariant_rows)
    reset_rows = [x for x in invariant_rows if x["invariant"] == "all_policies_equal_at_scheduled_keyframe_h31"]
    reset_max = max((float(x["observed_deviation"]) for x in reset_rows), default=0.0)
    summary = {
        "status": "MECHANICS_PASS" if all_pass else "MECHANICS_FAIL_DO_NOT_INTERPRET",
        "training_performed": False,
        "runtime_sec": time.perf_counter() - start_all,
        "preflight": pre,
        "counts": {
            "dynamic_frame_rows": len(dynamic_rows), "static_probe_rows": len(static_rows),
            "invariant_rows": len(invariant_rows), "summary_rows": len(summary_rows),
            "paired_effect_rows": len(paired_rows),
        },
        "mechanics": {
            "max_policy_difference_at_scheduled_refresh_h31_m": reset_max,
            "all_invariants_pass": all_pass,
            "threshold_m": 1e-3,
        },
        "interpretation_limits": [
            "R1/R4/every policies use geometry-consistent writeback: repair features and recompute radius/confidence.",
            "D checkpoint weights are tested under the same consistent writeback semantics to isolate application frequency; this is not a faithful reproduction of D post-readout semantics.",
            "Static R^k probe repeatedly applies the head to the same feature tensor without propagation; dynamic replay applies repair to evolving propagated states.",
            "h=31 is a scheduled keyframe reset control; interpret h=30 as the last pre-refresh propagation frame.",
            "Crowd2/Crowd3/Balloon2 are development sequences; overlapping snapshots are correlated.",
            "No model training, keyframe budget reduction, latency or compute-saving claim is established.",
        ],
    }
    write_json(out_dir / "e1r_summary.json", summary)
    readme = f"""# E1-R 重复修复动力学结果\n\n状态：**{summary['status']}**\n\n- 是否训练：否\n- 运行时间：{summary['runtime_sec']:.1f} 秒\n- 动态逐帧记录：{len(dynamic_rows)} 行\n- 静态重复探针：{len(static_rows)} 行\n- 关键帧刷新控制最大分支差：{reset_max:.8g} m\n\n## 策略定义\n\n- `R0_none`：快照与未来均不修复。\n- `R1_once`：仅在快照修复一次。\n- `R4_every4`：快照修复一次，之后在 h=4,8,...,28 修复。\n- `R1_every`：快照修复一次，之后每个传播帧 h=1..30 修复。\n- 静态探针在同一快照特征上连续应用修复头 1、2、4 次，不进行传播。\n\nC/D 两个检查点均按一致状态提交（修复 F 并从 F' 重读 r'）进行频率比较，以避免把写回频率和 C/D 的读出语义混在一起。D 权重在此处不是对原始 D post-readout 运行模式的忠实复现。\n\n## 输出\n\n`e1r_static_repeat_probe.csv`、`e1r_future_frame_metrics.csv`、`e1r_future_summary.csv`、`e1r_paired_effects.csv`、`e1r_invariants.csv` 和 `e1r_summary.json`。\n\nh=31 是预定刷新重置控制点；主要长期读数为 h=30。三个序列均为开发序列，不是最终无偏测试集。\n"""
    (out_dir / "E1R_README.md").write_text(readme, encoding="utf-8")
    review_zip = out_dir / "e1r_review.zip"
    include = ["e1r_preflight.json", "e1r_static_repeat_probe.csv", "e1r_future_frame_metrics.csv",
               "e1r_future_summary.csv", "e1r_paired_effects.csv", "e1r_invariants.csv",
               "e1r_summary.json", "E1R_README.md"]
    with zipfile.ZipFile(review_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in include:
            p = out_dir / rel
            if p.is_file():
                zf.write(p, rel)
        zf.write(Path(__file__).resolve(), "gt4a2_e1r_repeat_dynamics.py")
    print(f"[{summary['status']}] E1-R review package: {review_zip}", flush=True)
    return 0 if all_pass else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        raise
