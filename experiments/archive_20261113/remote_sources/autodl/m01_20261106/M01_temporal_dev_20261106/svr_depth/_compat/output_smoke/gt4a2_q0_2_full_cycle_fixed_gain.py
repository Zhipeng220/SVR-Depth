#!/usr/bin/env python3
"""SVR-Depth Q0-2: full-sequence repair-frequency sweep, frozen C3 heads only.

No model training. The learned repair head is manually applied after the current
frame readout, and only the persistent feature maps are committed (F-only).
C and D checkpoint weights are crossed under the same writeback semantics.
The historical C3 full-sequence CSV supplies A and native C/D reference rows.
"""
from __future__ import annotations

import argparse
import csv
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

import gt4a2_exp1_train_eval as exp1  # noqa: E402
import gt2_fixed_clip_repair_train as gt2  # noqa: E402

REFRESH_INTERVAL = 32
SEQUENCE_PATHS = {
    "crowd2": "outputs/gt4a2_exp1/gt4a2_exp1_crowd2_pairs_used.csv",
    "crowd3": "outputs/gt4a2_exp1/gt4a2_exp1_crowd3_pairs_used.csv",
    "balloon2": "outputs/gt4a2_exp1/gt4a2_exp1_balloon2_pairs_used.csv",
}
CHECKPOINT_PATHS = {
    "C": "outputs/gt4a2_c3/gt4a2_c3_C_final_step1000.pt",
    "D": "outputs/gt4a2_c3/gt4a2_c3_D_final_step1000.pt",
}
EXPECTED_SOURCE_HASHES = {
    "experiments/p1/gt4a2_c3_closed_loop.py":
        "76327208acac4543b2a87ccba8b7af435e4c5ce65267b42052e51b435cbad7f5",
    "experiments/p1/gt4a2_exp1_train_eval.py":
        "c24441cb405465c1a16a86127b51aade8f9af69b5d368f43d830d0e759ee3d7f",
    "experiments/p1/gt2_fixed_clip_repair_train.py":
        "bf7cd2021b1d0bbf66d75502dba6afed87f4c8ff3ef58ff7b18f56ac28044192",
    "velodepth/models/decoder_velo.py":
        "35ca12629accd1a2a77595f505165fa17dffab07c763a8b639604285b94bdb08",
    "velodepth/models/persistent_feature_repair.py":
        "6dfc8b0c5565ea9139b8c266c5ef337b3309ce0fde2f365428fc44390b84117c",
}
EXPECTED_CHECKPOINT_HASHES = {
    "C": "b1cd8f88a5bd45d75af1a6e3050ee1b5188fb545291b5c62ec304038839ae8d3",
    "D": "3ec06cd9ed9128f8a21afa6321312c3449414326a4e50580b939128bcba84f98",
}
POLICIES = ("R1_G0.25_F_only", "R1_G0.5_F_only", "R1_G1.0_F_only")
POLICY_GAINS = {
    "R1_G0.25_F_only": 0.25,
    "R1_G0.5_F_only": 0.5,
    "R1_G1.0_F_only": 1.0,
}
POLICY_DESC = {
    "R1_G0.25_F_only": "every propagation frame: F_next = F + 0.25 * (RepairHead(F) - F)",
    "R1_G0.5_F_only": "every propagation frame: F_next = F + 0.50 * (RepairHead(F) - F)",
    "R1_G1.0_F_only": "every propagation frame: F_next = RepairHead(F), full-gain R1 control",
}
METRICS = ("AbsRel", "RMSE", "delta1")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    ap = argparse.ArgumentParser(description="Q0-2 full-cycle fixed-gain sweep; no training.")
    ap.add_argument("--snapshot", default=os.environ.get("VDEEP_SNAPSHOT", ""))
    ap.add_argument("--crowd2-pair-csv", default=SEQUENCE_PATHS["crowd2"])
    ap.add_argument("--crowd3-pair-csv", default=SEQUENCE_PATHS["crowd3"])
    ap.add_argument("--balloon2-pair-csv", default=SEQUENCE_PATHS["balloon2"])
    ap.add_argument("--c-checkpoint", default=CHECKPOINT_PATHS["C"])
    ap.add_argument("--d-checkpoint", default=CHECKPOINT_PATHS["D"])
    ap.add_argument("--c3-frame-csv", default="outputs/gt4a2_c3/gt4a2_c3_full_sequence_frame_metrics.csv")
    ap.add_argument("--checkpoint", choices=("C", "D", "both"), default="both")
    ap.add_argument("--sequences", default="crowd2,crowd3,balloon2",
                    help="Comma-separated subset of crowd2,crowd3,balloon2.")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Smoke only: cap each selected sequence; 0 means full sequence.")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--out-dir", default="outputs/gt4a2_q0_2_validation")
    ap.add_argument("--metric-tolerance", type=float, default=1e-5,
                    help="Tolerance for D/R1 vs historical C3 native-D metric reproduction.")
    ap.add_argument("--keyframe-pixel-tolerance-m", type=float, default=1e-3)
    return ap.parse_args()


def parse_sequences(raw: str) -> list[str]:
    values = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("--sequences must be unique names from crowd2,crowd3,balloon2")
    unknown = sorted(set(values) - set(SEQUENCE_PATHS))
    if unknown:
        raise ValueError(f"Unknown sequence(s): {unknown}")
    return values


def verify_sources() -> dict:
    out = {}
    for rel, expected in EXPECTED_SOURCE_HASHES.items():
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"Required audited source missing: {path}")
        actual = sha256_file(path)
        out[rel] = {"expected": expected, "actual": actual, "match": actual == expected}
        if actual != expected:
            raise RuntimeError(f"Source hash mismatch; refusing Q0-2: {rel}: {actual} != {expected}")
    return out


def load_c3_reference(path: Path, manifests: dict[str, list[dict]], max_frames: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Historical C3 frame metrics CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        raise RuntimeError(f"Historical C3 frame metrics CSV is empty: {path}")
    by_seq_mode = defaultdict(dict)
    for r in raw:
        seq = str(r["sequence"]).strip().lower()
        mode = str(r["mode"]).strip()
        idx = int(r["frame_index_zero_based"])
        if idx in by_seq_mode[(seq, mode)]:
            raise RuntimeError(f"Duplicate C3 frame row: {seq}/{mode}/{idx}")
        by_seq_mode[(seq, mode)][idx] = r

    normalized = {}
    for seq, manifest in manifests.items():
        n = len(manifest)
        expected_indices = list(range(n))
        for mode in ("A", "C", "D"):
            got = by_seq_mode.get((seq, mode), {})
            if sorted(got) != expected_indices:
                raise RuntimeError(
                    f"C3 reference rows do not match manifest for {seq}/{mode}: "
                    f"expected {n} contiguous frames, got {len(got)}"
                )
            for idx, rec in enumerate(manifest):
                ref = got[idx]
                rgb_name = Path(rec["rgb_path"]).name
                if str(ref.get("rgb_file", "")) != rgb_name:
                    raise RuntimeError(
                        f"C3 RGB mismatch {seq}/{mode}/{idx}: {ref.get('rgb_file')} != {rgb_name}"
                    )
                status = str(rec.get("gt_match_status", "")).strip().upper()
                expected_matched = status == "MATCHED"
                expected_gt = Path(rec["depth_path"]).name if expected_matched else ""
                actual_gt = str(ref.get("gt_file", "") or "")
                ref_scored = str(ref.get("AbsRel", "")).strip() != ""
                if actual_gt != expected_gt:
                    raise RuntimeError(
                        f"C3 GT pairing mismatch {seq}/{mode}/{idx}: {actual_gt!r} != {expected_gt!r}"
                    )
                if ref_scored != expected_matched:
                    raise RuntimeError(
                        f"C3 scoring/matching mismatch {seq}/{mode}/{idx}: "
                        f"reference_scored={ref_scored}, manifest_matched={expected_matched}"
                    )
        limit = max_frames if max_frames > 0 else n
        if limit > n:
            raise ValueError(f"--max-frames={limit} exceeds {seq} length {n}")
        normalized[seq] = {}
        for mode in ("A", "C", "D"):
            normalized[seq][mode] = [by_seq_mode[(seq, mode)][i] for i in range(limit)]
    return normalized


def normalize_reference_rows(seq: str, refs: dict, manifests: dict[str, list[dict]],
                             max_frames: int) -> list[dict]:
    out = []
    manifest = manifests[seq][:max_frames] if max_frames > 0 else manifests[seq]
    for mode in ("A", "C", "D"):
        for idx, ref in enumerate(refs[mode]):
            age = idx % REFRESH_INTERVAL
            is_keyframe = age == 0
            matched = str(ref.get("AbsRel", "")).strip() != ""
            if mode == "A":
                checkpoint, policy, phase = "REF", "A_ref", "none"
            else:
                checkpoint, policy = mode, f"C3_native_{mode}"
                phase = "pre_readout" if mode == "C" else "post_readout"
            repair_this = (not is_keyframe) if mode in ("C", "D") else False
            out.append({
                "sequence": seq, "checkpoint": checkpoint, "policy": policy,
                "commit_semantics": phase, "frame_index_zero_based": idx,
                "refresh_cycle": idx // REFRESH_INTERVAL, "frame_offset_in_cycle": age,
                "is_keyframe": str(is_keyframe).lower(), "gt_matched": str(matched).lower(),
                "rgb_file": Path(manifest[idx]["rgb_path"]).name,
                "gt_file": Path(manifest[idx]["depth_path"]).name if matched else "",
                "AbsRel": ref.get("AbsRel", ""), "RMSE": ref.get("RMSE", ""),
                "delta1": ref.get("delta1", ""), "valid_ratio": ref.get("valid_ratio", ""),
                "gt_valid_ratio": ref.get("gt_valid_ratio", ""),
                "valid_pixels": ref.get("valid_pixels", ""), "total_pixels": ref.get("total_pixels", ""),
                "repair_applied_this_frame": str(repair_this).lower(),
                "repair_count_cycle": age if (mode in ("C", "D") and not is_keyframe) else 0,
                "repair_count_cumulative": (idx // REFRESH_INTERVAL) * 31 + age if mode in ("C", "D") else 0,
                "feature_delta_rel_l2_at_write": "", "feature_delta_max_abs_at_write": "",
                "frame_wall_sec": "", "run_wall_sec": "", "run_source": "historical_C3_frame_metrics",
            })
    return out


def checkpoint_metadata(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"C3 checkpoint missing: {path}")
    actual = sha256_file(path)
    expected_hash = EXPECTED_CHECKPOINT_HASHES[label]
    if actual != expected_hash:
        raise RuntimeError(f"{label} checkpoint hash mismatch: {actual} != {expected_hash}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    expected_mode = "pre_readout" if label == "C" else "post_readout"
    if ckpt.get("mode_label") != label or ckpt.get("mode") != expected_mode:
        raise RuntimeError(f"{label} checkpoint metadata mismatch: {ckpt.get('mode_label')}/{ckpt.get('mode')}")
    if int(ckpt.get("step_count", -1)) != 1000 or "repair_head_state_dict" not in ckpt:
        raise RuntimeError(f"{label} checkpoint is not the audited final C3 step-1000 checkpoint")
    meta = {"label": label, "path": str(path), "sha256": actual,
            "mode": expected_mode, "step_count": int(ckpt["step_count"]),
            "repair_head_state_keys": len(ckpt["repair_head_state_dict"])}
    del ckpt
    return meta


def build_preflight(args, sequences: list[str]) -> tuple[dict, dict[str, list[dict]], dict]:
    if not args.snapshot:
        raise RuntimeError("VDEEP_SNAPSHOT is unset; pass --snapshot or export it.")
    snapshot = Path(args.snapshot).resolve()
    if not (snapshot / "config.json").is_file() or not (snapshot / "model.safetensors").is_file():
        raise FileNotFoundError(f"Pretrained snapshot incomplete: {snapshot}")
    if args.max_frames < 0 or (args.max_frames > 0 and args.max_frames < 33):
        raise ValueError("--max-frames must be 0 (full) or >=33 so Q0-2 smoke includes a refresh reset")
    source_hashes = verify_sources()
    pair_arg_paths = {
        "crowd2": Path(args.crowd2_pair_csv).resolve(),
        "crowd3": Path(args.crowd3_pair_csv).resolve(),
        "balloon2": Path(args.balloon2_pair_csv).resolve(),
    }
    manifests = {}
    pair_meta = {}
    for seq in sequences:
        path = pair_arg_paths[seq]
        rows = exp1.read_manifest(path, training=False)
        manifests[seq] = rows
        pair_meta[seq] = {"path": str(path), "sha256": sha256_file(path), "rgb_rows": len(rows),
                          "effective_frames": min(len(rows), args.max_frames) if args.max_frames else len(rows)}
    c3_path = Path(args.c3_frame_csv).resolve()
    refs = load_c3_reference(c3_path, manifests, args.max_frames)
    checkpoint_arg_paths = {"C": Path(args.c_checkpoint).resolve(), "D": Path(args.d_checkpoint).resolve()}
    selected = ("C", "D") if args.checkpoint == "both" else (args.checkpoint,)
    ckpts = {label: checkpoint_metadata(checkpoint_arg_paths[label], label) for label in selected}
    pre = {
        "status": "PREFLIGHT_PASS", "root": str(ROOT), "snapshot": str(snapshot),
        "pretrained_weights_sha256": sha256_file(snapshot / "model.safetensors"),
        "source_hashes": source_hashes, "pair_manifests": pair_meta,
        "c3_reference_csv": {"path": str(c3_path), "sha256": sha256_file(c3_path)},
        "checkpoints": ckpts,
        "protocol": {
            "experiment": "Q0-2 full-sequence fixed-gain sweep", "refresh_interval": REFRESH_INTERVAL,
            "training_performed": False, "sequences": sequences,
            "max_frames_per_sequence": args.max_frames or "full_sequence",
            "historical_A_C_D_reference_reused": True,
            "gain_policies": POLICY_DESC,
            "writeback": "F-only post-readout residual scaling: F_next = F + alpha*(RepairHead(F)-F); current radius/confidence stay unchanged",
            "C_and_D_heads_crossed_under_same_writeback_semantics": True,
            "keyframe_reset_expected_to_be_identical_across_policies": True,
             "gain_values": POLICY_GAINS,
            "D_R1_G1.0_expected_to_reproduce_historical_C3_native_D": True,
        },
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    return pre, manifests, refs


def should_repair(policy: str, age: int) -> bool:
    if policy not in POLICY_GAINS:
        raise ValueError(f"Unknown Q0-2 policy: {policy}")
    return age > 0


def gain_for_policy(policy: str) -> float:
    try:
        return POLICY_GAINS[policy]
    except KeyError as exc:
        raise ValueError(f"Unknown Q0-2 policy: {policy}") from exc


def feature_delta(before: list[torch.Tensor], after: list[torch.Tensor]) -> tuple[float, float]:
    delta_sq = 0.0
    base_sq = 0.0
    max_abs = 0.0
    for a, b in zip(before, after):
        d = b.detach().float() - a.detach().float()
        delta_sq += float(d.square().sum().item())
        base_sq += float(a.detach().float().square().sum().item())
        max_abs = max(max_abs, float(d.abs().max().item()))
    return float(np.sqrt(delta_sq) / max(np.sqrt(base_sq), 1e-12)), max_abs


def load_crossed_head_model(snapshot: Path, checkpoint_path: Path, checkpoint_label: str,
                            device: torch.device):
    # Build the audited head architecture, then disable automatic C/D routing.
    model = gt2.load_model(snapshot, "pre_readout", device)
    head = model.pixel_decoder.persistent_feature_repair
    if head is None:
        raise RuntimeError("Repair head module was not registered by the audited model config")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_mode = "pre_readout" if checkpoint_label == "C" else "post_readout"
    if ckpt.get("mode_label") != checkpoint_label or ckpt.get("mode") != expected_mode:
        raise RuntimeError(f"Checkpoint metadata mismatch for {checkpoint_label}")
    head.load_state_dict(ckpt["repair_head_state_dict"], strict=True)
    del ckpt
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    model.pixel_decoder.persistent_feature_repair = None
    model.pixel_decoder.feature_repair_mode = "off"
    model.eval()
    return model, head


def run_policy(seq: str, policy: str, checkpoint_label: str, model, head,
               rows: list[dict], shape: dict, device: torch.device) -> tuple[list[dict], dict, dict]:
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    prev_frame = None
    radius = confidence = features = rays_map = None
    out_rows = []
    keyframe_predictions = {}
    repair_count_total = 0
    repair_count_cycle = 0
    start_run = time.perf_counter()

    with torch.no_grad():
        for idx, rec in enumerate(rows):
            frame_start = time.perf_counter()
            matched = str(rec.get("gt_match_status", "")).strip().upper() == "MATCHED"
            frame = exp1.load_frame(rec, shape, device, need_gt=matched)
            age = idx % REFRESH_INTERVAL
            cycle = idx // REFRESH_INTERVAL
            is_keyframe = age == 0
            repair_this = False
            feature_delta_rel = ""
            feature_delta_max = ""

            if is_keyframe:
                rays_map, radius, confidence, features = exp1.keyframe_state(model, frame, shape)
                repair_count_cycle = 0
            else:
                static = exp1.transition_static(model, prev_frame, frame, shape, dis)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    radius, confidence, features = exp1.one_fast_step(
                        model, prev_frame, frame, static, radius, confidence, features
                    )
                    repair_this = should_repair(policy, age)
                    if repair_this:
                        before = [x.detach().clone() for x in features]
                        # PersistentFeatureRepairHead.forward() returns x + block(x).
                        # Access the audited residual blocks directly so gain is applied
                        # to delta(x), avoiding subtraction of the rounded x+delta output
                        # under autocast: F_next = F + alpha * block(F).
                        # Current-frame radius/confidence remain unchanged.
                        alpha = gain_for_policy(policy)
                        if len(features) != len(head.blocks):
                            raise RuntimeError(
                                f"Repair head level mismatch: features={len(features)}, blocks={len(head.blocks)}"
                            )
                        residuals = [block(x) for x, block in zip(features, head.blocks)]
                        features = [
                            (x + alpha * delta).detach()
                            for x, delta in zip(features, residuals)
                        ]
                        feature_delta_rel, feature_delta_max = feature_delta(before, features)
                        repair_count_total += 1
                        repair_count_cycle += 1
                        del before, residuals
                del static

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                pred_t = gt2.make_z_depth(radius, rays_map, shape["padded_shape"],
                                          shape["paddings"], model.interpolation_mode)
            pred_np = pred_t.detach().float().cpu().numpy()
            if is_keyframe:
                keyframe_predictions[idx] = pred_np.copy()

            if matched:
                metrics = gt2.calculate_metrics(frame["gt_np"], pred_np)
                metric_values = {k: metrics[k] for k in METRICS}
                valid_ratio = metrics.get("valid_ratio", "")
                gt_valid_ratio = metrics.get("gt_valid_ratio", "")
                valid_pixels = metrics.get("valid_pixels", "")
                total_pixels = metrics.get("total_pixels", "")
            else:
                metric_values = {k: "" for k in METRICS}
                valid_ratio = gt_valid_ratio = valid_pixels = total_pixels = ""

            out_rows.append({
                "sequence": seq, "checkpoint": checkpoint_label, "policy": policy,
                "commit_semantics": "F_only_post_readout_residual_gain",
                "frame_index_zero_based": idx, "refresh_cycle": cycle, "frame_offset_in_cycle": age,
                "is_keyframe": str(is_keyframe).lower(), "gt_matched": str(matched).lower(),
                "rgb_file": frame["rgb_file"],
                "gt_file": Path(rec["depth_path"]).name if matched else "",
                **metric_values, "valid_ratio": valid_ratio, "gt_valid_ratio": gt_valid_ratio,
                "valid_pixels": valid_pixels, "total_pixels": total_pixels,
                "repair_applied_this_frame": str(repair_this).lower(),
                "repair_gain": gain_for_policy(policy) if repair_this else "",
                "repair_count_cycle": repair_count_cycle,
                "repair_count_cumulative": repair_count_total,
                "feature_delta_rel_l2_at_write": feature_delta_rel,
                "feature_delta_max_abs_at_write": feature_delta_max,
                "frame_wall_sec": time.perf_counter() - frame_start,
                "run_wall_sec": "", "run_source": "Q0-2_new_inference",
            })
            prev_frame = frame
            if idx == 0 or (idx + 1) % 100 == 0 or idx == len(rows) - 1:
                print(f"[{seq}/{checkpoint_label}/{policy}] {idx+1}/{len(rows)} frames", flush=True)

    elapsed = time.perf_counter() - start_run
    for r in out_rows:
        r["run_wall_sec"] = elapsed
    run_meta = {
        "sequence": seq, "checkpoint": checkpoint_label, "policy": policy,
        "frames_processed": len(rows), "repair_calls": repair_count_total,
        "keyframe_calls": sum(1 for i in range(len(rows)) if i % REFRESH_INTERVAL == 0),
        "run_wall_sec": elapsed, "ms_per_rgb_frame": elapsed / max(len(rows), 1) * 1000,
        "commit_semantics": "F_only_post_readout_residual_gain",
        "gain": gain_for_policy(policy),
    }
    return out_rows, keyframe_predictions, run_meta


def fnum(x):
    if x is None or str(x).strip() == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def is_true(x) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes"}


def build_paired_effects(all_rows: list[dict]) -> list[dict]:
    baseline = {(r["sequence"], int(r["frame_index_zero_based"])): r
                for r in all_rows if r["policy"] == "A_ref"}
    paired = []
    for r in all_rows:
        if r["policy"] == "A_ref" or not is_true(r.get("gt_matched")):
            continue
        key = (r["sequence"], int(r["frame_index_zero_based"]))
        a = baseline.get(key)
        if a is None or not is_true(a.get("gt_matched")):
            continue
        av = fnum(a.get("AbsRel")); cv = fnum(r.get("AbsRel"))
        ar = fnum(a.get("RMSE")); cr = fnum(r.get("RMSE"))
        ad = fnum(a.get("delta1")); cd = fnum(r.get("delta1"))
        paired.append({
            "sequence": r["sequence"], "checkpoint": r["checkpoint"], "policy": r["policy"],
            "frame_index_zero_based": int(r["frame_index_zero_based"]),
            "refresh_cycle": int(r["refresh_cycle"]), "frame_offset_in_cycle": int(r["frame_offset_in_cycle"]),
            "is_keyframe": r["is_keyframe"], "A_AbsRel": av, "policy_AbsRel": cv,
            "delta_AbsRel_policy_minus_A": cv - av,
            "relative_AbsRel_reduction_pct": (av - cv) / max(av, 1e-12) * 100.0,
            "delta_RMSE_policy_minus_A": cr - ar,
            "delta_delta1_policy_minus_A": cd - ad,
        })
    return paired


def build_sequence_summary(all_rows: list[dict], run_meta: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in all_rows:
        groups[(r["sequence"], r["checkpoint"], r["policy"])].append(r)
    runtime = {(r["sequence"], r["checkpoint"], r["policy"]): r for r in run_meta}
    out = []
    for key, rr in sorted(groups.items()):
        seq, ckpt, policy = key
        scored = [r for r in rr if is_true(r.get("gt_matched")) and fnum(r.get("AbsRel")) is not None]
        prop = [r for r in scored if not is_true(r.get("is_keyframe"))]
        keyframes = [r for r in scored if is_true(r.get("is_keyframe"))]
        baseline = {(r["sequence"], int(r["frame_index_zero_based"])): r for r in all_rows if r["policy"] == "A_ref"}
        def mean(rows, metric):
            vals = [fnum(r.get(metric)) for r in rows]
            vals = [x for x in vals if x is not None]
            return float(np.mean(vals)) if vals else ""
        deltas = []
        worse = []
        for r in scored:
            a = baseline.get((seq, int(r["frame_index_zero_based"])))
            if a and is_true(a.get("gt_matched")):
                d = fnum(r.get("AbsRel")) - fnum(a.get("AbsRel"))
                deltas.append(d); worse.append(d > 0)
        rt = runtime.get(key, {})
        out.append({
            "sequence": seq, "checkpoint": ckpt, "policy": policy,
            "commit_semantics": rr[0].get("commit_semantics", ""),
            "rgb_frames": len(rr), "scored_frames": len(scored),
            "scored_propagated_frames": len(prop), "scored_keyframes": len(keyframes),
            "AbsRel_all": mean(scored, "AbsRel"), "RMSE_all": mean(scored, "RMSE"),
            "delta1_all": mean(scored, "delta1"),
            "AbsRel_propagated_only": mean(prop, "AbsRel"),
            "RMSE_propagated_only": mean(prop, "RMSE"),
            "delta1_propagated_only": mean(prop, "delta1"),
            "AbsRel_keyframes_only": mean(keyframes, "AbsRel"),
            "mean_frame_delta_AbsRel_vs_A": float(np.mean(deltas)) if deltas else "",
            "fraction_scored_frames_worse_than_A": float(np.mean(worse)) if worse else "",
            "repair_calls": rt.get("repair_calls", sum(1 for r in rr if is_true(r.get("repair_applied_this_frame")))),
            "keyframe_calls": rt.get("keyframe_calls", sum(1 for r in rr if is_true(r.get("is_keyframe")))),
            "run_wall_sec": rt.get("run_wall_sec", ""),
            "model_load_sec_shared_for_checkpoint_sequence": rt.get("model_load_sec_shared_for_checkpoint_sequence", ""),
            "ms_per_rgb_frame": rt.get("ms_per_rgb_frame", ""),
            "runtime_source": "new_inference" if rt else "historical_C3_reference_no_new_timing",
        })
    return out


def build_cycle_summary(all_rows: list[dict]) -> list[dict]:
    baseline = {(r["sequence"], int(r["frame_index_zero_based"])): r
                for r in all_rows if r["policy"] == "A_ref"}
    groups = defaultdict(list)
    for r in all_rows:
        groups[(r["sequence"], r["checkpoint"], r["policy"], int(r["refresh_cycle"]))].append(r)
    out = []
    for (seq, ckpt, policy, cycle), rr in sorted(groups.items()):
        scored = [r for r in rr if is_true(r.get("gt_matched")) and fnum(r.get("AbsRel")) is not None]
        prop = [r for r in scored if not is_true(r.get("is_keyframe"))]
        keyframes = [r for r in scored if is_true(r.get("is_keyframe"))]
        a_rows = [baseline.get((seq, int(r["frame_index_zero_based"]))) for r in scored]
        a_rows = [a for a in a_rows if a and is_true(a.get("gt_matched"))]
        def mean(rows, metric):
            vals = [fnum(r.get(metric)) for r in rows]
            vals = [x for x in vals if x is not None]
            return float(np.mean(vals)) if vals else ""
        a_mean = mean(a_rows, "AbsRel")
        cur_mean = mean(scored, "AbsRel")
        frame_diffs = []
        for r in scored:
            a = baseline.get((seq, int(r["frame_index_zero_based"])))
            if a and is_true(a.get("gt_matched")):
                frame_diffs.append(fnum(r["AbsRel"]) - fnum(a["AbsRel"]))
        out.append({
            "sequence": seq, "checkpoint": ckpt, "policy": policy, "refresh_cycle": cycle,
            "rgb_frames_in_cycle": len(rr), "cycle_complete_32_frames": str(len(rr) == REFRESH_INTERVAL).lower(),
            "scored_frames": len(scored), "scored_propagated_frames": len(prop),
            "scored_keyframes": len(keyframes), "AbsRel_cycle_all": cur_mean,
            "AbsRel_cycle_propagated": mean(prop, "AbsRel"),
            "AbsRel_cycle_keyframes": mean(keyframes, "AbsRel"),
            "A_AbsRel_cycle_all": a_mean,
            "delta_cycle_AbsRel_policy_minus_A": (cur_mean - a_mean) if cur_mean != "" and a_mean != "" else "",
            "fraction_scored_frames_worse_than_A": float(np.mean([d > 0 for d in frame_diffs])) if frame_diffs else "",
            "n_scored_frames_worse_than_A": sum(d > 0 for d in frame_diffs),
            "n_scored_frames_compared_to_A": len(frame_diffs),
        })
    return out


def build_invariants(all_rows: list[dict], run_meta: list[dict], keyframe_arrays: dict,
                     c3_refs: dict, sequences: list[str], selected: tuple[str, ...],
                     metric_tolerance: float, pixel_tolerance: float) -> list[dict]:
    out = []
    for seq in sequences:
        ref_a = {int(r["frame_index_zero_based"]): r for r in all_rows
                 if r["sequence"] == seq and r["policy"] == "A_ref"}
        for ckpt in selected:
            for policy in POLICIES:
                run_rows = [r for r in all_rows if r["sequence"] == seq and r["checkpoint"] == ckpt and r["policy"] == policy]
                if not run_rows:
                    continue
                run_key = (seq, ckpt, policy)
                arrs = keyframe_arrays[run_key]
                diffs = []
                for idx, arr in arrs.items():
                    if idx in keyframe_arrays.get((seq, "_REFERENCE_", "_REFERENCE_"), {}):
                        ref_arr = keyframe_arrays[(seq, "_REFERENCE_", "_REFERENCE_")][idx]
                        diffs.append(float(np.max(np.abs(arr.astype(np.float32) - ref_arr.astype(np.float32)))))
                max_pixel_diff = max(diffs, default=0.0)
                out.append({"sequence": seq, "checkpoint": ckpt, "policy": policy,
                            "invariant": "keyframe_predictions_identical_across_custom_arms",
                            "observed_deviation": max_pixel_diff, "threshold": pixel_tolerance,
                            "unit": "m", "passed": max_pixel_diff <= pixel_tolerance})

                metric_diffs = []
                for r in run_rows:
                    if not is_true(r.get("is_keyframe")) or not is_true(r.get("gt_matched")):
                        continue
                    a = ref_a.get(int(r["frame_index_zero_based"]))
                    if not a or not is_true(a.get("gt_matched")):
                        continue
                    for metric in METRICS:
                        metric_diffs.append(abs(fnum(r[metric]) - fnum(a[metric])))
                max_metric_diff = max(metric_diffs, default=0.0)
                out.append({"sequence": seq, "checkpoint": ckpt, "policy": policy,
                            "invariant": "keyframe_metrics_match_historical_A",
                            "observed_deviation": max_metric_diff, "threshold": metric_tolerance,
                            "unit": "metric_value", "passed": max_metric_diff <= metric_tolerance})

                schedule_errors = []
                for r in run_rows:
                    age = int(r["frame_offset_in_cycle"])
                    expected = should_repair(policy, age)
                    actual = is_true(r["repair_applied_this_frame"])
                    if expected != actual:
                        schedule_errors.append(int(r["frame_index_zero_based"]))
                out.append({"sequence": seq, "checkpoint": ckpt, "policy": policy,
                            "invariant": "repair_schedule_exact",
                            "observed_deviation": len(schedule_errors), "threshold": 0,
                            "unit": "mismatched_frames", "passed": len(schedule_errors) == 0,
                            "detail": ",".join(map(str, schedule_errors[:20]))})

    # The D head under F-only post-readout R1 must reproduce the historical native-D run.
    if "D" in selected:
        for seq in sequences:
            custom = {(int(r["frame_index_zero_based"])): r for r in all_rows
                      if r["sequence"] == seq and r["checkpoint"] == "D" and r["policy"] == "R1_G1.0_F_only"}
            native = {(int(r["frame_index_zero_based"])): r for r in all_rows
                      if r["sequence"] == seq and r["checkpoint"] == "D" and r["policy"] == "C3_native_D"}
            diffs = []
            for idx, r in custom.items():
                n = native.get(idx)
                if n and is_true(r.get("gt_matched")) and is_true(n.get("gt_matched")):
                    for metric in METRICS:
                        diffs.append(abs(fnum(r[metric]) - fnum(n[metric])))
            max_diff = max(diffs, default=0.0)
            out.append({"sequence": seq, "checkpoint": "D",  "policy": "R1_G1.0_F_only",
                        "invariant": "D_R1_G1.0_F_only_reproduces_historical_C3_native_D",
                        "observed_deviation": max_diff, "threshold": metric_tolerance,
                        "unit": "metric_value", "passed": max_diff <= metric_tolerance})
    return out


def main() -> int:
    args = parse_args()
    sequences = parse_sequences(args.sequences)
    selected = ("C", "D") if args.checkpoint == "both" else (args.checkpoint,)
    out_dir = Path(args.out_dir).resolve()
    if not args.preflight_only and (out_dir / "q0_2_summary.json").exists():
        raise RuntimeError(f"Refusing to overwrite existing Q0-2 results: {out_dir / 'q0_2_summary.json'}")
    pre, manifests, refs = build_preflight(args, sequences)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "q0_2_preflight.json", pre)
    print(f"[PASS] Q0-2 preflight: sequences={sequences}, checkpoints={list(pre['checkpoints'])}", flush=True)
    if args.preflight_only:
        print("Q0-2 preflight complete; no inference or training performed.", flush=True)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("Q0-2 inference requires CUDA. No training is performed.")

    snapshot = Path(args.snapshot).resolve()
    checkpoint_paths = {"C": Path(args.c_checkpoint).resolve(), "D": Path(args.d_checkpoint).resolve()}
    torch.set_num_threads(4)
    torch.manual_seed(20261025)
    np.random.seed(20261025)
    torch.cuda.manual_seed_all(20261025)

    all_rows: list[dict] = []
    run_meta: list[dict] = []
    keyframe_arrays: dict[tuple, dict[int, np.ndarray]] = {}
    invariant_rows: list[dict] = []
    start_all = time.perf_counter()

    for seq in sequences:
        seq_rows = manifests[seq][:args.max_frames] if args.max_frames > 0 else manifests[seq]
        c3_ref_rows = {mode: refs[seq][mode] for mode in ("A", "C", "D")}
        all_rows.extend(normalize_reference_rows(seq, c3_ref_rows, manifests, args.max_frames))
        # Load one model per sequence/checkpoint; all policies reset their own streaming state from frame 0.
        shape = None
        keyframe_reference = None
        for ckpt_label in selected:
            model_load_start = time.perf_counter()
            model, head = load_crossed_head_model(snapshot, checkpoint_paths[ckpt_label], ckpt_label, torch.device("cuda"))
            model_load_sec = time.perf_counter() - model_load_start
            if shape is None:
                shape = exp1.shape_info_for(model, Path(seq_rows[0]["rgb_path"]), torch.device("cuda"))
            print(f"[{seq}/{ckpt_label}] model loaded in {model_load_sec:.1f}s; policies={list(POLICIES)}", flush=True)
            for policy in POLICIES:
                rows_out, keyframes, run = run_policy(
                    seq, policy, ckpt_label, model, head, seq_rows, shape, torch.device("cuda")
                )
                run["model_load_sec_shared_for_checkpoint_sequence"] = model_load_sec
                run_meta.append(run)
                all_rows.extend(rows_out)
                # Persist each completed arm so an interrupted long run does not lose prior arms.
                progress_dir = out_dir / "q0_2_progress"
                progress_dir.mkdir(parents=True, exist_ok=True)
                write_csv(progress_dir / f"{seq}_{ckpt_label}_{policy}.csv", rows_out)
                write_json(out_dir / "q0_2_progress.json", {
                    "status": "RUNNING", "completed_runs": run_meta,
                    "next_expected_run": {"sequence": seq, "checkpoint": ckpt_label,
                                           "policy_index": list(POLICIES).index(policy) + 1},
                    "training_performed": False,
                })
                key = (seq, ckpt_label, policy)
                keyframe_arrays[key] = keyframes
                if keyframe_reference is None:
                    keyframe_reference = {i: x.copy() for i, x in keyframes.items()}
                    keyframe_arrays[(seq, "_REFERENCE_", "_REFERENCE_")] = keyframe_reference
                print(f"[{seq}/{ckpt_label}/{policy}] done: repairs={run['repair_calls']}, "
                      f"wall={run['run_wall_sec']:.1f}s", flush=True)
            del head, model
            torch.cuda.empty_cache()

    # Build invariants and summaries. In smoke mode, the native-D reproduction check is prefix-only.
    invariant_rows = build_invariants(
        all_rows, run_meta, keyframe_arrays, refs, sequences, selected,
        args.metric_tolerance, args.keyframe_pixel_tolerance_m,
    )
    paired_rows = build_paired_effects(all_rows)
    seq_summary = build_sequence_summary(all_rows, run_meta)
    cycle_summary = build_cycle_summary(all_rows)
    all_pass = all(bool(r["passed"]) for r in invariant_rows)
    reset_max = max((float(r["observed_deviation"]) for r in invariant_rows
                     if r["invariant"] == "keyframe_predictions_identical_across_custom_arms"), default=0.0)
    summary = {
        "status": "MECHANICS_PASS" if all_pass else "MECHANICS_FAIL_DO_NOT_INTERPRET",
        "experiment": "Q0-2 full-sequence fixed-gain sweep",
        "training_performed": False,
        "runtime_sec": time.perf_counter() - start_all,
        "is_smoke_or_partial": bool(args.max_frames > 0 or len(sequences) < len(SEQUENCE_PATHS) or len(selected) < 2),
        "max_frames_per_sequence": args.max_frames or "full_sequence",
        "sequences": sequences, "checkpoints": list(selected), "policies": list(POLICIES),
        "gain_values": POLICY_GAINS,
        "counts": {"frame_metric_rows_including_historical_refs": len(all_rows),
                   "new_inference_frame_rows": sum(1 for r in all_rows if r.get("run_source") == "Q0-2_new_inference"),
                   "paired_effect_rows": len(paired_rows), "cycle_summary_rows": len(cycle_summary),
                   "sequence_summary_rows": len(seq_summary), "invariant_rows": len(invariant_rows)},
        "mechanics": {"all_invariants_pass": all_pass,
                      "max_keyframe_pixel_difference_m": reset_max,
                      "keyframe_pixel_tolerance_m": args.keyframe_pixel_tolerance_m,
                      "metric_tolerance": args.metric_tolerance},
        "protocol_limits": [
             "C/D checkpoint weights are crossed under the same F-only post-readout semantics for fixed residual gains 0.25/0.5/1.0.",
            "Historical C3 A/C/D per-frame metrics are included as references; no new A inference is run.",
            "D/R1-G1.0 F-only is required to reproduce native C3 D; C/R1-G1.0 is a cross-semantics control and is not expected to reproduce native C.",
            "K=32 is fixed; no training, output-only adapter, or new gate is included in Q0-2; only fixed residual gain varies.",
            "Wall time is descriptive and includes frame loading, preprocessing, optical flow, inference and metrics; do not claim a latency win from one run ordering.",
            "Crowd2/Crowd3/Balloon2 are development sequences, not final unbiased test domains.",
        ],
    }

    write_csv(out_dir / "q0_2_frame_metrics.csv", all_rows)
    write_csv(out_dir / "q0_2_sequence_summary.csv", seq_summary)
    write_csv(out_dir / "q0_2_cycle_summary.csv", cycle_summary)
    write_csv(out_dir / "q0_2_paired_effects.csv", paired_rows)
    write_csv(out_dir / "q0_2_invariants.csv", invariant_rows)
    write_json(out_dir / "q0_2_summary.json", summary)

    readme = f"""# Q0-2 Full-Sequence Fixed-Gain Sweep

Status: **{summary['status']}**

- Training performed: no
- Sequences: {', '.join(sequences)}
- Checkpoints: {', '.join(selected)}
- K: {REFRESH_INTERVAL}
- New inference frame rows: {summary['counts']['new_inference_frame_rows']}
- Runtime: {summary['runtime_sec']:.1f} seconds
- Maximum cross-arm keyframe prediction difference: {reset_max:.8g} m

## Fixed-gain arms

- `A_ref`: historical native A rows from the audited C3 full-sequence frame CSV.
- `C3_native_C` / `C3_native_D`: historical native C/D rows from that same CSV.
- `R1_G0.25_F_only`: every propagation frame uses `F_next = F + 0.25 * (RepairHead(F) - F)`.
- `R1_G0.5_F_only`: every propagation frame uses `F_next = F + 0.50 * (RepairHead(F) - F)`.
- `R1_G1.0_F_only`: every propagation frame uses `F_next = RepairHead(F)`, the full-gain R1 control.

The audited repair head returns `x + delta(x)`, so the gain is applied only to the learned residual. Current-frame radius/confidence are not recomputed. Both C and D checkpoints are evaluated under the same F-only post-readout semantics. D with gain 1.0 must reproduce historical native C3-D within tolerance; C with gain 1.0 is a cross-semantics control and is not expected to reproduce native C3-C.

## Outputs

`q0_2_frame_metrics.csv`, `q0_2_sequence_summary.csv`, `q0_2_cycle_summary.csv`, `q0_2_paired_effects.csv`, `q0_2_invariants.csv`, `q0_2_preflight.json`, `q0_2_summary.json`.

Q0-2 does not train a model. It evaluates quality and failure risk under fixed residual gains; it does not establish a latency or compute-saving claim. Crowd2/Crowd3/Balloon2 are development sequences, not unbiased held-out domains. Calibration baselines remain for Q0-3.
"""
    (out_dir / "Q0_2_README.md").write_text(readme, encoding="utf-8")

    review_zip = out_dir / "q0_2_review.zip"
    include = ["q0_2_preflight.json", "q0_2_frame_metrics.csv", "q0_2_sequence_summary.csv",
               "q0_2_cycle_summary.csv", "q0_2_paired_effects.csv", "q0_2_invariants.csv",
               "q0_2_summary.json", "Q0_2_README.md"]
    with zipfile.ZipFile(review_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in include:
            p = out_dir / rel
            if p.is_file():
                zf.write(p, rel)
        zf.write(Path(__file__).resolve(), "gt4a2_q0_2_full_cycle_frequency.py")
    print(f"[{summary['status']}] Q0-2 review package: {review_zip}", flush=True)
    return 0 if all_pass else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        raise
