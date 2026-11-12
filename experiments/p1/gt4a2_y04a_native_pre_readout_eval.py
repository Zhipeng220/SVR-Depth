#!/usr/bin/env python3
"""Y04-A: evaluate the existing P_multi head in its trained native semantics.

Only NEW inference: P_multi, pre_readout, full residual (gain=1), fixed K=32.
A, O_multi and old post-readout P_multi scores are read from Y04-Fix3 CSV.
No training, oracle fitting, gain search, output-only replay or timing claim.
Install this file in experiments/p1 and run from the VeloDepth repository root.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
import zipfile
from collections import defaultdict
from types import SimpleNamespace

VERSION = "Y04-A-1.0"
ROOT = Path(__file__).resolve().parents[2]
K = 32
TAIL_START = {"crowd2": 768, "crowd3": 256, "balloon2": 256}
NEW_ARM = "Y04A_P_multi_pre_readout_gain1"
REF_ARMS = ("A_native", "Y04_O_multi", "Y04_P_multi_gain0.5")
MODEL_SHA = "615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e"
P_SHA = "77a86cdf523f776608b6b591d6c6552002b90398e85d00d3a3fe29afe14a0a24"
Y04_SCRIPT_SHA = "2a24e5a67487d55177e3374f9c6be5e92795b508dfcdb299cefe46e37cc31d4f"
SOURCE_SHA = {
    "experiments/p1/gt4a2_exp1_train_eval.py": "c24441cb405465c1a16a86127b51aade8f9af69b5d368f43d830d0e759ee3d7f",
    "experiments/p1/gt2_fixed_clip_repair_train.py": "bf7cd2021b1d0bbf66d75502dba6afed87f4c8ff3ef58ff7b18f56ac28044192",
    "velodepth/models/decoder_velo.py": "35ca12629accd1a2a77595f505165fa17dffab07c763a8b639604285b94bdb08",
    "velodepth/models/persistent_feature_repair.py": "6dfc8b0c5565ea9139b8c266c5ef337b3309ce0fde2f365428fc44390b84117c",
}
METRICS = ("AbsRel", "RMSE", "delta1")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def json_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def csv_read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csv_write(path, rows):
    if not rows:
        raise RuntimeError(f"Refusing empty result table: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def truth(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def percentile(values, q):
    values = sorted(float(x) for x in values)
    require(bool(values), "Cannot take percentile of empty data")
    p = (len(values) - 1) * q / 100.0
    lo, hi = math.floor(p), math.ceil(p)
    return values[lo] + (values[hi] - values[lo]) * (p - lo)


def plan_indices(seq, n, smoke):
    start = TAIL_START[seq] if smoke else 0
    stop = min(n, start + 64) if smoke else n
    require(start % K == 0 and start < stop, f"Invalid sequence range: {seq}/{n}")
    return list(range(start, stop))


def age_bin(age):
    if age < 2:
        return str(age)
    return "2-7" if age < 8 else "8-15" if age < 16 else "16-23" if age < 24 else "24-31"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", default=os.environ.get("VDEEP_SNAPSHOT", ""))
    p.add_argument("--y04-dir", default="outputs/gt4a2_y04_validation_fix3_20261025_225902")
    p.add_argument("--p-checkpoint", default="", help="Default: P_multi final checkpoint inside --y04-dir")
    p.add_argument("--sequences", default="crowd2,crowd3,balloon2")
    for seq in TAIL_START:
        p.add_argument(f"--{seq}-pair-csv", default=f"outputs/gt4a2_exp1/gt4a2_exp1_{seq}_pairs_used.csv")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--smoke-only", action="store_true")
    mode.add_argument("--self-test", action="store_true", help="Stdlib-only checks; no CUDA, checkpoint or repo required")
    p.add_argument("--out-dir", default="outputs/gt4a2_y04a_validation")
    a = p.parse_args(argv)
    a.seqs = [s.strip().lower() for s in a.sequences.split(",") if s.strip()]
    if not a.seqs or len(set(a.seqs)) != len(a.seqs) or set(a.seqs) - set(TAIL_START):
        p.error("--sequences must contain unique names from crowd2,crowd3,balloon2")
    if not a.self_test and not a.snapshot:
        p.error("Set VDEEP_SNAPSHOT or pass --snapshot")
    return a


def index_references(rows):
    result = {}
    for row in rows:
        if row.get("arm") not in REF_ARMS:
            continue
        key = (row["sequence"], row["arm"], int(row["frame_index_zero_based"]))
        require(key not in result, f"Duplicate Y04 reference key: {key}")
        result[key] = row
    return result


def validate_pair_identity(seq, rows, refs):
    for idx, row in enumerate(rows):
        for arm in REF_ARMS:
            key = (seq, arm, idx)
            require(key in refs, f"Missing Y04 reference: {key}")
            ref = refs[key]
            matched = str(row.get("gt_match_status", "")).strip().upper() == "MATCHED"
            require(truth(ref["gt_matched"]) == matched, f"GT matching differs: {key}")
            require(ref["rgb_file"] == Path(row["rgb_path"]).name, f"RGB identity differs: {key}")
            expected_gt = Path(row["depth_path"]).name if matched else ""
            require(ref["gt_file"] == expected_gt, f"GT identity differs: {key}")
            require(int(ref["frame_offset_in_cycle"]) == idx % K, f"Wrong reference age: {key}")
            require(int(ref["refresh_cycle"]) == idx // K, f"Wrong reference cycle: {key}")
            require(truth(ref["is_keyframe"]) == (idx % K == 0), f"Wrong reference keyframe: {key}")
            if matched:
                require(all(math.isfinite(float(ref[m])) for m in METRICS), f"Invalid reference metric: {key}")
                a = refs[(seq, "A_native", idx)]
                require(int(ref["valid_pixels"]) == int(a["valid_pixels"]), f"Reference valid-pixel mismatch: {key}")
    for arm in REF_ARMS:
        count = sum(1 for s, a, _ in refs if s == seq and a == arm)
        require(count == len(rows), f"Reference/manifest length differs for {seq}/{arm}")


def validate_checkpoint_metadata(ckpt, identity):
    expected_slots = [[s, c] for s in TAIL_START for c in range(8)]
    conditions = {
        "Y04 version": ckpt.get("y04_version") == "Y04-Fix3",
        "multi-scene label": ckpt.get("y04_label") == "Y04_P_multi",
        "persistent kind": ckpt.get("y04_kind") == "persistent",
        "training mode": ckpt.get("mode") == "pre_readout" and ckpt.get("mode_label") == "C",
        "step count": int(ckpt.get("step_count", -1)) == 1000 and int(ckpt.get("optimizer_steps", -1)) == 1000,
        "base weights": ckpt.get("pretrained_weights_sha256") == MODEL_SHA,
        "Y04 base weights": ckpt.get("y04_pretrained_weights_sha256") == MODEL_SHA,
        "Y04 source": ckpt.get("y04_script_sha256") == identity["script_sha256"],
        "training slots": ckpt.get("training_cycle_slots") == expected_slots,
        "training RGB ranges": ckpt.get("training_rgb_indices_by_sequence") == {s: [0, 255] for s in TAIL_START},
        "initialization": ckpt.get("y04_initial_head_sha256") == identity["common_initial_head_sha256"],
        "head state": bool(ckpt.get("repair_head_state_dict")),
    }
    require(all(conditions.values()), f"Checkpoint provenance mismatch: {[k for k,v in conditions.items() if not v]}")


def load_runtime():
    # Import only the two low-level helpers. Never call any train/main entry point.
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "experiments/p1"))
    import cv2
    import numpy as np
    import torch
    import gt4a2_exp1_train_eval as exp1
    import gt2_fixed_clip_repair_train as gt2
    return SimpleNamespace(cv2=cv2, np=np, torch=torch, exp1=exp1, gt2=gt2)


def preflight(a):
    ydir, snapshot = Path(a.y04_dir).resolve(), Path(a.snapshot).resolve()
    require((snapshot / "config.json").is_file(), f"Missing snapshot config: {snapshot}")
    require((snapshot / "model.safetensors").is_file(), f"Missing base weights: {snapshot}")
    require(digest(snapshot / "model.safetensors") == MODEL_SHA, "Base model SHA differs from Y04-Fix3")
    identity = json_read(ydir / "y04_run_identity.json")
    ysummary = json_read(ydir / "y04_summary.json")
    require(identity.get("version") == "Y04-Fix3" and identity.get("script_sha256") == Y04_SCRIPT_SHA,
            "This script requires the reviewed Y04-Fix3 run identity")
    require(identity.get("pretrained_weights_sha256") == MODEL_SHA, "Y04 run uses different base weights")
    require(identity.get("temporal_tail_eval_start") == TAIL_START, "Y04 tail split differs")
    require(ysummary.get("status") == "MECHANICS_PASS" and ysummary.get("version") == "Y04-Fix3",
            "Y04 evaluation is not complete")
    source_hashes = {}
    for rel, expected in SOURCE_SHA.items():
        path = ROOT / rel
        require(path.is_file(), f"Required repository file missing: {path}")
        actual = digest(path)
        require(actual == expected, f"Source differs from reviewed Fix3: {rel}; do not overwrite old code to bypass this check")
        source_hashes[rel] = actual
    for rel in ("velodepth/models/velodepth.py", "velodepth/models/encoder_velo.py"):
        source_hashes[rel] = digest(ROOT / rel)
    cp = Path(a.p_checkpoint).resolve() if a.p_checkpoint else ydir / "training/P_multi/gt4a2_c3_C_final_step1000.pt"
    require(cp.is_file(), f"Missing P_multi final checkpoint: {cp}")
    require(digest(cp) == P_SHA, "P_multi checkpoint SHA differs from reviewed Fix3")
    training_summary = json_read(ydir / "training/P_multi/gt4a2_c3_C_training_summary.json")
    require(training_summary.get("checkpoint_sha256") == P_SHA, "P_multi summary/checkpoint SHA mismatch")
    refs_path = ydir / "y04_all_frame_metrics.csv"
    refs = index_references(csv_read(refs_path))
    rt = load_runtime()
    ckpt = rt.torch.load(cp, map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(ckpt, identity)
    head_state = ckpt["repair_head_state_dict"]
    require(all(rt.torch.isfinite(t).all().item() for t in head_state.values()), "Nonfinite P_multi weights")
    require(rt.exp1.state_sha256(head_state) == training_summary["final_head_state_sha256"], "P_multi head tensor SHA mismatch")
    require(ckpt.get("sample_order_sha256") == training_summary.get("sample_order_sha256"), "P_multi sample order mismatch")
    manifests, manifest_info = {}, {}
    for seq in a.seqs:
        path = Path(getattr(a, seq + "_pair_csv")).resolve()
        require(path.is_file(), f"Missing pair manifest: {path}")
        actual = digest(path)
        require(actual == identity["pair_csv_sha256"][seq], f"Pair CSV SHA changed for {seq}; keep the Y04 pairing and paths")
        rows = rt.exp1.read_manifest(path, training=False)
        validate_pair_identity(seq, rows, refs)
        indices = plan_indices(seq, len(rows), a.smoke_only)
        manifests[seq] = rows
        manifest_info[seq] = {"path": str(path), "sha256": actual, "rgb_frames": len(rows),
            "process_start": indices[0], "process_stop_exclusive": indices[-1] + 1,
            "score_start": TAIL_START[seq],
            "expected_scored_frames": sum(str(rows[i].get("gt_match_status", "")).upper() == "MATCHED"
                                          for i in indices if i >= TAIL_START[seq])}
    protocol = {"status": "PREFLIGHT_PASS", "version": VERSION, "training_performed": False,
        "new_inference_arms": [NEW_ARM], "reference_arms": list(REF_ARMS),
        "K": K, "mode": "pre_readout", "gain": 1.0, "sequences": a.seqs,
        "partial_smoke": a.smoke_only, "snapshot": str(snapshot), "base_weights_sha256": MODEL_SHA,
        "snapshot_config_sha256": digest(snapshot / "config.json"), "checkpoint": str(cp),
        "checkpoint_sha256": P_SHA, "head_sha256": training_summary["final_head_state_sha256"],
        "source_sha256": source_hashes, "script_sha256": digest(__file__),
        "y04_dir": str(ydir), "y04_identity_sha256": digest(ydir / "y04_run_identity.json"),
        "reference_csv_sha256": digest(refs_path), "manifests": manifest_info,
        "geometry": "native network grid; padded canvas only for output restoration",
        "smoke": "64 RGB frames starting at the first tail keyframe; global indices retained",
        "formal": "full continuous sequence from RGB index 0, score original Y04 tails only",
        "timing": "diagnostic wall time includes GT/metrics; not Y06-B or a speed benchmark"}
    return protocol, manifests, refs, rt, head_state


def check_head_calls(before, after, is_keyframe):
    expected = 0 if is_keyframe else 1
    require(after - before == expected,
            f"Repair head called {after-before} times; expected {expected} (keyframe={is_keyframe})")


def native_step(rt, model, prev, frame, shape, dis, state):
    """Native repair is inside forward_fast. Never call head or rescale again here."""
    static = rt.exp1.transition_static(model, prev, frame, shape, dis)
    radius, confidence, features, rays = state
    radius, confidence, features = rt.exp1.one_fast_step(model, prev, frame, static, radius, confidence, features)
    return radius, confidence, features, rays


def run_sequence(a, seq, manifest, refs, rt, head_state, out):
    torch, np, exp1, gt2 = rt.torch, rt.np, rt.exp1, rt.gt2
    device = torch.device("cuda")
    model = gt2.load_model(Path(a.snapshot).resolve(), "pre_readout", device)
    head = model.pixel_decoder.persistent_feature_repair
    require(head is not None and model.pixel_decoder.feature_repair_mode == "pre_readout", "Native pre_readout mode missing")
    head.load_state_dict(head_state, strict=True)
    require(sum(p.numel() for p in head.parameters()) == 170496, "Unexpected repair-head architecture")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    head.eval()
    count = {"value": 0}
    def count_forward(_module, _inputs, _outputs):
        count["value"] += 1
    hook = head.register_forward_hook(count_forward)
    indices = plan_indices(seq, len(manifest), a.smoke_only)
    shape = exp1.shape_info_for(model, Path(manifest[indices[0]]["rgb_path"]), device)
    dis = rt.cv2.DISOpticalFlow_create(rt.cv2.DISOPTICAL_FLOW_PRESET_FAST)
    prev, state = None, None
    results, invariants = [], []
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            for pos, idx in enumerate(indices):
                rec = manifest[idx]
                matched = str(rec.get("gt_match_status", "")).strip().upper() == "MATCHED"
                frame = exp1.load_frame(rec, shape, device, need_gt=matched)
                key = idx % K == 0
                before = count["value"]
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    if key:
                        rays, radius, confidence, features = exp1.keyframe_state(model, frame, shape)
                        state = radius, confidence, features, rays
                        network = tuple(shape["network_shape"])
                        require(tuple(model.pixel_decoder.radial_module.original_shapes) == network,
                                "Decoder readout grid is not network_shape")
                        require(tuple(radius.shape[-2:]) == network and tuple(rays.shape[-2:]) == network,
                                "Radius/rays do not share the native network grid")
                    else:
                        require(state is not None and prev is not None, "Missing persistent state")
                        state = native_step(rt, model, prev, frame, shape, dis, state)
                    radius, confidence, features, rays = state
                    pred_t = gt2.make_z_depth(radius, rays, shape["padded_shape"], shape["paddings"], model.interpolation_mode)
                check_head_calls(before, count["value"], key)
                pred = pred_t.detach().float().cpu().numpy()
                require(tuple(pred.shape) == tuple(shape["original_shape"]), f"Output geometry differs at {seq}/{idx}")
                metrics = {m: "" for m in (*METRICS, "valid_ratio", "gt_valid_ratio")}
                metrics.update(valid_pixels=0, total_pixels=int(np.prod(shape["original_shape"])))
                metric_dev = ""
                if matched:
                    gt = frame["gt_np"]
                    valid = (gt > 0) & np.isfinite(gt)
                    require(np.all(np.isfinite(pred[valid]) & (pred[valid] > 0)),
                            f"Invalid predictions at GT-valid pixels: {seq}/{idx}; do not shrink evaluation mask")
                    metrics = gt2.calculate_metrics(gt, pred)
                    require(all(math.isfinite(float(metrics[m])) for m in METRICS), f"Nonfinite metrics: {seq}/{idx}")
                    a_ref = refs[(seq, "A_native", idx)]
                    require(int(metrics["valid_pixels"]) == int(a_ref["valid_pixels"]), f"GT-valid pixel count changed: {seq}/{idx}")
                    require(int(metrics["total_pixels"]) == int(a_ref["total_pixels"]), f"GT canvas changed: {seq}/{idx}")
                    if key:
                        metric_dev = max(abs(float(metrics[m]) - float(a_ref[m])) for m in METRICS)
                        require(metric_dev <= 2e-5, f"Keyframe metrics differ from Y04 A: {seq}/{idx}, deviation={metric_dev}")
                result = {"sequence": seq, "arm": NEW_ARM, "frame_index_zero_based": idx,
                    "refresh_cycle": idx // K, "frame_offset_in_cycle": idx % K,
                    "is_keyframe": str(key).lower(), "gt_matched": str(matched).lower(),
                    "rgb_file": Path(rec["rgb_path"]).name,
                    "gt_file": Path(rec["depth_path"]).name if matched else "",
                    "y04_eval_start_index": TAIL_START[seq],
                    "y04_split": "temporal_tail_eval" if idx >= TAIL_START[seq] else "train_prefix_not_scored",
                    "mode": "pre_readout", "gain": 1.0, "repair_head_calls_this_frame": count["value"] - before,
                    "additional_readout_passes": 0, **metrics}
                results.append(result)
                invariants.append({"sequence": seq, "frame_index_zero_based": idx, "is_keyframe": key,
                    "expected_head_calls": 0 if key else 1, "observed_head_calls": count["value"] - before,
                    "keyframe_max_metric_deviation_vs_Y04_A": metric_dev,
                    "same_gt_valid_pixel_count": True if matched else "not_scored", "passed": True})
                prev = frame
                if pos == 0 or (pos + 1) % 100 == 0 or pos + 1 == len(indices):
                    print(f"[{seq}/{NEW_ARM}] {pos+1}/{len(indices)} RGB, global_index={idx}", flush=True)
        require(count["value"] == sum(i % K != 0 for i in indices), "Total native repair-call count differs")
        require(model.pixel_decoder.feature_repair_mode == "pre_readout", "Repair mode changed during evaluation")
        csv_write(out / "progress" / f"{seq}_frame_metrics.csv", results)
        return results, invariants, {"sequence": seq, "processed_rgb_frames": len(indices),
            "keyframe_calls": sum(i % K == 0 for i in indices), "repair_head_calls": count["value"],
            "additional_readout_passes": 0, "diagnostic_wall_sec": time.perf_counter() - t0,
            "timing_claim": False, "network_shape": list(shape["network_shape"]),
            "padded_shape": list(shape["padded_shape"]), "original_shape": list(shape["original_shape"])}
    finally:
        hook.remove()
        del model, head
        torch.cuda.empty_cache()


def comparison_tables(new_rows, refs):
    combined, paired = [], []
    for row in new_rows:
        idx, seq = int(row["frame_index_zero_based"]), row["sequence"]
        if idx < TAIL_START[seq]:
            continue
        combined.append(row)
        for arm in REF_ARMS:
            ref = dict(refs[(seq, arm, idx)])
            combined.append(ref)
            if truth(row["gt_matched"]):
                paired.append({"sequence": seq, "frame_index_zero_based": idx,
                    "refresh_cycle": idx // K, "frame_offset_in_cycle": idx % K,
                    "candidate": NEW_ARM, "reference": arm,
                    **{f"delta_{m}_new_minus_reference": float(row[m]) - float(ref[m]) for m in METRICS}})
    return combined, paired


def summarize(combined):
    by = defaultdict(list)
    lookup = {}
    for row in combined:
        key = row["sequence"], row["arm"], int(row["frame_index_zero_based"])
        require(key not in lookup, f"Duplicate comparison row: {key}")
        lookup[key] = row
        by[key[:2]].append(row)
    sequences, cycles, ages = [], [], []
    for (seq, arm), rows in sorted(by.items()):
        scored = [r for r in rows if truth(r["gt_matched"])]
        require(bool(scored), f"No scored tail frames: {seq}/{arm}")
        cyc = defaultdict(list)
        for r in rows:
            cyc[int(r["refresh_cycle"])].append(r)
        complete_harms = []
        for c, rr in sorted(cyc.items()):
            ss = [r for r in rr if truth(r["gt_matched"])]
            if not ss:
                continue
            ds = [float(r["AbsRel"]) - float(lookup[(seq, "A_native", int(r["frame_index_zero_based"]))]["AbsRel"]) for r in ss]
            complete = len(rr) == K and {int(r["frame_offset_in_cycle"]) for r in rr} == set(range(K))
            if complete:
                complete_harms.append(statistics.mean(ds) > 1e-8)
            cycles.append({"sequence": seq, "arm": arm, "refresh_cycle": c, "complete_rgb_cycle": complete,
                "rgb_frames": len(rr), "scored_frames": len(ss),
                "mean_AbsRel": statistics.mean(float(r["AbsRel"]) for r in ss),
                "delta_AbsRel_vs_A": statistics.mean(ds), "harmed_vs_A": statistics.mean(ds) > 1e-8})
        diffs = [float(r["AbsRel"]) - float(lookup[(seq, "A_native", int(r["frame_index_zero_based"]))]["AbsRel"]) for r in scored]
        means = {f"mean_{m}": statistics.mean(float(r[m]) for r in scored) for m in METRICS}
        prop = [r for r in scored if not truth(r["is_keyframe"])]
        entry = {"sequence": seq, "arm": arm, "eval_start_index": TAIL_START[seq], "n_frames": len(scored), **means,
            "frame_AbsRel_p95": percentile([r["AbsRel"] for r in scored], 95),
            "propagated_frame_AbsRel_p95": percentile([r["AbsRel"] for r in prop], 95) if prop else "",
            "harmed_frames_vs_A": sum(d > 1e-8 for d in diffs),
            "harmed_frame_fraction_vs_A": statistics.mean(d > 1e-8 for d in diffs),
            "complete_rgb_cycles": len(complete_harms), "harmed_complete_cycles_vs_A": sum(complete_harms),
            "harm_tolerance_absrel": 1e-8}
        for refarm, label in (("A_native", "A"), ("Y04_O_multi", "O_multi"), ("Y04_P_multi_gain0.5", "old_P_multi")):
            refmean = statistics.mean(float(lookup[(seq, refarm, int(r["frame_index_zero_based"]))]["AbsRel"]) for r in scored)
            entry[f"relative_AbsRel_change_pct_vs_{label}"] = 100 * (means["mean_AbsRel"] / refmean - 1)
        sequences.append(entry)
        age_groups = defaultdict(list)
        for r in scored:
            age_groups[age_bin(int(r["frame_offset_in_cycle"]))].append(r)
        for age, rr in age_groups.items():
            ages.append({"sequence": seq, "arm": arm, "age_bin": age, "n_frames": len(rr),
                **{f"mean_{m}": statistics.mean(float(r[m]) for r in rr) for m in METRICS}})
    macro = []
    for arm in ("A_native", "Y04_O_multi", "Y04_P_multi_gain0.5", NEW_ARM):
        rr = [r for r in sequences if r["arm"] == arm]
        macro.append({"arm": arm, "n_sequences": len(rr),
            **{f"sequence_macro_{m}": statistics.mean(r[f"mean_{m}"] for r in rr) for m in METRICS}})
    return sequences, cycles, ages, macro


@contextlib.contextmanager
def exclusive_output(out, ydir):
    import fcntl
    require(out != ydir and ydir not in out.parents, "--out-dir must be outside the original Y04 directory")
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".y04a.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f"Another Y04-A process is using {out}") from e
        require(not (out / "y04a_summary.json").exists(), f"Completed results exist: {out}; choose a new output directory")
        yield


def review_archive(out, ydir):
    dest = out / "y04a_review.zip"
    temporary = out / "y04a_review.zip.tmp"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=7) as z:
        for p in sorted(out.glob("y04a_*.csv")) + sorted(out.glob("y04a_*.json")):
            if p.name != "y04a_failure.json":
                z.write(p, p.name)
        z.write(__file__, Path(__file__).name)
        for rel in SOURCE_SHA:
            z.write(ROOT / rel, "source/" + rel)
        for name in ("y04_run_identity.json", "y04_summary.json", "y04_sequence_summary.csv"):
            z.write(ydir / name, "reference/" + name)
        z.write(ydir / "training/P_multi/gt4a2_c3_C_training_summary.json", "reference/P_multi_training_summary.json")
    temporary.replace(dest)
    return dest


def main(argv=None):
    a = parse_args(argv)
    if a.self_test:
        return self_test()
    out, ydir = Path(a.out_dir).resolve(), Path(a.y04_dir).resolve()
    with exclusive_output(out, ydir):
        try:
            pre, manifests, refs, rt, head_state = preflight(a)
            previous = out / "y04a_preflight.json"
            if previous.exists():
                require(json_read(previous) == pre, "Existing output identity differs; choose a new --out-dir")
            json_write(previous, pre)
            print(f"[PREFLIGHT_PASS] {VERSION}: P_multi/pre_readout/gain1 only; references reused from Y04", flush=True)
            if a.preflight_only:
                print("No inference or training performed.", flush=True)
                return 0
            torch = rt.torch
            require(torch.cuda.is_available(), "CUDA is required for frozen VeloDepth evaluation")
            torch.set_num_threads(4)
            torch.manual_seed(20261025)
            rt.np.random.seed(20261025)
            torch.cuda.manual_seed_all(20261025)
            all_new, all_checks, execution = [], [], []
            for seq in a.seqs:
                rows, checks, meta = run_sequence(a, seq, manifests[seq], refs, rt, head_state, out)
                all_new.extend(rows)
                all_checks.extend(checks)
                execution.append(meta)
            # Verify immutable inputs again before committing results.
            require(digest(pre["checkpoint"]) == P_SHA, "Checkpoint changed while evaluating")
            require(digest(ydir / "y04_all_frame_metrics.csv") == pre["reference_csv_sha256"], "Y04 references changed while evaluating")
            for info in pre["manifests"].values():
                require(digest(info["path"]) == info["sha256"], "Pair manifest changed while evaluating")
            for rel, sha in pre["source_sha256"].items():
                require(digest(ROOT / rel) == sha, f"Source changed while evaluating: {rel}")
            require(digest(__file__) == pre["script_sha256"], "Y04-A script changed while evaluating")
            combined, paired = comparison_tables(all_new, refs)
            seq_rows, cyc_rows, age_rows, macro = summarize(combined)
            tables = {"y04a_all_frame_metrics.csv": all_new, "y04a_temporal_tail_metrics.csv": combined,
                "y04a_paired_differences.csv": paired, "y04a_sequence_summary.csv": seq_rows,
                "y04a_cycle_summary.csv": cyc_rows, "y04a_age_summary.csv": age_rows,
                "y04a_macro_summary.csv": macro, "y04a_invariants.csv": all_checks}
            for name, rows in tables.items():
                csv_write(out / name, rows)
            json_write(out / "y04a_execution.json", execution)
            summary = {"status": "SMOKE_PASS" if a.smoke_only else "MECHANICS_PASS", "version": VERSION,
                "training_performed": False, "partial_smoke": a.smoke_only,
                "sequences": a.seqs, "full_three_sequence_evaluation": not a.smoke_only and set(a.seqs) == set(TAIL_START),
                "new_inference_arms": [NEW_ARM], "reference_arms": list(REF_ARMS), "K": K,
                "checkpoint_sha256": P_SHA, "mode": "pre_readout", "gain": 1.0,
                "new_rgb_records": len(all_new), "comparison_tail_records": len(combined),
                "invariants_passed": True, "preflight_sha256": digest(previous),
                "script_sha256": digest(__file__), "sequence_macro_metrics": macro,
                "method_verdict": "NOT_AUTOMATICALLY_DECIDED; assess raw accuracy, harm and future cost evidence",
                "limits": ["Development temporal tails, not independent scene/domain tests",
                    "Historical comparators are imported metrics; not newly replayed predictions",
                    "GT frame identities and pixel counts are checked; old per-pixel masks are not independently available",
                    "No per-frame GT affine oracle is used", "No speedup claim from diagnostic wall times",
                    "This changes order, gain and associated native readout/state together; not a pure order ablation"]}
            json_write(out / "y04a_summary.json", summary)
            dest = review_archive(out, ydir)
            (out / "y04a_failure.json").unlink(missing_ok=True)
            for r in seq_rows:
                if r["arm"] == NEW_ARM:
                    print(f"[RESULT] {r['sequence']}: AbsRel={r['mean_AbsRel']:.6f}; "
                          f"vs O_multi={r['relative_AbsRel_change_pct_vs_O_multi']:+.2f}%; "
                          f"harmed cycles={r['harmed_complete_cycles_vs_A']}/{r['complete_rgb_cycles']}", flush=True)
            print(f"[{summary['status']}] {dest}", flush=True)
            return 0
        except Exception as exc:
            (out / "y04a_summary.json").unlink(missing_ok=True)
            json_write(out / "y04a_failure.json", {"status": "FAILED", "error_type": type(exc).__name__,
                "error": str(exc), "training_performed": False, "traceback": traceback.format_exc()})
            raise


def self_test():
    import unittest
    from unittest.mock import Mock

    class Checks(unittest.TestCase):
        def test_smoke_uses_tail_keyframe_global_indices(self):
            self.assertEqual(plan_indices("crowd2", 895, True), list(range(768, 832)))
            self.assertEqual(plan_indices("balloon2", 469, False), list(range(469)))

        def test_no_extra_or_keyframe_repair(self):
            check_head_calls(4, 4, True)
            check_head_calls(4, 5, False)
            with self.assertRaises(RuntimeError):
                check_head_calls(4, 6, False)
            with self.assertRaises(RuntimeError):
                check_head_calls(4, 5, True)

        def test_native_state_outputs_are_carried_forward(self):
            transition = Mock(return_value="static")
            step = Mock(side_effect=[("r1", "c1", ["f1"]), ("r2", "c2", ["f2"])])
            rt = SimpleNamespace(exp1=SimpleNamespace(transition_static=transition, one_fast_step=step))
            s = native_step(rt, "model", "frame0", "frame1", "shape", "dis", ("r0", "c0", ["f0"], "rays"))
            self.assertEqual(s, ("r1", "c1", ["f1"], "rays"))
            native_step(rt, "model", "frame1", "frame2", "shape", "dis", s)
            self.assertEqual(step.call_args.args[-3:], ("r1", "c1", ["f1"]))

        def test_references_reject_duplicates(self):
            row = {"sequence": "crowd2", "arm": "A_native", "frame_index_zero_based": "0"}
            with self.assertRaises(RuntimeError):
                index_references([row, row])

        def test_tail_risk_excludes_prefix_and_counts_missing_gt_cycle(self):
            refs, new = {}, []
            for i in range(256, 290):
                for arm, value in [("A_native", 0.1), ("Y04_O_multi", 0.08), ("Y04_P_multi_gain0.5", 0.12)]:
                    refs[("balloon2", arm, i)] = {"sequence": "balloon2", "arm": arm,
                        "frame_index_zero_based": i, "refresh_cycle": i // K, "frame_offset_in_cycle": i % K,
                        "is_keyframe": i % K == 0, "gt_matched": i != 260,
                        "AbsRel": value, "RMSE": value * 2, "delta1": 0.9}
                new.append({**refs[("balloon2", "A_native", i)], "arm": NEW_ARM, "AbsRel": 0.11})
            combined, pairs = comparison_tables(new, refs)
            seq, cycles, ages, macro = summarize(combined)
            actual = next(r for r in seq if r["arm"] == NEW_ARM)
            self.assertEqual(actual["n_frames"], 33)
            self.assertEqual(actual["complete_rgb_cycles"], 1)
            self.assertEqual(actual["harmed_complete_cycles_vs_A"], 1)
            self.assertEqual(actual["harmed_frames_vs_A"], 33)
            self.assertEqual(len(pairs), 99)
            self.assertAlmostEqual(actual["relative_AbsRel_change_pct_vs_O_multi"], 37.5)

        def test_pair_identity_rejects_changed_gt(self):
            row = {"rgb_path": "/data/a.png", "depth_path": "/data/d.png", "gt_match_status": "MATCHED"}
            refs = {( "crowd2", arm, 0): {"rgb_file": "a.png", "gt_file": "wrong.png",
                "gt_matched": "true", "frame_offset_in_cycle": 0, "refresh_cycle": 0,
                "is_keyframe": "true", "AbsRel": 0.1, "RMSE": 0.2, "delta1": 0.9, "valid_pixels": 100}
                for arm in REF_ARMS}
            with self.assertRaises(RuntimeError):
                validate_pair_identity("crowd2", [row], refs)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
