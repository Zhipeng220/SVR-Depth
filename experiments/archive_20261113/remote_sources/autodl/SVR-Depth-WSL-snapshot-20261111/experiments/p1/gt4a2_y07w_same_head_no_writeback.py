#!/usr/bin/env python3
"""Y07-W: one frozen P_multi head used for current output, with native A state.

Only N is newly evaluated. A/O are imported from Y04; P from completed Y04-A.
No training, gain/K search, oracle, new GPU timing or additional P replay.
Install in experiments/p1. Requires the original, unchanged Y04-A helper.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
import importlib.util
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[2]
VERSION = "Y07-W-1.0"
HELPER_SHA = "6fbe1a2464a7f9756f9ea68cf05f3476a89b005b8e6f889dd1a0093f231c3116"
K = 32
TAIL = {"crowd2": 768, "crowd3": 256, "balloon2": 256}
A_ARM, O_ARM = "A_native", "Y04_O_multi"
P_ARM = "Y04A_P_multi_pre_readout_gain1"
N_ARM = "Y07W_P_multi_current_output_no_writeback"
REF_ARMS = (A_ARM, O_ARM, P_ARM)
ARMS = (*REF_ARMS, N_ARM)
METRICS = ("AbsRel", "RMSE", "delta1")
METRIC_TOL, HARM_TOL = 2e-5, 1e-8


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def import_helper():
    import hashlib
    path = ROOT / "experiments/p1/gt4a2_y04a_native_pre_readout_eval.py"
    require(path.is_file(), f"Install original Y04-A helper first: {path}")
    require(hashlib.sha256(path.read_bytes()).hexdigest() == HELPER_SHA,
        "Y04-A helper differs; keep the original helper unchanged")
    spec = importlib.util.spec_from_file_location("y07w_y04a_helper", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", default=os.environ.get("VDEEP_SNAPSHOT", ""))
    p.add_argument("--y04-dir", default="outputs/gt4a2_y04_validation_fix3_20261025_225902")
    p.add_argument("--y04a-dir", default="auto")
    p.add_argument("--sequences", default=",".join(TAIL))
    for seq in TAIL:
        p.add_argument(f"--{seq}-pair-csv", default=f"outputs/gt4a2_exp1/gt4a2_exp1_{seq}_pairs_used.csv")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--smoke-only", action="store_true")
    modes.add_argument("--self-test", action="store_true")
    p.add_argument("--out-dir", default="outputs/gt4a2_y07w_validation")
    a = p.parse_args(argv)
    a.seqs = [s.strip().lower() for s in a.sequences.split(",") if s.strip()]
    if not a.seqs or len(set(a.seqs)) != len(a.seqs) or set(a.seqs) - set(TAIL):
        p.error("Choose unique crowd2,crowd3,balloon2 sequences")
    if not a.self_test and not a.snapshot:
        p.error("Set VDEEP_SNAPSHOT or pass --snapshot")
    return a


def resolve_y04a(value, helper):
    candidates = list((ROOT / "outputs").glob("gt4a2_y04a_validation*")) if value == "auto" else [Path(value).resolve()]
    valid = []
    for path in candidates:
        try:
            s = helper.json_read(path / "y04a_summary.json")
            if (s.get("status") == "MECHANICS_PASS" and s.get("full_three_sequence_evaluation")
                and not s.get("partial_smoke") and s.get("script_sha256") == HELPER_SHA
                and s.get("checkpoint_sha256") == helper.P_SHA and s.get("mode") == "pre_readout"
                and s.get("gain") == 1.0 and s.get("invariants_passed")):
                valid.append(path.resolve())
        except (OSError, ValueError):
            continue
    require(bool(valid), "No matching full three-sequence Y04-A result; pass --y04a-dir explicitly")
    return max(valid, key=lambda p: (p / "y04a_summary.json").stat().st_mtime_ns)


def infer_head_spec(state):
    require("blocks.0.0.weight" in state, "Missing P input projection")
    first = tuple(state["blocks.0.0.weight"].shape)
    require(len(first) == 4 and first[2:] == (1, 1), "Invalid P projection shape")
    bottleneck, channels = first[:2]
    levels = sum(key.startswith("blocks.") and key.endswith(".0.weight") for key in state)
    require(min(channels, bottleneck, levels) > 0, "Invalid head dimensions")
    block = {"0.weight": (bottleneck, channels, 1, 1),
        "1.weight": (bottleneck,), "1.bias": (bottleneck,),
        "3.weight": (bottleneck, bottleneck, 3, 3),
        "4.weight": (bottleneck,), "4.bias": (bottleneck,),
        "6.weight": (channels, bottleneck, 1, 1), "6.bias": (channels,)}
    expected = {f"blocks.{level}.{key}": shape for level in range(levels) for key, shape in block.items()}
    require(set(state) == set(expected), "P state keys differ from the audited architecture")
    for key, shape in expected.items():
        require(tuple(state[key].shape) == shape, f"P weight shape differs: {key}")
    return {"channels": int(channels), "bottleneck": int(bottleneck), "levels": int(levels)}


def build_head(state):
    from velodepth.models.persistent_feature_repair import PersistentFeatureRepairHead
    spec = infer_head_spec(state)
    require(spec == {"channels": 512, "bottleneck": 32, "levels": 4}, f"Pinned P architecture differs: {spec}")
    head = PersistentFeatureRepairHead(**spec)
    head.load_state_dict(state, strict=True)
    require(sum(p.numel() for p in head.parameters()) == 170496, "P parameter count differs")
    for p in head.parameters():
        p.requires_grad_(False)
    head.eval()
    return head


def preflight(a, helper):
    qa = SimpleNamespace(snapshot=a.snapshot, y04_dir=a.y04_dir, p_checkpoint="", seqs=a.seqs,
        smoke_only=a.smoke_only, **{s + "_pair_csv": getattr(a, s + "_pair_csv") for s in TAIL})
    base, manifests, oldrefs, rt, head_state = helper.preflight(qa)
    ydir = resolve_y04a(a.y04a_dir, helper)
    prior = helper.json_read(ydir / "y04a_preflight.json")
    for key in ("source_sha256", "snapshot_config_sha256", "reference_csv_sha256", "checkpoint_sha256"):
        require(prior[key] == base[key], f"Inputs differ from completed Y04-A: {key}")
    refs = {key: row for key, row in oldrefs.items() if key[0] in a.seqs and key[1] in (A_ARM, O_ARM)}
    for row in helper.csv_read(ydir / "y04a_all_frame_metrics.csv"):
        if row["sequence"] not in a.seqs:
            continue
        key = row["sequence"], row["arm"], int(row["frame_index_zero_based"])
        require(row["arm"] == P_ARM and key not in refs, f"Unexpected/duplicate P reference: {key}")
        refs[key] = row
    for seq, manifest in manifests.items():
        require(sum(s == seq and arm == P_ARM for s, arm, _ in refs) == len(manifest), f"P reference length differs: {seq}")
        for idx in range(len(manifest)):
            ar, pr = refs[seq, A_ARM, idx], refs[seq, P_ARM, idx]
            for key in ("rgb_file", "gt_file"):
                require(ar[key] == pr[key], f"P/A frame identity differs: {seq}/{idx}/{key}")
            require(helper.truth(ar["gt_matched"]) == helper.truth(pr["gt_matched"]), "P/A GT matching differs")
            require(int(pr["frame_offset_in_cycle"]) == idx % K and int(pr["refresh_cycle"]) == idx // K,
                "P age/cycle differs")
            require(helper.truth(pr["is_keyframe"]) == (idx % K == 0), "P keyframe schedule differs")
            if helper.truth(ar["gt_matched"]):
                require(int(ar["valid_pixels"]) == int(pr["valid_pixels"]), "P/A valid pixel count differs")
                require(all(math.isfinite(float(pr[m])) for m in METRICS), "Nonfinite P reference")
    checked_head = build_head(head_state)  # Strict CPU load before any model inference.
    del checked_head
    files = {base["checkpoint"]: base["checkpoint_sha256"],
        str(Path(a.snapshot).resolve() / "model.safetensors"): helper.MODEL_SHA,
        str(Path(a.snapshot).resolve() / "config.json"): base["snapshot_config_sha256"],
        str(Path(a.y04_dir).resolve() / "y04_all_frame_metrics.csv"): base["reference_csv_sha256"],
        str(ROOT / "experiments/p1/gt4a2_y04a_native_pre_readout_eval.py"): HELPER_SHA}
    for name in ("y04a_summary.json", "y04a_preflight.json", "y04a_all_frame_metrics.csv", "y04a_sequence_summary.csv"):
        files[str(ydir / name)] = helper.digest(ydir / name)
    for rel, sha in base["source_sha256"].items():
        files[str(ROOT / rel)] = sha
    for info in base["manifests"].values():
        files[info["path"]] = info["sha256"]
    pre = {"status": "PREFLIGHT_PASS", "version": VERSION, "script_sha256": helper.digest(__file__),
        "training_performed": False, "timing_claim": False, "new_inference_arms": [N_ARM],
        "reference_arms": list(REF_ARMS), "K": K, "gain": 1.0, "head_spec": infer_head_spec(head_state),
        "checkpoint": base["checkpoint"], "checkpoint_sha256": helper.P_SHA,
        "head_sha256": base["head_sha256"], "strict_cpu_load_passed": True,
        "y04_dir": str(Path(a.y04_dir).resolve()), "y04a_dir": str(ydir),
        "sequences": a.seqs, "partial_smoke": a.smoke_only,
        "manifests": base["manifests"], "input_sha256": files,
        "recursive_state": "native A radius/confidence/all four features/rays; no repaired value is committed",
        "output": "same frozen P_multi residual head, gain=1, temporary pre-readout features only",
        "native_A_guard": "all GT-matched RGB frames checked against original A metrics; no mask shrinkage",
        "state_guard": "tensor version counters on every output-only call; exact cloned-value checks at ages 1 and 31",
        "P_equivalence_guard": "keyframes and age1 matched-frame metrics compared to completed native P",
        "amp": "native A historical boundary; flow built before autocast; depth and output adapter inside autocast",
        "limits": ["Development temporal tails only; no independent generalization result",
            "Frozen-head feedback attribution; not a fair retraining comparison with output-only",
            "N may see a different state distribution from P training; P>N alone is not general necessity",
            "All radius/confidence/feature feedback is removed together, not an F-only ablation",
            "Extra N readout and diagnostics are not a new speed benchmark",
            "Historical per-pixel predictions/masks are unavailable; frame identity and valid counts are checked"]}
    return pre, manifests, refs, rt, head_state


def readout_radius(rt, model, features):
    """Exact native fast depth readout, without changing recursive state."""
    radial = model.pixel_decoder.radial_module
    require(len(features) == 4, "Expected four feature levels")
    lateral = [features[0]] + [radial.process_features[i](features[i + 1]) for i in range(len(radial.ups))]
    latent, outputs = lateral[0], []
    for i, up in enumerate(radial.ups):
        latent = up(latent + lateral[i + 1])
        outputs.append(latent)
    return rt.torch.exp(radial.depth_proj(outputs).clip(min=-8.0, max=8.0) + 2.0)


def output_only_radius(rt, model, head, native_state, debug):
    radius, confidence, features, rays = native_state
    tensors = [radius, confidence, rays, *features]
    versions = [t._version for t in tensors]
    clones = [t.detach().clone() for t in tensors] if debug else None
    if debug:
        reconstructed = readout_radius(rt, model, features)
        require(float((reconstructed.float() - radius.float()).abs().max().item()) <= 1e-3,
            "Native readout reconstruction differs")
    adapted = head(features)
    result = readout_radius(rt, model, adapted)
    require([t._version for t in tensors] == versions, "N adapter/readout modified native state in-place")
    if debug:
        require(all(float((t.float() - c.float()).abs().max().item()) == 0 for t, c in zip(tensors, clones)),
            "N adapter/readout changed native radius/confidence/features/rays")
    return result


def checked_metrics(rt, gt, pred, ref, label):
    valid = (gt > 0) & rt.np.isfinite(gt)
    require(rt.np.all(rt.np.isfinite(pred[valid]) & (pred[valid] > 0)), f"Invalid output on GT-valid pixels: {label}")
    actual = rt.gt2.calculate_metrics(gt, pred)
    require(all(math.isfinite(float(actual[m])) for m in METRICS), f"Nonfinite metrics: {label}")
    for key in ("valid_pixels", "total_pixels"):
        require(int(actual[key]) == int(ref[key]), f"Metric mask/canvas changed: {label}/{key}")
    return actual


def metric_deviation(actual, ref):
    return max(abs(float(actual[m]) - float(ref[m])) for m in METRICS)


def check_equivalence(actual, reference, label):
    dev = metric_deviation(actual, reference)
    require(dev <= METRIC_TOL, f"{label}: metric deviation={dev} exceeds {METRIC_TOL}")
    return dev


def run_sequence(a, helper, rt, seq, manifest, refs, head_state, out):
    torch, exp1, gt2 = rt.torch, rt.exp1, rt.gt2
    device = torch.device("cuda")
    model = gt2.load_model(Path(a.snapshot).resolve(), "off", device)
    require(model.pixel_decoder.feature_repair_mode == "off" and model.pixel_decoder.persistent_feature_repair is None,
        "N requires the native A model, without an internal repair head")
    head = build_head(head_state)
    head.to(device)
    require(int(model.pixel_decoder.radial_module.hidden_dim) == head.channels == 512, "Readout/head channels differ")
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    head.eval()
    indices = helper.plan_indices(seq, len(manifest), a.smoke_only)
    shape = exp1.shape_info_for(model, Path(manifest[indices[0]]["rgb_path"]), device)
    dis = rt.cv2.DISOpticalFlow_create(rt.cv2.DISOPTICAL_FLOW_PRESET_FAST)
    calls = {"n": 0}
    def count_forward(*_args):
        calls["n"] += 1
    hook = head.register_forward_hook(count_forward)
    prev, state = None, None
    rows, checks = [], []
    start = time.perf_counter()
    try:
        with torch.no_grad():
            for pos, idx in enumerate(indices):
                rec = manifest[idx]
                matched = str(rec.get("gt_match_status", "")).strip().upper() == "MATCHED"
                frame = exp1.load_frame(rec, shape, device, need_gt=matched)
                key, age = idx % K == 0, idx % K
                if not key:
                    require(state is not None and prev is not None, "Missing native A previous state")
                    static = exp1.transition_static(model, prev, frame, shape, dis)
                before = calls["n"]
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    if key:
                        rays, radius, confidence, features = exp1.keyframe_state(model, frame, shape)
                    else:
                        old_r, old_c, old_f, rays = state
                        radius, confidence, features = exp1.one_fast_step(model, prev, frame, static, old_r, old_c, old_f)
                    # Commit ONLY native A values. output_only_radius returns a temporary output.
                    state = (radius, confidence, features, rays)
                    require(tuple(model.pixel_decoder.radial_module.original_shapes) == tuple(shape["network_shape"]),
                        "Readout grid is not native network_shape")
                    native_pred_t = gt2.make_z_depth(radius, rays, shape["padded_shape"], shape["paddings"], model.interpolation_mode)
                    output_radius = radius if key else output_only_radius(rt, model, head, state, age in (1, 31))
                    pred_t = native_pred_t if key else gt2.make_z_depth(output_radius, rays,
                        shape["padded_shape"], shape["paddings"], model.interpolation_mode)
                helper.check_head_calls(before, calls["n"], key)
                pred = pred_t.detach().float().cpu().numpy()
                require(tuple(pred.shape) == tuple(shape["original_shape"]), "N output geometry differs")
                metrics = {m: "" for m in (*METRICS, "valid_ratio", "gt_valid_ratio")}
                metrics.update(valid_pixels=0, total_pixels=int(rt.np.prod(shape["original_shape"])))
                a_dev, p_dev = "", ""
                if matched:
                    aref = refs[seq, A_ARM, idx]
                    native_pred = native_pred_t.detach().float().cpu().numpy()
                    native_metrics = checked_metrics(rt, frame["gt_np"], native_pred, aref, f"A guard {seq}/{idx}")
                    a_dev = check_equivalence(native_metrics, aref, f"Native A trajectory changed {seq}/{idx}")
                    metrics = checked_metrics(rt, frame["gt_np"], pred, aref, f"N {seq}/{idx}")
                    if age in (0, 1):
                        p_dev = check_equivalence(metrics, refs[seq, P_ARM, idx], f"P/N age{age} identity failed {seq}/{idx}")
                rows.append({"sequence": seq, "arm": N_ARM, "frame_index_zero_based": idx,
                    "refresh_cycle": idx // K, "frame_offset_in_cycle": age,
                    "is_keyframe": str(key).lower(), "gt_matched": str(matched).lower(),
                    "rgb_file": Path(rec["rgb_path"]).name, "gt_file": Path(rec["depth_path"]).name if matched else "",
                    "y04_eval_start_index": TAIL[seq],
                    "y04_split": "temporal_tail_eval" if idx >= TAIL[seq] else "train_prefix_not_scored",
                    "mode": "output_only_same_P_head_native_A_state", "gain": 1.0,
                    "head_calls_this_frame": calls["n"] - before, "additional_readout_passes": 0 if key else 1,
                    **metrics})
                checks.append({"sequence": seq, "frame_index_zero_based": idx, "age": age,
                    "gt_matched": matched, "observed_head_calls": calls["n"] - before,
                    "native_A_max_metric_deviation": a_dev, "P_N_age0_or_1_max_metric_deviation": p_dev,
                    "state_version_guard": "not_applicable_keyframe" if key else "pass",
                    "state_exact_clone_guard": "pass" if age in (1, 31) else "not_sampled",
                    "same_gt_valid_pixel_count": True if matched else "not_scored", "passed": True})
                prev = frame
                if pos == 0 or (pos + 1) % 100 == 0 or pos + 1 == len(indices):
                    print(f"[N {seq}] {pos+1}/{len(indices)} RGB; global_index={idx}; A/P guards pass", flush=True)
        require(calls["n"] == sum(i % K != 0 for i in indices), "Total N head-call count differs")
        require(model.pixel_decoder.feature_repair_mode == "off" and model.pixel_decoder.persistent_feature_repair is None,
            "Internal model unexpectedly enabled repair")
        require(not head.training and all(not p.requires_grad for p in head.parameters()), "N head was not frozen/eval")
        helper.csv_write(out / "progress" / f"{seq}_N_frames.csv", rows)
        helper.csv_write(out / "progress" / f"{seq}_N_checks.csv", checks)
        return rows, checks, {"sequence": seq, "processed_rgb_frames": len(indices),
            "keyframe_calls": sum(i % K == 0 for i in indices), "head_calls": calls["n"],
            "additional_readout_passes": calls["n"], "diagnostic_wall_sec": time.perf_counter() - start,
            "timing_claim": False, "native_A_metric_checks": sum(c["gt_matched"] for c in checks),
            "P_N_age0_or_1_metric_checks": sum(c["P_N_age0_or_1_max_metric_deviation"] != "" for c in checks)}
    finally:
        hook.remove()
        del model, head
        torch.cuda.empty_cache()


def feedback_row(helper, seq, scope, n_rows, refs, **extra):
    scored = [r for r in n_rows if helper.truth(r["gt_matched"])]
    require(bool(scored), "Empty feedback group")
    p = [refs[seq, P_ARM, int(r["frame_index_zero_based"])] for r in scored]
    diffs = [float(n["AbsRel"]) - float(pr["AbsRel"]) for n, pr in zip(scored, p)]
    nmean, pmean = [statistics.mean(float(r["AbsRel"]) for r in rr) for rr in (scored, p)]
    return {"sequence": seq, "scope": scope, "rgb_frames": len(n_rows), "scored_frames": len(scored),
        "N_mean_AbsRel": nmean, "P_mean_AbsRel": pmean,
        "feedback_benefit_AbsRel_N_minus_P": statistics.mean(diffs),
        "P_relative_AbsRel_change_pct_vs_N": 100 * (pmean / nmean - 1),
        "feedback_benefit_RMSE_N_minus_P": statistics.mean(float(n["RMSE"]) - float(pr["RMSE"]) for n, pr in zip(scored, p)),
        "feedback_benefit_delta1_P_minus_N": statistics.mean(float(pr["delta1"]) - float(n["delta1"]) for n, pr in zip(scored, p)),
        "frames_P_better_than_N_AbsRel": sum(d > HARM_TOL for d in diffs),
        "frames_P_worse_than_N_AbsRel": sum(d < -HARM_TOL for d in diffs), **extra}


def build_tables(helper, new_rows, refs):
    combined, paired = [], []
    for row in new_rows:
        seq, idx = row["sequence"], int(row["frame_index_zero_based"])
        if idx < TAIL[seq]:
            continue
        combined.extend([row, *(dict(refs[seq, arm, idx]) for arm in REF_ARMS)])
        if helper.truth(row["gt_matched"]):
            pr = refs[seq, P_ARM, idx]
            paired.append({"sequence": seq, "frame_index_zero_based": idx, "age": idx % K, "refresh_cycle": idx // K,
                "N_AbsRel": float(row["AbsRel"]), "P_AbsRel": float(pr["AbsRel"]),
                "feedback_benefit_AbsRel_N_minus_P": float(row["AbsRel"]) - float(pr["AbsRel"]),
                "feedback_benefit_RMSE_N_minus_P": float(row["RMSE"]) - float(pr["RMSE"]),
                "feedback_benefit_delta1_P_minus_N": float(pr["delta1"]) - float(row["delta1"])})
    groups = defaultdict(list)
    lookup = {}
    for row in combined:
        key = row["sequence"], row["arm"], int(row["frame_index_zero_based"])
        require(key not in lookup, f"Duplicate comparison row: {key}")
        lookup[key] = row
        groups[key[:2]].append(row)
    sequences, cycles, ages, feedback = [], [], [], []
    for (seq, arm), rows in sorted(groups.items()):
        scored = [r for r in rows if helper.truth(r["gt_matched"])]
        require(bool(scored), f"No scored tail: {seq}/{arm}")
        diffs = [float(r["AbsRel"]) - float(lookup[seq, A_ARM, int(r["frame_index_zero_based"])]["AbsRel"]) for r in scored]
        bycycle, byage = defaultdict(list), defaultdict(list)
        for r in rows:
            bycycle[int(r["refresh_cycle"])].append(r)
        for r in scored:
            byage[helper.age_bin(int(r["frame_offset_in_cycle"]))].append(r)
        complete_harm = []
        for cycle, rr in sorted(bycycle.items()):
            ss = [r for r in rr if helper.truth(r["gt_matched"])]
            if not ss:
                continue
            complete = len(rr) == K and {int(r["frame_offset_in_cycle"]) for r in rr} == set(range(K))
            delta = statistics.mean(float(r["AbsRel"]) - float(lookup[seq, A_ARM, int(r["frame_index_zero_based"])]["AbsRel"]) for r in ss)
            if complete:
                complete_harm.append(delta > HARM_TOL)
            cycles.append({"sequence": seq, "arm": arm, "refresh_cycle": cycle, "complete_rgb_cycle": complete,
                "rgb_frames": len(rr), "scored_frames": len(ss),
                **{f"mean_{m}": statistics.mean(float(r[m]) for r in ss) for m in METRICS},
                "delta_AbsRel_vs_A": delta, "harmed_vs_A": delta > HARM_TOL})
            if arm == N_ARM:
                feedback.append(feedback_row(helper, seq, "cycle", rr, refs, refresh_cycle=cycle, complete_rgb_cycle=complete))
        propagated = [r for r in scored if not helper.truth(r["is_keyframe"])]
        entry = {"sequence": seq, "arm": arm, "n_frames": len(scored), "n_rgb_frames": len(rows), "eval_start_index": TAIL[seq],
            **{f"mean_{m}": statistics.mean(float(r[m]) for r in scored) for m in METRICS},
            "frame_AbsRel_p95": helper.percentile([r["AbsRel"] for r in scored], 95),
            "propagated_frame_AbsRel_p95": helper.percentile([r["AbsRel"] for r in propagated], 95) if propagated else "",
            "harmed_frames_vs_A": sum(d > HARM_TOL for d in diffs),
            "harmed_frame_fraction_vs_A": statistics.mean(d > HARM_TOL for d in diffs),
            "complete_rgb_cycles": len(complete_harm), "harmed_complete_cycles_vs_A": sum(complete_harm),
            "harm_tolerance_absrel": HARM_TOL}
        for refarm, label in ((A_ARM, "A"), (O_ARM, "O_multi"), (P_ARM, "P_multi")):
            refmean = statistics.mean(float(lookup[seq, refarm, int(r["frame_index_zero_based"])]["AbsRel"]) for r in scored)
            entry[f"relative_AbsRel_change_pct_vs_{label}"] = 100 * (entry["mean_AbsRel"] / refmean - 1)
        sequences.append(entry)
        for age, rr in byage.items():
            ages.append({"sequence": seq, "arm": arm, "age_bin": age, "n_frames": len(rr),
                **{f"mean_{m}": statistics.mean(float(r[m]) for r in rr) for m in METRICS}})
            if arm == N_ARM:
                feedback.append(feedback_row(helper, seq, "age", rr, refs, age_bin=age))
        if arm == N_ARM:
            feedback.append(feedback_row(helper, seq, "tail", rows, refs))
    macro = []
    for arm in ARMS:
        rr = [r for r in sequences if r["arm"] == arm]
        macro.append({"arm": arm, "n_sequences": len(rr),
            **{f"sequence_macro_{m}": statistics.mean(r[f"mean_{m}"] for r in rr) for m in METRICS}})
    return {"y07w_all_frame_metrics.csv": new_rows, "y07w_temporal_tail_metrics.csv": combined,
        "y07w_feedback_paired_frames.csv": paired, "y07w_sequence_summary.csv": sequences,
        "y07w_cycle_summary.csv": cycles, "y07w_age_summary.csv": ages,
        "y07w_feedback_summary.csv": feedback, "y07w_macro_summary.csv": macro}


def review_archive(out, helper, pre):
    dest, tmp = out / "y07w_review.zip", out / "y07w_review.zip.tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=7) as z:
        for pattern in ("y07w_*.csv", "y07w_*.json"):
            for path in sorted(out.glob(pattern)):
                if path.name != "y07w_failure.json":
                    z.write(path, path.name)
        z.write(__file__, Path(__file__).name)
        sources = ["experiments/p1/gt4a2_y04a_native_pre_readout_eval.py", *helper.SOURCE_SHA,
            "velodepth/models/velodepth.py", "velodepth/models/encoder_velo.py"]
        for rel in sources:
            z.write(ROOT / rel, "source/" + rel)
        for name in ("y04a_summary.json", "y04a_preflight.json", "y04a_sequence_summary.csv"):
            z.write(Path(pre["y04a_dir"]) / name, "reference/" + name)
    with zipfile.ZipFile(tmp) as z:
        require(z.testzip() is None, "Result archive integrity check failed")
    tmp.replace(dest)
    return dest


def main(argv=None):
    a = parse_args(argv)
    if a.self_test:
        return self_test()
    helper = import_helper()
    out, ydir = Path(a.out_dir).resolve(), Path(a.y04_dir).resolve()
    y04a = resolve_y04a(a.y04a_dir, helper)
    require(out != y04a and y04a not in out.parents, "Output must be outside existing Y04-A results")
    a.y04a_dir = str(y04a)
    require(not (out / "y07w_summary.json").exists(), "Completed Y07-W output exists; choose a new directory")
    print(f"[Y04-A REFERENCE] {y04a}", flush=True)
    with helper.exclusive_output(out, ydir):
        require(not (out / "y07w_summary.json").exists(), "Completed Y07-W output exists; choose a new directory")
        try:
            pre, manifests, refs, rt, head_state = preflight(a, helper)
            previous = out / "y07w_preflight.json"
            if previous.exists():
                require(helper.json_read(previous) == pre, "Existing output identity differs; choose a new directory")
            helper.json_write(previous, pre)
            print(f"[PREFLIGHT_PASS] {VERSION}; same P head 512/32/4 strictly loaded; only N inference", flush=True)
            if a.preflight_only:
                print("No model inference or training performed.", flush=True)
                return 0
            require(rt.torch.cuda.is_available(), "CUDA is required")
            rt.torch.set_num_threads(4)
            rt.torch.manual_seed(20261025)
            rt.np.random.seed(20261025)
            rt.torch.cuda.manual_seed_all(20261025)
            all_rows, all_checks, execution = [], [], []
            for seq in a.seqs:
                rr, cc, meta = run_sequence(a, helper, rt, seq, manifests[seq], refs, head_state, out)
                all_rows.extend(rr); all_checks.extend(cc); execution.append(meta)
            for path, sha in pre["input_sha256"].items():
                require(helper.digest(path) == sha, f"Input changed during evaluation: {path}")
            require(helper.digest(__file__) == pre["script_sha256"], "Y07-W script changed while evaluating")
            tables = build_tables(helper, all_rows, refs)
            tables["y07w_invariants.csv"] = all_checks
            for name, rows in tables.items():
                helper.csv_write(out / name, rows)
            helper.json_write(out / "y07w_execution.json", execution)
            summary = {"status": "SMOKE_PASS" if a.smoke_only else "MECHANICS_PASS", "version": VERSION,
                "training_performed": False, "timing_claim": False, "partial_smoke": a.smoke_only,
                "sequences": a.seqs, "full_three_sequence_evaluation": not a.smoke_only and set(a.seqs) == set(TAIL),
                "new_inference_arms": [N_ARM], "reference_arms": list(REF_ARMS), "K": K, "gain": 1.0,
                "checkpoint_sha256": helper.P_SHA, "head_sha256": pre["head_sha256"],
                "new_rgb_records": len(all_rows), "tail_comparison_records": len(tables["y07w_temporal_tail_metrics.csv"]),
                "scored_feedback_pairs": len(tables["y07w_feedback_paired_frames.csv"]), "invariants_passed": True,
                "native_A_matched_frame_checks": sum(e["native_A_metric_checks"] for e in execution),
                "P_N_age0_or_1_metric_checks": sum(e["P_N_age0_or_1_metric_checks"] for e in execution),
                "script_sha256": helper.digest(__file__), "preflight_sha256": helper.digest(previous),
                "feedback_sign": "N-P AbsRel/RMSE > 0 means existing P feedback helps; delta1 uses P-N",
                "method_verdict": "NOT_AUTOMATICALLY_DECIDED; inspect sequence/cycle/age, harm and fair O baseline",
                "limits": pre["limits"]}
            helper.json_write(out / "y07w_summary.json", summary)
            dest = review_archive(out, helper, pre)
            (out / "y07w_failure.json").unlink(missing_ok=True)
            for row in tables["y07w_feedback_summary.csv"]:
                if row["scope"] == "tail":
                    print(f"[FEEDBACK] {row['sequence']}: N={row['N_mean_AbsRel']:.6f}; P={row['P_mean_AbsRel']:.6f}; "
                        f"N-P={row['feedback_benefit_AbsRel_N_minus_P']:+.6f}", flush=True)
            print(f"[{summary['status']}] {dest}", flush=True)
            return 0
        except Exception as exc:
            (out / "y07w_summary.json").unlink(missing_ok=True)
            helper.json_write(out / "y07w_failure.json", {"status": "FAILED", "error": str(exc),
                "traceback": traceback.format_exc(), "training_performed": False})
            raise


def self_test():
    import unittest

    class Checks(unittest.TestCase):
        def test_feedback_positive_means_P_helps(self):
            helper = SimpleNamespace(truth=lambda x: str(x).lower() == "true")
            rows = [{"gt_matched": True, "frame_index_zero_based": 256, "AbsRel": .2, "RMSE": .3, "delta1": .8}]
            refs = {("balloon2", P_ARM, 256): {"AbsRel": .1, "RMSE": .2, "delta1": .9}}
            row = feedback_row(helper, "balloon2", "tail", rows, refs)
            self.assertAlmostEqual(row["feedback_benefit_AbsRel_N_minus_P"], .1)
            self.assertAlmostEqual(row["feedback_benefit_delta1_P_minus_N"], .1)
            self.assertEqual(row["frames_P_better_than_N_AbsRel"], 1)

        def test_reference_mismatch_is_rejected(self):
            ref = dict(AbsRel=.1, RMSE=.2, delta1=.9)
            self.assertEqual(check_equivalence(ref, ref, "equal"), 0)
            with self.assertRaisesRegex(RuntimeError, "age1"):
                check_equivalence(dict(ref, AbsRel=.11), ref, "age1")

        def test_head_spec_uses_512(self):
            block = {"0.weight": (32,512,1,1), "1.weight": (32,), "1.bias": (32,),
                "3.weight": (32,32,3,3), "4.weight": (32,), "4.bias": (32,),
                "6.weight": (512,32,1,1), "6.bias": (512,)}
            state = {f"blocks.{i}.{key}": SimpleNamespace(shape=shape) for i in range(4) for key, shape in block.items()}
            self.assertEqual(infer_head_spec(state), {"channels":512, "bottleneck":32, "levels":4})
            state["blocks.3.6.bias"] = SimpleNamespace(shape=(256,))
            with self.assertRaisesRegex(RuntimeError, "blocks.3.6.bias"):
                infer_head_spec(state)

        def test_empty_snapshot_is_clear(self):
            from unittest.mock import patch
            import io
            with patch.dict(os.environ, {"VDEEP_SNAPSHOT":""}), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args([])

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
