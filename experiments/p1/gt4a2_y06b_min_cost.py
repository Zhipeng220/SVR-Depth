#!/usr/bin/env python3
"""Minimal Y06-B cost check: frozen A / O_multi / native P_multi, K=32.

Three balanced repeats after 64 warm-up frames. Existing Y04/Y04-A quality
is reused. GT/metrics, model loading, warm-up, telemetry and writes are untimed.
Only small output samples are checked against the existing metric CSVs.
Requires the previously installed gt4a2_y04a_native_pre_readout_eval.py.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
import csv
import gc
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[2]
VERSION = "Y06-B-min-1.1-Fix1"
ARMS = ("A", "O_multi", "P_multi")
REF = {"A": "A_native", "O_multi": "Y04_O_multi", "P_multi": "Y04A_P_multi_pre_readout_gain1"}
O_SHA = "558e1850d7b08939bef0db0eb4f7a48fdff8a2fb58417cd4c380ee3c6a6f6948"
Y04A_SHA = "6fbe1a2464a7f9756f9ea68cf05f3476a89b005b8e6f889dd1a0093f231c3116"
K, WARMUP = 32, 64
SEQUENCES = ("crowd2", "crowd3", "balloon2")
TAIL = {"crowd2": 768, "crowd3": 256, "balloon2": 256}
EXPECTED_HEAD_SPEC = {"channels": 512, "bottleneck": 32, "levels": 4}

def require(test, message):
    if not test:
        raise RuntimeError(message)


def import_helper():
    import hashlib
    path = ROOT / "experiments/p1/gt4a2_y04a_native_pre_readout_eval.py"
    require(path.is_file(), f"Install the original Y04-A script first: {path}")
    require(hashlib.sha256(path.read_bytes()).hexdigest() == Y04A_SHA, "Y04-A helper differs from the completed run")
    spec = importlib.util.spec_from_file_location("y06b_y04a_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def orders(repeats, sequence_index=0):
    return [list(ARMS[(r + sequence_index) % 3:] + ARMS[:(r + sequence_index) % 3]) for r in range(repeats)]


def expected_counts(arm, indices):
    keys = sum(i % K == 0 for i in indices)
    prop = len(indices) - keys
    return {"keyframe_calls": keys, "flow_calls": prop, "native_fast_calls": prop,
        "repair_head_calls": prop if arm == "P_multi" else 0,
        "output_adapter_calls": prop if arm == "O_multi" else 0,
        "additional_readout_passes": prop if arm == "O_multi" else 0}


def measured_call(fn, sync, clock=time.perf_counter):
    sync()
    start = clock()
    result = fn()
    sync()
    return result, (clock() - start) * 1000.0


def args_parser(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", default=os.environ.get("VDEEP_SNAPSHOT", ""))
    p.add_argument("--y04-dir", default="outputs/gt4a2_y04_validation_fix3_20261025_225902")
    p.add_argument("--y04a-dir", default="auto", help="auto selects the latest completed three-sequence native Y04-A run")
    p.add_argument("--sequences", default=",".join(SEQUENCES))
    for seq in SEQUENCES:
        p.add_argument(f"--{seq}-pair-csv", default=f"outputs/gt4a2_exp1/gt4a2_exp1_{seq}_pairs_used.csv")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--smoke-only", action="store_true")
    modes.add_argument("--self-test", action="store_true", help="Stdlib-only checks; no model/CUDA needed")
    p.add_argument("--nvidia-smi", default="auto", help="auto also tries /usr/lib/wsl/lib/nvidia-smi")
    p.add_argument("--out-dir", default="outputs/gt4a2_y06b_min_validation")
    a = p.parse_args(argv)
    a.seqs = [s.strip().lower() for s in a.sequences.split(",") if s.strip()]
    if not a.seqs or len(set(a.seqs)) != len(a.seqs) or set(a.seqs) - set(SEQUENCES):
        p.error("Choose unique crowd2,crowd3,balloon2 sequences")
    if not a.self_test and not a.snapshot:
        p.error("Set VDEEP_SNAPSHOT or pass --snapshot")
    return a


def find_y04a(helper, value, announce=True):
    candidates = [Path(value).resolve()] if value != "auto" else list((ROOT / "outputs").glob("gt4a2_y04a_validation*"))
    valid = []
    for directory in candidates:
        try:
            summary = helper.json_read(directory / "y04a_summary.json")
            if (summary.get("status") == "MECHANICS_PASS" and summary.get("full_three_sequence_evaluation")
                and not summary.get("partial_smoke") and summary.get("script_sha256") == Y04A_SHA
                and summary.get("checkpoint_sha256") == helper.P_SHA
                and summary.get("mode") == "pre_readout" and summary.get("gain") == 1.0):
                valid.append(directory.resolve())
        except (OSError, ValueError):
            continue
    require(bool(valid), "No matching completed Y04-A run found; pass --y04a-dir with its output directory")
    chosen = max(valid, key=lambda x: (x / "y04a_summary.json").stat().st_mtime_ns)
    if announce:
        print(f"[Y04-A REFERENCE] {chosen}", flush=True)
    return chosen


def readout_radius(rt, model, features):
    """Exact Q0-3/Y04 output-only depth readout; never commit adapted features."""
    radial = model.pixel_decoder.radial_module
    require(len(features) == 4, "Expected four feature levels")
    lateral = [features[0]] + [radial.process_features[i](features[i + 1]) for i in range(len(radial.ups))]
    latent, outputs = lateral[0], []
    for i, up in enumerate(radial.ups):
        latent = up(latent + lateral[i + 1])
        outputs.append(latent)
    logdepth = radial.depth_proj(outputs)
    return rt.torch.exp(logdepth.clip(min=-8.0, max=8.0) + 2.0)


def infer_head_spec(state):
    """Infer actual checkpoint dimensions and validate every residual block."""
    require("blocks.0.0.weight" in state, "Missing adapter input projection")
    first = tuple(state["blocks.0.0.weight"].shape)
    require(len(first) == 4 and first[2:] == (1, 1), "Invalid adapter input projection")
    bottleneck, channels = first[:2]
    levels = len([key for key in state if key.startswith("blocks.") and key.endswith(".0.weight")])
    require(min(channels, bottleneck, levels) > 0, "Invalid adapter dimensions")
    block_shapes = {"0.weight": (bottleneck, channels, 1, 1),
        "1.weight": (bottleneck,), "1.bias": (bottleneck,),
        "3.weight": (bottleneck, bottleneck, 3, 3),
        "4.weight": (bottleneck,), "4.bias": (bottleneck,),
        "6.weight": (channels, bottleneck, 1, 1), "6.bias": (channels,)}
    expected = {f"blocks.{level}.{key}": shape for level in range(levels) for key, shape in block_shapes.items()}
    require(set(state) == set(expected), "Adapter state keys differ from the audited residual architecture")
    for key, shape in expected.items():
        require(tuple(state[key].shape) == shape,
            f"Adapter shape mismatch: {key}: expected {shape}, found {tuple(state[key].shape)}")
    return {"channels": int(channels), "bottleneck": int(bottleneck), "levels": levels}


def build_output_head(state):
    from velodepth.models.persistent_feature_repair import PersistentFeatureRepairHead
    spec = infer_head_spec(state)
    require(spec == EXPECTED_HEAD_SPEC, f"Pinned Y04 adapter architecture differs: {spec}")
    head = PersistentFeatureRepairHead(**spec)
    head.load_state_dict(state, strict=True)
    require(sum(p.numel() for p in head.parameters()) == 170496, "Adapter parameter count differs")
    return head


def preflight(a, helper):
    qa = SimpleNamespace(snapshot=a.snapshot, y04_dir=a.y04_dir, p_checkpoint="", seqs=a.seqs,
        smoke_only=False, **{seq + "_pair_csv": getattr(a, seq + "_pair_csv") for seq in SEQUENCES})
    base, manifests, oldrefs, rt, p_state = helper.preflight(qa)
    y04a_dir = find_y04a(helper, a.y04a_dir, announce=False)
    ypre = helper.json_read(y04a_dir / "y04a_preflight.json")
    require(ypre["source_sha256"] == base["source_sha256"], "Runtime sources changed since Y04-A")
    require(ypre["snapshot_config_sha256"] == base["snapshot_config_sha256"], "Snapshot config changed")
    require(ypre["reference_csv_sha256"] == base["reference_csv_sha256"], "Y04 reference CSV changed")
    cp = Path(a.y04_dir).resolve() / "training/O_multi/Y04_O_multi_final_step1000.pt"
    require(cp.is_file() and helper.digest(cp) == O_SHA, "O_multi final checkpoint missing or SHA differs")
    summary = helper.json_read(cp.parent / "Y04_O_multi_training_summary.json")
    require(summary["checkpoint_sha256"] == O_SHA, "O_multi training summary/checkpoint mismatch")
    ck = rt.torch.load(cp, map_location="cpu", weights_only=False)
    require(ck.get("mode") == "Y04_O_multi" and int(ck.get("step_count", -1)) == 1000
        and ck.get("y04_version") == "Y04-Fix3" and ck.get("y04_label") == "Y04_O_multi"
        and ck.get("pretrained_weights_sha256") == helper.MODEL_SHA
        and ck.get("training_rgb_indices_by_sequence") == {s: [0,255] for s in SEQUENCES},
        "O_multi checkpoint provenance differs")
    o_state = ck["adapter_state_dict"]
    require(all(rt.torch.isfinite(x).all().item() for x in o_state.values()), "Nonfinite O_multi weights")
    p_spec, o_spec = infer_head_spec(p_state), infer_head_spec(o_state)
    require(p_spec == o_spec == EXPECTED_HEAD_SPEC, f"P/O adapter architecture differs: P={p_spec}, O={o_spec}")
    # Strict CPU load before any model inference; catch architecture mistakes in preflight.
    checked_head = build_output_head(o_state)
    del checked_head
    print(f"[ADAPTER_CHECK_PASS] channels={o_spec['channels']}, bottleneck={o_spec['bottleneck']}, levels={o_spec['levels']}; strict O checkpoint load", flush=True)
    refs = {(s, arm, i): oldrefs[(s, REF[arm], i)] for s in a.seqs for arm in ("A", "O_multi")
            for i in range(len(manifests[s]))}
    p_rows = helper.csv_read(y04a_dir / "y04a_all_frame_metrics.csv")
    for row in p_rows:
        if row["sequence"] not in a.seqs:
            continue
        key = row["sequence"], "P_multi", int(row["frame_index_zero_based"])
        require(row["arm"] == REF["P_multi"] and key not in refs, f"Unexpected/duplicate Y04-A row: {key}")
        refs[key] = row
    quality = []
    for seq in a.seqs:
        for arm in ARMS:
            seq_rows = [refs[(seq, arm, i)] for i in range(len(manifests[seq]))]
            for i, row in enumerate(seq_rows):
                ref = refs[seq, "A", i]
                require(row["rgb_file"] == ref["rgb_file"] and row["gt_file"] == ref["gt_file"]
                    and helper.truth(row["gt_matched"]) == helper.truth(ref["gt_matched"]), f"Reference frame identity differs: {seq}/{arm}/{i}")
            scored = [r for r in seq_rows if int(r["frame_index_zero_based"]) >= TAIL[seq] and helper.truth(r["gt_matched"])]
            quality.append({"sequence": seq, "arm": arm, "quality_source": "existing Y04/Y04-A CSV; no new quality sweep",
                "scored_tail_frames": len(scored),
                **{f"mean_{metric}": statistics.mean(float(r[metric]) for r in scored) for metric in helper.METRICS}})
    files = {str(y04a_dir / name): helper.digest(y04a_dir / name) for name in
        ("y04a_summary.json", "y04a_preflight.json", "y04a_all_frame_metrics.csv", "y04a_sequence_summary.csv")}
    files[str(Path(a.y04_dir).resolve() / "y04_all_frame_metrics.csv")] = base["reference_csv_sha256"]
    files[str(cp)] = O_SHA
    files[base["checkpoint"]] = helper.P_SHA
    for info in base["manifests"].values():
        files[info["path"]] = info["sha256"]
    files[str(ROOT / "experiments/p1/gt4a2_y04a_native_pre_readout_eval.py")] = Y04A_SHA
    for rel, value in base["source_sha256"].items():
        files[str(ROOT / rel)] = value
    info = {"status": "PREFLIGHT_PASS", "version": VERSION, "training_performed": False,
        "script_sha256": helper.digest(__file__), "snapshot": str(Path(a.snapshot).resolve()),
        "y04_dir": str(Path(a.y04_dir).resolve()), "y04a_dir": str(y04a_dir),
        "sequences": a.seqs, "arms": list(ARMS), "K": K, "warmup_rgb_frames": WARMUP,
        "repeats": 1 if a.smoke_only else 3, "partial_smoke": a.smoke_only,
        "P_mode": "pre_readout", "P_gain": 1.0, "P_checkpoint_sha256": helper.P_SHA,
        "O_checkpoint_sha256": O_SHA, "adapter_spec": o_spec, "O_strict_cpu_load_passed": True, "input_sha256": files,
        "orders": {s: orders(1 if a.smoke_only else 3, SEQUENCES.index(s)) for s in a.seqs},
        "processing_ranges": {s: [TAIL[s], min(len(manifests[s]), TAIL[s]+64)] if a.smoke_only else [0,len(manifests[s])] for s in a.seqs},
        "timing": {"input": "RGB files; OS cache primed before all trials for each sequence",
            "includes": ["RGB decode/read", "preprocessing and H2D", "CPU DIS flow", "native propagation/keyframe",
                "required repair or output adapter/readout", "z-depth restoration and CPU output", "CUDA completion"],
            "excludes": ["model/checkpoint loading", "64-frame warm-up", "GT reads and metrics",
                "output consistency checks", "telemetry", "CSV/log writes", "debug clones and hooks"],
            "schedule": "serialized per-frame latency, not asynchronous maximum throughput"},
        "reporting": "Tail RGB latency is primary and paired with existing tail quality; whole-stream latency also reported",
        "limits": ["No training, K scan, Auto, new refiner or device deployment", "No quality-matched speedup claim",
            "GPU monitoring may be unavailable; unverified monitoring is explicitly labeled",
            "Endpoint process queries do not exclude transient interference"]}
    return info, manifests, refs, quality, rt, p_state, o_state


def gpu_status(rt, executable):
    """Best-effort endpoint check, with explicit WSL/namespace limitations."""
    binary = shutil.which("nvidia-smi") if executable == "auto" else executable
    if not binary and executable == "auto" and Path("/usr/lib/wsl/lib/nvidia-smi").is_file():
        binary = "/usr/lib/wsl/lib/nvidia-smi"
    out = {"available": False, "mapping_verified": False, "process_query_verified": False,
        "foreign_processes_confirmed": [], "unmapped_processes": []}
    if not binary:
        out["error"] = "nvidia-smi not found in PATH or /usr/lib/wsl/lib"
        return out
    try:
        def query(flag, fields):
            result = subprocess.run([binary, flag + fields, "--format=csv,noheader,nounits"],
                text=True, capture_output=True, timeout=10, check=True)
            return list(csv.reader(io.StringIO(result.stdout)))
        inventory = query("--query-gpu=", "index,uuid,name,temperature.gpu,clocks.sm,power.draw")
        cuda_uuid = str(getattr(rt.torch.cuda.get_device_properties(0), "uuid", ""))
        selected = next((x for x in inventory if x[1].strip() == cuda_uuid), None)
        if selected is None and len(inventory) == 1 and rt.torch.cuda.device_count() == 1:
            selected = inventory[0]
        if selected is None:
            out["error"] = "Cannot unambiguously map CUDA device 0 to nvidia-smi GPU"
            return out
        out.update(available=True, mapping_verified=True, executable=binary,
            gpu_uuid=selected[1].strip(), gpu_name=selected[2].strip(),
            temperature_c=selected[3].strip(), sm_clock_mhz=selected[4].strip(), power_w=selected[5].strip())
        for row in query("--query-compute-apps=", "gpu_uuid,pid,process_name"):
            if len(row) < 3 or row[0].strip() != out["gpu_uuid"]:
                continue
            try:
                pid = int(row[1].strip())
            except ValueError:
                out["unmapped_processes"].append(row)
                continue
            if pid == os.getpid():
                continue
            # Host/container/WSL PIDs need not match this Linux PID namespace.
            command = Path(f"/proc/{pid}/cmdline")
            raw_command = command.read_bytes().split(b"\0")[0] if command.is_file() else b""
            local_name = Path(raw_command.decode(errors="replace")).name
            reported_name = Path(row[2].strip()).name
            same_program = local_name == reported_name or (local_name.startswith("python") and reported_name.startswith("python"))
            if raw_command and same_program:
                out["foreign_processes_confirmed"].append({"pid":pid,"name":row[2].strip()})
            else:
                out["unmapped_processes"].append(row)
        out["process_query_verified"] = not out["unmapped_processes"]
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
        out["error"] = str(exc)
    return out


def telemetry_clean(info):
    return info["available"] and info["mapping_verified"] and info["process_query_verified"] and not info["foreign_processes_confirmed"]


def check_busy(info):
    require(not info["foreign_processes_confirmed"],
        f"Other GPU compute processes are visible; pause them before timing: {info['foreign_processes_confirmed']}")


def make_model(rt, a, arm, p_state, o_state):
    device = rt.torch.device("cuda:0")
    model = rt.gt2.load_model(Path(a.snapshot).resolve(), "pre_readout" if arm == "P_multi" else "off", device)
    head = None
    if arm == "P_multi":
        head = model.pixel_decoder.persistent_feature_repair
        require(model.pixel_decoder.feature_repair_mode == "pre_readout" and head is not None, "P native mode differs")
        head.load_state_dict(p_state, strict=True)
    elif arm == "O_multi":
        head = build_output_head(o_state)
        head.to(device)
    if head is not None:
        require(int(model.pixel_decoder.radial_module.hidden_dim) == infer_head_spec(
            p_state if arm == "P_multi" else o_state)["channels"], "Adapter/readout channels differ")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    if head is not None:
        for p in head.parameters():
            p.requires_grad_(False)
        head.eval()
        require(sum(p.numel() for p in head.parameters()) == 170496, "Adapter architecture differs")
    return model, head, device


class Runner:
    def __init__(self, rt, model, head, arm, shape, device):
        self.rt, self.model, self.head, self.arm = rt, model, head, arm
        self.shape, self.device = shape, device
        self.dis = rt.cv2.DISOpticalFlow_create(rt.cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.reset()

    def reset(self):
        self.prev = self.state = None

    def step(self, index, rec, debug=False):
        rt, model = self.rt, self.model
        # Never request GT inside a measured frame.
        frame = rt.exp1.load_frame(rec, self.shape, self.device, need_gt=False)
        key = index % K == 0
        if not key:
            require(self.prev is not None and self.state is not None, "Missing prior state")
            old_r, old_c, old_f, rays = self.state
            # Match each historical runner's autocast boundary exactly:
            # Y04 A/O builds static flow before autocast; native Y04-A P inside it.
            if self.arm != "P_multi":
                static = rt.exp1.transition_static(model, self.prev, frame, self.shape, self.dis)
        with rt.torch.autocast(device_type="cuda", dtype=rt.torch.float16, enabled=True):
            if key:
                rays, radius, confidence, features = rt.exp1.keyframe_state(model, frame, self.shape)
            else:
                if self.arm == "P_multi":
                    static = rt.exp1.transition_static(model, self.prev, frame, self.shape, self.dis)
                radius, confidence, features = rt.exp1.one_fast_step(model, self.prev, frame, static, old_r, old_c, old_f)
            self.state = radius, confidence, features, rays
            out_radius = radius
            if self.arm == "O_multi" and not key:
                if debug:
                    before = [x.detach().clone() for x in [radius,confidence,*features]]
                    reconstructed = readout_radius(rt, model, features)
                    require(float((reconstructed.float()-radius.float()).abs().max().item()) <= 1e-3,
                        "O native readout reconstruction differs")
                adapted = self.head(features)
                out_radius = readout_radius(rt, model, adapted)
                if debug:
                    require(all(float((x.float()-y.float()).abs().max().item()) == 0
                        for x,y in zip([radius,confidence,*features],before)), "O adapter mutated native recursive state")
                del adapted
            pred = rt.gt2.make_z_depth(out_radius, rays, self.shape["padded_shape"],
                self.shape["paddings"], model.interpolation_mode)
        self.prev = frame
        return pred.detach().float().cpu().numpy()


def check_output(rt, seq, arm, idx, rec, pred, ref, shape):
    if str(rec.get("gt_match_status", "")).upper() != "MATCHED":
        return None
    raw = rt.cv2.imread(str(rec["depth_path"]), rt.cv2.IMREAD_UNCHANGED)
    require(raw is not None and raw.dtype == rt.np.uint16 and tuple(raw.shape) == tuple(shape["original_shape"]),
        f"Invalid audit GT: {seq}/{idx}")
    gt = raw.astype(rt.np.float32) / rt.exp1.DEPTH_SCALE
    mask = (gt > 0) & rt.np.isfinite(gt)
    require(rt.np.all(rt.np.isfinite(pred[mask]) & (pred[mask] > 0)), "Invalid output on GT pixels")
    actual = rt.gt2.calculate_metrics(gt, pred)
    delta = max(abs(float(actual[k])-float(ref[k])) for k in ("AbsRel","RMSE","delta1"))
    require(delta <= 2e-5 and int(actual["valid_pixels"]) == int(ref["valid_pixels"]),
        f"Timed path changed quality: {seq}/{arm}/{idx}, max metric deviation={delta}")
    return {"sequence":seq,"arm":arm,"frame_index":idx,"max_metric_deviation":delta,
        "valid_pixels":int(actual["valid_pixels"]),"passed":True}


def run_trial(a, helper, rt, seq, manifest, arm, repeat, position, refs, p_state, o_state):
    model, head, device = make_model(rt, a, arm, p_state, o_state)
    shape = rt.exp1.shape_info_for(model, Path(manifest[0]["rgb_path"]), device)
    runner = Runner(rt, model, head, arm, shape, device)
    sync = lambda: rt.torch.cuda.synchronize(device)
    hook = None
    calls = {"n":0}
    if head is not None:
        def record_call(*_args): calls["n"] += 1
        hook = head.register_forward_hook(record_call)
    try:
        with rt.torch.no_grad():
            for idx in range(WARMUP):
                runner.step(idx, manifest[idx], debug=(idx == 1 and arm == "O_multi"))
            if head is not None:
                require(calls["n"] == WARMUP-WARMUP//K, "Warm-up repair/adapter call count differs")
                hook.remove(); hook = None
            require(tuple(model.pixel_decoder.radial_module.original_shapes) == tuple(shape["network_shape"]),
                "Readout grid is not network_shape")
            runner.reset()
            gc.collect(); sync()
            before = gpu_status(rt, a.nvidia_smi)
            check_busy(before)
            start = TAIL[seq] if a.smoke_only else 0
            stop = min(len(manifest), start+64) if a.smoke_only else len(manifest)
            indices = list(range(start, stop))
            samples = {start, start+1, start+K, TAIL[seq], TAIL[seq]+1, stop-1}
            latencies, checks = [], []
            for idx in indices:
                pred, ms = measured_call(lambda: runner.step(idx, manifest[idx]), sync)
                require(math.isfinite(ms) and ms > 0, "Invalid frame latency")
                latencies.append({"sequence":seq,"arm":arm,"repeat":repeat,"order_position":position,
                    "frame_index":idx,"is_keyframe":idx%K==0,"in_quality_tail":idx>=TAIL[seq],"latency_ms":ms})
                if idx in samples:
                    checked = check_output(rt, seq, arm, idx, manifest[idx], pred, refs[(seq,arm,idx)], shape)
                    if checked is not None:
                        checks.append({**checked,"repeat":repeat})
                del pred
            sync()
            after = gpu_status(rt, a.nvidia_smi)
            check_busy(after)
        require(head is None or not head.training, "Adapter unexpectedly entered train mode")
        tail_times = [r["latency_ms"] for r in latencies if r["in_quality_tail"]]
        all_times = [r["latency_ms"] for r in latencies]
        meta = {"sequence":seq,"arm":arm,"repeat":repeat,"order_position":position,
            "processed_rgb_frames":len(indices),"tail_rgb_frames":len(tail_times),
            "mean_ms":statistics.mean(all_times),"p95_ms":helper.percentile(all_times,95),
            "tail_mean_ms":statistics.mean(tail_times),"tail_p95_ms":helper.percentile(tail_times,95),
            **expected_counts(arm, indices), "counts_scope":"executed branches; warm-up hooks verified separately",
            "all_endpoint_checks_clean":telemetry_clean(before) and telemetry_clean(after),
            "gpu_before":before,"gpu_after":after,"output_audit_samples":len(checks)}
        return latencies, meta, checks
    finally:
        if hook is not None: hook.remove()
        del runner, model, head
        gc.collect(); rt.torch.cuda.empty_cache()


def build_summary(helper, trials, frames, quality):
    qmap = {(r["sequence"],r["arm"]):r for r in quality}
    result = []
    for seq in sorted({r["sequence"] for r in trials}):
        for arm in ARMS:
            selected = [r for r in trials if r["sequence"]==seq and r["arm"]==arm]
            pooled = [r["latency_ms"] for r in frames if r["sequence"]==seq and r["arm"]==arm and r["in_quality_tail"]]
            result.append({"sequence":seq,"arm":arm,"repeats":len(selected),
                "median_trial_tail_mean_ms":statistics.median(r["tail_mean_ms"] for r in selected),
                "min_trial_tail_mean_ms":min(r["tail_mean_ms"] for r in selected),
                "max_trial_tail_mean_ms":max(r["tail_mean_ms"] for r in selected),
                "pooled_tail_frame_p95_ms":helper.percentile(pooled,95),
                "median_trial_whole_stream_mean_ms":statistics.median(r["mean_ms"] for r in selected),
                "all_endpoint_checks_clean":all(r["all_endpoint_checks_clean"] for r in selected),
                **{f"existing_tail_{k}":qmap[seq,arm]["mean_"+k] for k in ("AbsRel","RMSE","delta1")}})
    lookup = {(r["sequence"],r["arm"]):r for r in result}
    for row in result:
        for arm in ("A","O_multi"):
            row[f"tail_latency_ratio_to_{arm}"] = row["median_trial_tail_mean_ms"] / lookup[row["sequence"],arm]["median_trial_tail_mean_ms"]
    return result


def archive(out, helper, pre):
    dest, tmp = out/"y06b_min_review.zip", out/"y06b_min_review.zip.tmp"
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for pattern in ("y06b_min_*.csv","y06b_min_*.json"):
            for path in sorted(out.glob(pattern)):
                if path.name != "y06b_min_failure.json":z.write(path,path.name)
        z.write(__file__,Path(__file__).name)
        for path in [ROOT/"experiments/p1/gt4a2_y04a_native_pre_readout_eval.py", *[ROOT/rel for rel in helper.SOURCE_SHA]]:
            z.write(path,"source/"+str(path.relative_to(ROOT)))
        for name in ("y04a_summary.json","y04a_preflight.json","y04a_sequence_summary.csv"):
            z.write(Path(pre["y04a_dir"])/name,"reference/"+name)
    tmp.replace(dest)
    return dest


def main(argv=None):
    a = args_parser(argv)
    if a.self_test:return self_test()
    helper = import_helper()
    out, ydir = Path(a.out_dir).resolve(), Path(a.y04_dir).resolve()
    resolved_a = find_y04a(helper, a.y04a_dir)
    require(out != resolved_a and resolved_a not in out.parents, "Output must be outside existing Y04-A results")
    a.y04a_dir = str(resolved_a)
    require(not (out/"y06b_min_summary.json").exists(), "Completed output exists; choose a new --out-dir")
    with helper.exclusive_output(out,ydir):
        require(not (out/"y06b_min_summary.json").exists(), "Completed output exists; choose a new --out-dir")
        try:
            pre, manifests, refs, quality, rt, p_state, o_state = preflight(a,helper)
            previous = out/"y06b_min_preflight.json"
            if previous.exists():require(helper.json_read(previous)==pre,"Output identity differs; use a new --out-dir")
            helper.json_write(previous,pre)
            helper.csv_write(out/"y06b_min_existing_quality.csv",quality)
            print(f"[PREFLIGHT_PASS] {VERSION}: A/O_multi/P_multi only, K32, warm-up64",flush=True)
            if a.preflight_only:
                print("No model inference or training performed.",flush=True);return 0
            require(rt.torch.cuda.is_available(),"CUDA is required")
            rt.torch.set_num_threads(4);rt.torch.manual_seed(20261026);rt.np.random.seed(20261026)
            rt.torch.cuda.manual_seed_all(20261026)
            print(f"[DEVICE] {rt.torch.cuda.get_device_name(0)}; torch={rt.torch.__version__}",flush=True)
            status = gpu_status(rt,a.nvidia_smi);check_busy(status)
            if not telemetry_clean(status):print("[GPU MONITOR UNVERIFIED] Timing can run; monitoring limitations will be recorded.",flush=True)
            frames,trials,checks = [],[],[]
            for seq in a.seqs:
                # Prime the same encoded RGB files once for all methods; no GT reads here.
                for rec in manifests[seq]:
                    with Path(rec["rgb_path"]).open("rb") as f:
                        for _ in iter(lambda:f.read(1024*1024),b""):pass
                for repeat,order in enumerate(pre["orders"][seq]):
                    for position,arm in enumerate(order):
                        print(f"[TIMING] {seq} repeat={repeat+1}/{pre['repeats']} arm={arm}",flush=True)
                        ff,meta,cc=run_trial(a,helper,rt,seq,manifests[seq],arm,repeat,position,refs,p_state,o_state)
                        frames.extend(ff);trials.append(meta);checks.extend(cc)
                        helper.csv_write(out/"progress"/f"{seq}_{repeat}_{arm}_latencies.csv",ff)
                        helper.json_write(out/"progress"/f"{seq}_{repeat}_{arm}_trial.json",meta)
                        print(f"[TRIAL] tail mean={meta['tail_mean_ms']:.3f} ms; tail P95={meta['tail_p95_ms']:.3f} ms",flush=True)
            for path,sha in pre["input_sha256"].items():require(helper.digest(path)==sha,f"Input changed during timing: {path}")
            require(helper.digest(__file__)==pre["script_sha256"],"Timing script changed during run")
            summary_rows=build_summary(helper,trials,frames,quality)
            helper.csv_write(out/"y06b_min_latency_frames.csv",frames)
            helper.csv_write(out/"y06b_min_output_checks.csv",checks)
            helper.csv_write(out/"y06b_min_cost_quality_summary.csv",summary_rows)
            helper.json_write(out/"y06b_min_trials.json",trials)
            clean=all(t["all_endpoint_checks_clean"] for t in trials)
            summary={"status":"SMOKE_PASS" if a.smoke_only else "TIMING_COMPLETE" if clean else "TIMING_COMPLETE_GPU_UNVERIFIED",
                "version":VERSION,"training_performed":False,"partial_smoke":a.smoke_only,
                "sequences":a.seqs,"full_three_sequence_evaluation":not a.smoke_only and set(a.seqs)==set(SEQUENCES),
                "arms":list(ARMS),"K":K,"repeats":pre["repeats"],"warmup_frames_per_trial":WARMUP,
                "trial_count":len(trials),"timed_rgb_records":len(frames),"sampled_output_checks":len(checks),
                "all_output_checks_passed":True,"all_endpoint_checks_clean":clean,
                "quality_source":"existing Y04/Y04-A scores; only sparse consistency checks rerun",
                "environment":{"python":sys.version,"torch":rt.torch.__version__,"GPU":rt.torch.cuda.get_device_name(0)},
                "script_sha256":helper.digest(__file__),"preflight_sha256":helper.digest(previous),
                "method_verdict":"NOT_AUTOMATICALLY_DECIDED; compare per-sequence quality, risk and cost",
                "limits":pre["limits"]}
            helper.json_write(out/"y06b_min_summary.json",summary)
            dest=archive(out,helper,pre)
            (out/"y06b_min_failure.json").unlink(missing_ok=True)
            for row in summary_rows:
                if row["arm"]=="P_multi":print(f"[RESULT] {row['sequence']}: P/A={row['tail_latency_ratio_to_A']:.3f}; P/O={row['tail_latency_ratio_to_O_multi']:.3f}",flush=True)
            print(f"[{summary['status']}] {dest}",flush=True)
            return 0
        except Exception as exc:
            (out/"y06b_min_summary.json").unlink(missing_ok=True)
            helper.json_write(out/"y06b_min_failure.json",{"status":"FAILED","error":str(exc),"traceback":traceback.format_exc(),"training_performed":False})
            raise


def self_test():
    import unittest
    from unittest.mock import Mock, patch

    class Checks(unittest.TestCase):
        @staticmethod
        def checkpoint_shapes():
            block = {"0.weight": (32,512,1,1), "1.weight": (32,), "1.bias": (32,),
                "3.weight": (32,32,3,3), "4.weight": (32,), "4.bias": (32,),
                "6.weight": (512,32,1,1), "6.bias": (512,)}
            return {f"blocks.{level}.{key}":SimpleNamespace(shape=shape)
                for level in range(4) for key,shape in block.items()}

        def test_checkpoint_512_channels_load_regression(self):
            # Reproduce the uploaded checkpoint's actual shapes without requiring torch.
            state = self.checkpoint_shapes()
            class Parameter:
                def numel(self):return 170496
                def requires_grad_(self, flag):self.frozen = not flag
            class Head:
                def __init__(self, channels, bottleneck, levels):
                    self.channels, self.bottleneck, self.levels = channels,bottleneck,levels
                    self.parameter=Parameter();self.training=True
                def load_state_dict(self, weights, strict):
                    require(strict and tuple(weights["blocks.0.0.weight"].shape) ==
                        (self.bottleneck,self.channels,1,1), "Checkpoint/model channel mismatch")
                def parameters(self):return [self.parameter]
                def to(self, device):self.device=device
                def eval(self):self.training=False
            model=SimpleNamespace(pixel_decoder=SimpleNamespace(radial_module=SimpleNamespace(hidden_dim=512)),
                parameters=lambda:[], eval=lambda:None)
            rt=SimpleNamespace(torch=SimpleNamespace(device=lambda value:value),
                gt2=SimpleNamespace(load_model=lambda *args:model))
            module=SimpleNamespace(PersistentFeatureRepairHead=Head)
            with patch.dict(sys.modules,{"velodepth.models.persistent_feature_repair":module}):
                _,head,device=make_model(rt,SimpleNamespace(snapshot="/snapshot"),"O_multi",state,state)
            self.assertEqual((head.channels,head.bottleneck,head.levels),(512,32,4))
            self.assertTrue(head.parameter.frozen)
            self.assertFalse(head.training)
            self.assertEqual(device,"cuda:0")

        def test_checkpoint_inconsistent_level_rejected(self):
            state=self.checkpoint_shapes()
            state["blocks.3.6.bias"]=SimpleNamespace(shape=(256,))
            with self.assertRaisesRegex(RuntimeError,"blocks.3.6.bias"):
                infer_head_spec(state)

        def test_balanced_order(self):
            for s in range(3):
                rows=orders(3,s)
                for arm in ARMS:self.assertEqual(sorted(row.index(arm) for row in rows),[0,1,2])

        def test_timer_completion_boundary(self):
            events=[]
            def sync():events.append("sync")
            def clock():events.append("clock");return 1.0 if events.count("clock")==1 else 1.25
            result,ms=measured_call(lambda:events.append("frame") or "pred",sync,clock)
            self.assertEqual(events,["sync","clock","frame","sync","clock"])
            self.assertEqual((result,ms),("pred",250.0))

        def test_full_and_smoke_counts(self):
            self.assertEqual(expected_counts("P_multi",list(range(895)))["repair_head_calls"],867)
            self.assertEqual(expected_counts("O_multi",list(range(768,832)))["additional_readout_passes"],62)
            self.assertEqual(expected_counts("A",list(range(64)))["keyframe_calls"],2)

        def test_monitor_unavailable_not_clean(self):
            with patch("shutil.which",return_value=None),patch.object(Path,"is_file",return_value=False):
                info=gpu_status(None,"auto")
            self.assertFalse(telemetry_clean(info))
            self.assertFalse(info["foreign_processes_confirmed"])

        def test_confirmed_busy_rejected(self):
            with self.assertRaises(RuntimeError):check_busy({"foreign_processes_confirmed":[{"pid":123}]})

        def test_O_keeps_native_state(self):
            rt=SimpleNamespace(torch=SimpleNamespace(float16=None,autocast=lambda **kw:contextlib.nullcontext()),
                cv2=SimpleNamespace(DISOPTICAL_FLOW_PRESET_FAST=1,DISOpticalFlow_create=lambda _:None))
            class Output:
                def detach(self):return self
                def float(self):return self
                def cpu(self):return self
                def numpy(self):return "adapted_output"
            step=Mock(return_value=("native_r1","native_c1",["native_f1"]))
            rt.exp1=SimpleNamespace(load_frame=lambda rec,shape,device,need_gt: {"gt_requested":need_gt},
                transition_static=lambda *args:"flow",one_fast_step=step)
            rt.gt2=SimpleNamespace(make_z_depth=lambda *args:Output())
            runner=Runner(rt,SimpleNamespace(interpolation_mode="bilinear"),lambda f:["adapted_f"],"O_multi",
                {"padded_shape":(2,2),"paddings":(0,0,0,0)},None)
            runner.prev="frame0";runner.state=("r0","c0",["f0"],"rays")
            with patch(__name__+".readout_radius",return_value="adapted_radius"):
                runner.step(1,{})
            self.assertEqual(runner.state,("native_r1","native_c1",["native_f1"],"rays"))
            self.assertFalse(runner.prev["gt_requested"])

    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    return 0 if result.wasSuccessful() else 1


if __name__=="__main__":
    try:raise SystemExit(main())
    except Exception:
        traceback.print_exc();raise SystemExit(1)
