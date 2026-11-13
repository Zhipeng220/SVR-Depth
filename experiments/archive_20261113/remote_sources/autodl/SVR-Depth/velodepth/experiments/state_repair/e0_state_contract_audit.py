#!/usr/bin/env python3
"""E0: real-model state-contract and fast-readout replay audit for SVR-Depth.

This is a diagnostic only: no training, no weight changes, no local repair,
and no historical outputs are overwritten. It reuses the existing P1 stream
adapter and state snapshot helpers. The model path is based on the 2026-10-21
handoff snapshot and must be checked against the active checkout before use.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import fields, is_dataclass

import torch

ROOT = Path(__file__).resolve().parents[2]
P1 = ROOT / "experiments" / "p1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(P1))

from velodepth.models.velodepth import VeloDepth, _postprocess, get_paddings
from stream_state import StreamState, clone_state
from stream_state_snapshot import capture_state, restore_state
from velodepth_stream import VeloDepthStream
from test_action_branch_isolation_gpu import clone_payload, load_rgb


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def json_safe(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return repr(x)


def save_json(path: Path, obj):
    path.write_text(json.dumps(json_safe(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def tensor_schema(obj):
    if isinstance(obj, torch.Tensor):
        return {
            "type": "Tensor", "shape": list(obj.shape), "dtype": str(obj.dtype),
            "device": str(obj.device), "numel": obj.numel(),
            "finite": bool(torch.isfinite(obj).all().item()) if obj.is_floating_point() else True,
        }
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: tensor_schema(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): tensor_schema(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [tensor_schema(v) for v in obj]
    return {"type": type(obj).__name__, "value": json_safe(obj)}


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def max_abs_diff(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if tuple(a.shape) != tuple(b.shape):
            return float("inf")
        if a.numel() == 0:
            return 0.0
        d = (a.detach().float() - b.detach().float()).abs()
        if not bool(torch.isfinite(d).all()):
            return float("inf")
        return float(d.max().item())
    if type(a) is not type(b):
        return float("inf")
    if is_dataclass(a) and not isinstance(a, type):
        return max((max_abs_diff(getattr(a, f.name), getattr(b, f.name)) for f in fields(a)), default=0.0)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return float("inf")
        return max((max_abs_diff(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return float("inf")
        return max((max_abs_diff(a[k], b[k]) for k in a), default=0.0)
    return 0.0 if a == b else float("inf")



def storage_ptrs(obj, acc=None):
    if acc is None:
        acc = set()
    if isinstance(obj, torch.Tensor):
        if obj.numel() > 0:
            acc.add(obj.untyped_storage().data_ptr())
    elif is_dataclass(obj) and not isinstance(obj, type):
        for f in fields(obj):
            storage_ptrs(getattr(obj, f.name), acc)
    elif isinstance(obj, dict):
        for v in obj.values():
            storage_ptrs(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            storage_ptrs(v, acc)
    return acc


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rgb", default="/root/autodl-tmp/datasets/rgbd_bonn_crowd2/rgb")
    ap.add_argument("--frames", type=int, default=3, help="Smoke only; minimum 3 frames")
    ap.add_argument("--resolution_level", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=1e-4, help="Original-vs-stream adapter tolerance")
    ap.add_argument("--state-tolerance", type=float, default=1e-6, help="Same-snapshot replay/readout tolerance")
    ap.add_argument("--output-root", default="outputs/state_repair")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    if args.frames < 3:
        ap.error("--frames must be >= 3")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"e0_{stamp}"
    out_dir = (ROOT / args.output_root / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)

    meta = {
        "experiment": "E0_state_contract_audit",
        "run_id": run_id,
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "git_head": git_head(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "rgb_dir": str(Path(args.rgb).resolve()),
        "frames_requested": args.frames,
        "resolution_level": args.resolution_level,
        "tolerance": args.tolerance,
        "state_tolerance": args.state_tolerance,
        "source_sha256": {
            p: sha256_file(ROOT / p) for p in [
                "velodepth/models/velodepth.py",
                "velodepth/models/decoder_velo.py",
                "experiments/p1/stream_state.py",
                "experiments/p1/stream_state_snapshot.py",
                "experiments/p1/velodepth_stream.py",
                "experiments/p1/test_action_branch_isolation_gpu.py",
            ]
        },
    }
    save_json(out_dir / "run_metadata.json", meta)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "E0 real-model audit requires CUDA. No diagnostic or package changes were made; "
                "if this occurs despite an active GPU instance, stop here and retain this run folder."
            )
        torch.set_grad_enabled(False)
        model = VeloDepth.from_pretrained("lpiccinelli/velodepth")
        model.to("cuda")
        model.eval()
        model.resolution_level = args.resolution_level
        model.interpolation_mode = "bilinear"

        frames = load_rgb(args.rgb, args.frames)
        captured = {}
        original_encode_decode_test = model.encode_decode_test

        def capture_inputs(inputs, image_metas):
            captured["inputs"] = clone_payload(inputs)
            captured["metas"] = clone_payload(image_metas)
            result = original_encode_decode_test(inputs, image_metas)
            captured["original_internal"] = clone_payload(result[1])
            return result

        model.encode_decode_test = capture_inputs
        try:
            with torch.no_grad():
                official = model.infer(frames, normalize=True)
        finally:
            model.encode_decode_test = original_encode_decode_test

        if "inputs" not in captured or "original_internal" not in captured:
            raise RuntimeError("Could not capture packed inputs and original internal outputs from model.infer().")

        # Gate 0a: compare the existing streaming adapter against native infer()
        # using the exact packed inputs captured at encode_decode_test().
        equivalence_stream = VeloDepthStream(model)
        with torch.no_grad():
            alternative = equivalence_stream.run(
                clone_payload(captured["inputs"]), clone_payload(captured["metas"])
            )
        original_internal = captured["original_internal"]
        stream_rows = []
        stream_pass = []
        for key in ("radius", "rays", "confidence", "points"):
            diff = max_abs_diff(original_internal[key], alternative[key])
            ok = diff <= args.tolerance
            stream_pass.append(ok)
            stream_rows.append({"check": f"native_vs_stream_{key}", "max_abs_diff": diff,
                                "tolerance": args.tolerance, "passed": ok})
            print(f"{'PASS' if ok else 'FAIL'} native_vs_stream_{key}: max_abs_diff={diff:.9g}; tol={args.tolerance:g}")

        H, W = frames.shape[-2:]
        paddings, padded_hw = get_paddings((H, W), model.shape_constraints["ratio_bounds"])
        stream_points = _postprocess(
            alternative["points"], padded_hw, paddings=paddings,
            interpolation_mode=model.interpolation_mode
        )
        stream_depth = stream_points[:, -1:]
        depth_diff = max_abs_diff(official["depth"], stream_depth)
        depth_ok = depth_diff <= args.tolerance
        stream_pass.append(depth_ok)
        stream_rows.append({"check": "native_vs_stream_final_depth", "max_abs_diff": depth_diff,
                            "tolerance": args.tolerance, "passed": depth_ok})
        print(f"{'PASS' if depth_ok else 'FAIL'} native_vs_stream_final_depth: max_abs_diff={depth_diff:.9g}; tol={args.tolerance:g}")
        write_csv(out_dir / "e0_stream_equivalence.csv", stream_rows)
        save_json(out_dir / "e0_stream_equivalence_trace.json", {
            "trace": equivalence_stream.trace, "all_checks_passed": all(stream_pass),
            "checks": stream_rows,
            "note": "This file is written to this run folder; historical outputs/p1_trace/stream_equivalence.json is not touched."
        })
        if not all(stream_pass):
            raise AssertionError("Native/stream equivalence failed. Stop before state-contract experiments; inspect e0_stream_equivalence.csv.")

        inputs = clone_payload(captured["inputs"])
        metas = clone_payload(captured["metas"])
        B, T = metas[0]["B"], metas[0]["T"]
        if B != 1 or T < 3:
            raise RuntimeError(f"Expected B=1 and T>=3, got B={B}, T={T}")

        model._set_image_paddings(inputs)
        inputs["flow_bwd"], inputs["grid_sample"] = model._normalize_and_grid_from_flow(
            inputs["flow_bwd"], inputs["image"], inputs["image_paddings"], inputs["depth_paddings"]
        )
        old_camera_flag = model.pixel_decoder.test_gt_camera
        if "rays" in inputs:
            model.pixel_decoder.test_gt_camera = True

        stream = VeloDepthStream(model)
        state0 = StreamState()
        call_counts = {"forward_keyframe": 0, "forward_fast": 0}
        call_times = {"forward_keyframe_s": [], "forward_fast_s": []}
        fast_capture = {}
        original_keyframe = model.pixel_decoder.forward_keyframe
        original_fast = model.pixel_decoder.forward_fast

        def wrapped_keyframe(*a, **kw):
            call_counts["forward_keyframe"] += 1
            sync_cuda()
            t0 = time.perf_counter()
            try:
                return original_keyframe(*a, **kw)
            finally:
                sync_cuda()
                call_times["forward_keyframe_s"].append(time.perf_counter() - t0)

        def wrapped_fast(fast_inputs, fast_metas):
            call_counts["forward_fast"] += 1
            if "inputs" not in fast_capture:
                fast_capture["inputs"] = clone_payload(fast_inputs)
                fast_capture["metas"] = clone_payload(fast_metas)
            sync_cuda()
            t0 = time.perf_counter()
            result = original_fast(fast_inputs, fast_metas)
            sync_cuda()
            elapsed = time.perf_counter() - t0
            call_times["forward_fast_s"].append(elapsed)
            if "output" not in fast_capture:
                fast_capture["output"] = clone_payload(result)
            return result

        model.pixel_decoder.forward_keyframe = wrapped_keyframe
        model.pixel_decoder.forward_fast = wrapped_fast
        try:
            with torch.no_grad():
                _, state0 = stream.keyframe_step(inputs, 0, state0, T)
                snapshot0 = capture_state(model, state0)
                snapshot_reference = clone_state(snapshot0.stream_state)

                # Replay the same frame-1 propagation twice from the exact same snapshot.
                def run_branch():
                    branch_state = clone_state(snapshot0.stream_state)
                    restore_state(model, branch_state, snapshot0)
                    out, branch_state, restart = stream.propagate_step(inputs, 1, branch_state, T)
                    return clone_payload(out), clone_state(branch_state), bool(restart)

                out_a, state_a, restart_a = run_branch()
                out_b, state_b, restart_b = run_branch()
        finally:
            model.pixel_decoder.forward_keyframe = original_keyframe
            model.pixel_decoder.forward_fast = original_fast
            model.pixel_decoder.test_gt_camera = old_camera_flag

        if "inputs" not in fast_capture or "output" not in fast_capture:
            raise RuntimeError("forward_fast was not called/captured during propagation.")

        # Re-run the exact native fast readout with the captured same inputs and runtime shapes.
        scratch_state = clone_state(snapshot0.stream_state)
        restore_state(model, scratch_state, snapshot0)
        with torch.no_grad():
            sync_cuda()
            t0 = time.perf_counter()
            replay_output = original_fast(
                clone_payload(fast_capture["inputs"]), clone_payload(fast_capture["metas"])
            )
            sync_cuda()
            replay_seconds = time.perf_counter() - t0

        rows = []
        def add_metric(name, diff, detail=""):
            passed = bool(diff <= args.state_tolerance)
            rows.append({"check": name, "max_abs_diff": diff, "tolerance": args.state_tolerance,
                         "passed": passed, "detail": detail})
            print(f"{'PASS' if passed else 'FAIL'} {name}: max_abs_diff={diff:.9g}; tol={args.state_tolerance:g}")
            return passed

        passed_checks = []
        passed_checks.append(add_metric("same_snapshot_output_replay", max_abs_diff(out_a, out_b),
                                        "Two propagation runs restored from the same snapshot"))
        passed_checks.append(add_metric("same_snapshot_state_replay", max_abs_diff(state_a, state_b),
                                        "All StreamState fields, including scheduler fields"))
        passed_checks.append(add_metric("restart_decision_replay", 0.0 if restart_a == restart_b else float("inf"),
                                        f"branch_a={restart_a}, branch_b={restart_b}"))
        passed_checks.append(add_metric("native_fast_readout_replay", max_abs_diff(fast_capture["output"], replay_output),
                                        "Same captured fast inputs, same decoder runtime shapes"))
        passed_checks.append(add_metric("state_radius_matches_fast_output", max_abs_diff(state_a.radius_teacher, fast_capture["output"][0]),
                                        "First post-keyframe fast step; no overlap blending expected"))
        passed_checks.append(add_metric("state_confidence_matches_fast_output", max_abs_diff(state_a.confidence_teacher, fast_capture["output"][1]),
                                        "First post-keyframe fast step; no overlap blending expected"))
        passed_checks.append(add_metric("state_features_match_fast_output", max_abs_diff(state_a.features_teacher, fast_capture["output"][2]),
                                        "Persisted features_teacher should be returned feature_fast"))
        passed_checks.append(add_metric("snapshot_unchanged_after_branches", max_abs_diff(snapshot0.stream_state, snapshot_reference),
                                        "Snapshot must remain immutable across restored branches"))

        ptr_a = storage_ptrs(state_a)
        ptr_b = storage_ptrs(state_b)
        ptr_s = storage_ptrs(snapshot0.stream_state)
        alias_a_b = len(ptr_a & ptr_b)
        alias_a_s = len(ptr_a & ptr_s)
        alias_b_s = len(ptr_b & ptr_s)
        alias_ok = (alias_a_b == 0 and alias_a_s == 0 and alias_b_s == 0)
        passed_checks.append(add_metric("branch_storage_isolation", 0.0 if alias_ok else float("inf"),
                                        f"shared_storage_ptrs A/B={alias_a_b}, A/snapshot={alias_a_s}, B/snapshot={alias_b_s}"))

        # Mutation probe: modifying branch A must not mutate branch B or the snapshot.
        state_a_before_mutation = clone_state(state_a)
        state_b_before_mutation = clone_state(state_b)
        snapshot_before_mutation = clone_state(snapshot0.stream_state)
        mutation_probe_possible = bool(state_a.features_teacher and state_a.features_teacher[0].numel() > 0)
        if mutation_probe_possible:
            state_a.features_teacher[0].view(-1)[0].add_(1.0)
            mut_b_diff = max_abs_diff(state_b, state_b_before_mutation)
            mut_s_diff = max_abs_diff(snapshot0.stream_state, snapshot_before_mutation)
            passed_checks.append(add_metric("branch_a_mutation_does_not_touch_branch_b", mut_b_diff,
                                            "Intentional +1 mutation to branch A first feature element"))
            passed_checks.append(add_metric("branch_a_mutation_does_not_touch_snapshot", mut_s_diff,
                                            "Snapshot must not alias branch A storage"))
        else:
            passed_checks.append(add_metric("mutation_probe_available", float("inf"), "No feature tensor available"))

        state_contract = {
            "initial_state_after_keyframe_0": tensor_schema(snapshot0.stream_state),
            "post_propagation_branch_a_before_mutation_probe": tensor_schema(state_a_before_mutation),
            "captured_forward_fast_inputs": tensor_schema(fast_capture["inputs"]),
            "captured_forward_fast_output": tensor_schema(fast_capture["output"]),
            "note": "Current forward_fast is not a pure neck-only decoder: it consumes neck features plus residual features, refined flow and flow features, and returns radius/confidence plus propagated features.",
        }
        save_json(out_dir / "e0_state_contract.json", state_contract)
        write_csv(out_dir / "e0_readout_equivalence.csv", rows)
        save_json(out_dir / "e0_runtime_audit.json", {
            "call_counts_during_keyframe_and_two_branches": call_counts,
            "call_times_seconds": call_times,
            "fast_readout_replay_seconds": replay_seconds,
            "restart_branch_a": restart_a,
            "restart_branch_b": restart_b,
            "storage_alias_counts": {"A_B": alias_a_b, "A_snapshot": alias_a_s, "B_snapshot": alias_b_s},
            "all_checks_passed": all(passed_checks),
            "check_count": len(passed_checks),
            "failed_checks": [r["check"] for r in rows if not r["passed"]],
        })

        meta.update({
            "status": "passed" if all(passed_checks) else "failed_checks",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "captured_input_shape": tensor_schema(inputs.get("image")),
            "frames_captured": T,
            "run_dir": str(out_dir),
        })
        save_json(out_dir / "run_metadata.json", meta)

        summary = [
            "# E0 State Contract Audit",
            "",
            f"- Run ID: `{run_id}`",
            f"- Status: **{meta['status']}**",
            f"- Git HEAD: `{meta.get('git_head')}`",
            f"- RGB: `{Path(args.rgb).resolve()}`",
            f"- Frames: {args.frames} requested; packed T={T}",
            f"- Native/stream tolerance: `{args.tolerance}`",
            f"- State replay tolerance: `{args.state_tolerance}`",
            f"- Passed checks: {sum(passed_checks)}/{len(passed_checks)}",
            "",
            "## Interpretation boundary",
            "This audit checks deterministic replay, fast-readout replay, persisted-state consistency and branch isolation. It does not validate a local repair operator, GT depth improvement, future-horizon benefit, or real compute savings.",
            "",
            "## Files",
            "- `run_metadata.json`",
            "- `e0_state_contract.json`",
            "- `e0_stream_equivalence.csv` and `e0_stream_equivalence_trace.json`",
            "- `e0_readout_equivalence.csv`",
            "- `e0_runtime_audit.json`",
        ]
        (out_dir / "REPORT.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
        print(f"\nE0 output directory: {out_dir}")
        print(f"E0 status: {meta['status']}")
        if not all(passed_checks):
            raise AssertionError(f"E0 checks failed: {[r['check'] for r in rows if not r['passed']]}")

    except Exception as exc:
        meta.update({
            "status": "error",
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "error": repr(exc),
        })
        save_json(out_dir / "run_metadata.json", meta)
        (out_dir / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"E0 FAILED. Preserve output directory: {out_dir}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
