#!/usr/bin/env python3
"""Fixed-schedule GEOM_ONLY commit enumeration; frozen VeloDepth, diagnostic only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import itertools
import json
import math
import random
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass, asdict, fields
from datetime import datetime, timezone
from pathlib import Path

COMMIT_FIELDS = ("features_teacher", "radius_teacher", "confidence_teacher", "rays_base")
HERE = Path(__file__).resolve().parent


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    names = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        w.writerows(rows)


def text_hash(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@dataclass
class Clocks:
    frame_idx: int = -1
    steps_since_kf_reset: int = 0
    steps_since_geom_commit: int = 0

    def advance(self, t, native=False, commit=False):
        if t != self.frame_idx + 1:
            raise RuntimeError("Absolute frame clock must advance exactly once per frame")
        self.frame_idx = t
        self.steps_since_kf_reset = 0 if native else self.steps_since_kf_reset + 1
        self.steps_since_geom_commit = 0 if native or commit else self.steps_since_geom_commit + 1


def finite_mean(values):
    values = [float(x) for x in values if math.isfinite(float(x))]
    return sum(values) / len(values) if values else float("nan")


def summarize(rows_by_plan, schedule, gamma=0.9, horizon=16):
    """Require the same metric mask; compute the exact best *evaluated plan*."""
    plans = sorted(rows_by_plan)
    reference = rows_by_plan[plans[0]]
    summaries = []
    for plan in plans:
        rows = rows_by_plan[plan]
        if len(rows) != len(reference):
            raise RuntimeError("Unequal trajectory lengths")
        for metric in ("absrel", "rmse", "delta1"):
            mask = [math.isfinite(float(r[metric])) for r in rows]
            if mask != [math.isfinite(float(r[metric])) for r in reference]:
                raise RuntimeError(f"Different valid metric frames for {plan}/{metric}; do not compare means")
        n = sum(math.isfinite(float(r["absrel"])) for r in rows)
        if not n:
            raise RuntimeError("No valid AbsRel frames")
        s = {
            "plan": plan, "frames": len(rows), "valid_absrel_frames": n,
            "K_requested": len(schedule),
            "refresh_candidates_optional": sum(int(r["optional_candidate_computed"]) for r in rows),
            "geom_commits_optional": sum(int(r["geom_commit"]) for r in rows),
            "native_restarts": sum(int(r["native_restart"]) for r in rows),
            "scheduled_native_collisions": sum(int(r["scheduled_native_collision"]) for r in rows),
            "mean_absrel": finite_mean(r["absrel"] for r in rows),
            "mean_rmse": finite_mean(r["rmse"] for r in rows),
            "mean_delta1": finite_mean(r["delta1"] for r in rows),
            "propagation_ms": sum(float(r["propagation_ms"]) for r in rows),
            "keyframe_ms": sum(float(r["keyframe_ms"]) for r in rows),
            "input_payload_clone_ms": sum(float(r.get("input_payload_clone_ms", 0)) for r in rows),
            "snapshot_restore_commit_ms": sum(float(r["snapshot_restore_commit_ms"]) for r in rows),
        }
        summaries.append(s)
    indexed = {r["plan"]: r for r in summaries}
    never, always = "0" * len(schedule), "1" * len(schedule)
    best = min(summaries, key=lambda r: (r["mean_absrel"], r["plan"]))
    base = min(indexed[never]["mean_absrel"], indexed[always]["mean_absrel"])
    diagnostic = {
        "best_plan_by_full_clip_absrel": best["plan"],
        "never_commit_absrel": indexed[never]["mean_absrel"],
        "always_commit_absrel": indexed[always]["mean_absrel"],
        "best_enumerated_absrel": best["mean_absrel"],
        "gain_over_better_unconditional_absrel": base - best["mean_absrel"],
        "relative_gain_over_better_unconditional": (base - best["mean_absrel"]) / base if base > 0 else None,
        "scope": "Exact minimum over evaluated binary commit plans at this fixed schedule, "
                 "with native restart priority. Uses full-clip GT; not deployable, not a general policy upper bound.",
        "same_optional_candidate_count": len({s["refresh_candidates_optional"] for s in summaries}) == 1,
        "same_native_restart_count": len({s["native_restarts"] for s in summaries}) == 1,
        "all_plans_have_K_optional_candidates": all(s["refresh_candidates_optional"] == len(schedule) for s in summaries),
        "timing_claim": "Instrumented diagnostic replay only; no end-to-end speedup or equal-total-cost claim.",
    }
    # A pair differs only at one bit. Its prefix is identical and future planned controls
    # are identical. Native restarts may differ as a downstream state effect.
    pairs = []
    for j, anchor in enumerate(schedule):
        for p0 in plans:
            if p0[j] != "0":
                continue
            p1 = p0[:j] + "1" + p0[j + 1:]
            if p1 not in rows_by_plan:
                continue
            a, b = rows_by_plan[p0], rows_by_plan[p1]
            for t in range(anchor + 1):
                for metric in ("absrel", "rmse", "delta1"):
                    x, y = float(a[t][metric]), float(b[t][metric])
                    if math.isfinite(x) and abs(x-y) > 1e-6:
                        raise RuntimeError(f"Paired plans disagree before/at intervention: {p0}/{p1}, frame={t}, metric={metric}")
            valid, total = 0, 0.0
            for k in range(1, horizon + 1):
                t = anchor + k
                if t >= len(a):
                    break
                x, y = float(a[t]["absrel"]), float(b[t]["absrel"])
                if math.isfinite(x) and math.isfinite(y):
                    total += gamma ** k * (x - y)
                    valid += 1
            pairs.append({
                "anchor": anchor, "discard_plan": p0, "commit_plan": p1,
                "same_future_planned_bits": p0[j+1:] == p1[j+1:],
                "H_requested": horizon, "future_pairs_valid": valid,
                "discounted_future_absrel_gain_sum": total if valid else float("nan"),
                "discounted_future_absrel_gain_per_valid_frame": total / valid if valid else float("nan"),
                "full_clip_mean_absrel_gain": indexed[p0]["mean_absrel"] - indexed[p1]["mean_absrel"],
                "anchor_native_collision": int(a[anchor]["scheduled_native_collision"]),
            })
    return summaries, diagnostic, pairs


def static_preflight(root):
    pins = json.loads((HERE / "audited_source_hashes.json").read_text(encoding="utf-8"))
    rows = []
    for rel, expected in pins.items():
        p = root / rel
        actual = text_hash(p) if p.is_file() else None
        rows.append({"path": rel, "expected_normalized_sha256": expected,
                     "actual_normalized_sha256": actual, "matches_audited_source": actual == expected})
    required = root / "experiments/p1/p1_multihorizon_counterfactual.py"
    rows.append({"path": str(required.relative_to(root)), "exists": required.is_file(),
                 "actual_normalized_sha256": text_hash(required) if required.is_file() else None,
                 "note": "Runtime helper contract checked separately; not present in historical source archive."})
    if not all(r.get("matches_audited_source", r.get("exists", False)) for r in rows):
        raise RuntimeError("Source differs from audited inputs or a helper is missing. Inspect source_audit.json "
                           "and send review ZIP for adaptation; no model inference has started.")
    return rows


class GPUExperiment:
    def __init__(self, root, args):
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / "experiments/p1"))
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        import numpy as np
        self.torch, self.np = torch, np
        self.h = importlib.import_module("p2_3_online_replay_v5_11_action_audit")
        self.cf = self.h.cf
        for name in ("read_index", "load_frames", "load_gt", "eval_output"):
            if not callable(getattr(self.cf, name, None)):
                raise RuntimeError(f"Missing counterfactual helper: {name}")
        if set(self.cf.COMMIT_FIELDS) != set(COMMIT_FIELDS):
            raise RuntimeError("COMMIT_FIELDS do not match the audited four-field action")
        self.ss = importlib.import_module("stream_state")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; select your existing svrdepth GPU environment")
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.seed(args.seed)
        rgb = self.cf.read_index(args.data_root, "rgb.txt", "rgb")[:args.frames]
        depth = self.cf.read_index(args.data_root, "depth.txt", "depth")
        if len(rgb) != args.frames:
            raise RuntimeError(f"Requested {args.frames} frames; found {len(rgb)}")
        self.records = self.h.match_gt_tolerant(rgb, depth, args.max_time_diff)
        self.frames = self.cf.load_frames(self.records)
        self.gt = [self.cf.load_gt(r["gt_path"], args.gt_scale) if r["gt_available"] else None
                   for r in self.records]
        if not any(g is not None for g in self.gt):
            raise RuntimeError("No GT matches; check data-root and timestamp tolerance")
        print("Loading cached, frozen VeloDepth weights; online downloads disabled.", flush=True)
        self.model = self.h.VeloDepth.from_pretrained("lpiccinelli/velodepth")
        self.model.to("cuda")
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.resolution_level = args.resolution_level
        self.model.interpolation_mode = "bilinear"
        start = time.perf_counter()
        self.inputs, metas = self.h.capture_packed_inputs_without_full_inference(self.model, self.frames)
        self.sync()
        self.preprocess_flow_capture_ms = (time.perf_counter() - start) * 1000
        self.T = int(metas[0]["T"])
        if self.T != args.frames:
            raise RuntimeError("Packed frame count mismatch")
        height, width = map(int, self.frames.shape[-2:])
        self.paddings, self.padded_shape = self.h.get_paddings(
            (height, width), self.model.shape_constraints["ratio_bounds"])
        if "rays" in self.inputs:
            self.model.pixel_decoder.test_gt_camera = True
        self.base = self.h.capture_state(self.model, self.h.StreamState())
        self.args = args
        self.stream = self.h.VeloDepthStream(self.model)

    def seed(self, seed):
        random.seed(seed)
        self.np.random.seed(seed)
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)

    def sync(self):
        self.torch.cuda.synchronize()

    def timed(self, fn):
        self.sync()
        start = time.perf_counter()
        result = fn()
        self.sync()
        return result, (time.perf_counter() - start) * 1000

    def capture(self, state):
        return self.h.capture_state(self.model, state)

    def restore(self, snapshot):
        state = self.h.StreamState()
        self.h.restore_state(self.model, state, snapshot)
        return state

    def diff(self, a, b):
        if isinstance(a, self.torch.Tensor) and isinstance(b, self.torch.Tensor):
            if a.shape != b.shape:
                return float("inf")
            if not a.numel():
                return 0.0
            d = (a.detach().float() - b.detach().float()).abs()
            return float(d.max().item()) if bool(self.torch.isfinite(d).all()) else float("inf")
        if type(a) is not type(b):
            return float("inf")
        if isinstance(a, (tuple, list)):
            return max((self.diff(x, y) for x, y in zip(a, b)), default=0.0) if len(a) == len(b) else float("inf")
        if isinstance(a, dict):
            return max((self.diff(a[k], b[k]) for k in a), default=0.0) if a.keys() == b.keys() else float("inf")
        return 0.0 if a == b else float("inf")

    def tensors(self, x):
        if isinstance(x, self.torch.Tensor):
            yield x
        elif isinstance(x, (tuple, list)):
            for value in x:
                yield from self.tensors(value)
        elif isinstance(x, dict):
            for value in x.values():
                yield from self.tensors(value)

    def call_stream(self, method, state, t):
        # Match the historical harness: isolate any in-place mutation of packed
        # inputs by downstream model helpers, not only the explicit StreamState.
        payload, clone_ms = self.timed(lambda: self.h.clone_payload(self.inputs))
        result, step_ms = self.timed(lambda: getattr(self.stream, method)(payload, t, state, self.T))
        return result, step_ms, clone_ms

    def prepare_step(self, state, t, scheduled):
        pre, snapshot_ms = self.timed(lambda: self.capture(state)) if scheduled else (None, 0.0)
        if t == 0:
            (out, state), kf_ms, clone_ms = self.call_stream("keyframe_step", state, t)
            return out, state, {"native_restart": 0, "native_initialization": 1,
                               "optional_candidate_computed": 0, "scheduled_native_collision": 0,
                               "propagation_ms": 0.0, "keyframe_ms": kf_ms,
                               "input_payload_clone_ms": clone_ms,
                               "snapshot_restore_commit_ms": snapshot_ms}, None
        (out, state, restart), prop_ms, clone_ms = self.call_stream("propagate_step", state, t)
        event = {"native_restart": int(restart), "native_initialization": 0,
                 "optional_candidate_computed": 0, "scheduled_native_collision": int(scheduled and restart),
                 "propagation_ms": prop_ms, "keyframe_ms": 0.0,
                 "input_payload_clone_ms": clone_ms,
                 "snapshot_restore_commit_ms": snapshot_ms}
        if restart:
            (out, state), event["keyframe_ms"], clone_ms = self.call_stream("keyframe_step", state, t)
            event["input_payload_clone_ms"] += clone_ms
            return out, state, event, None
        if not scheduled:
            return out, state, event, None
        post, ms = self.timed(lambda: self.capture(state))
        event["snapshot_restore_commit_ms"] += ms
        candidate_state, ms = self.timed(lambda: self.restore(pre))
        event["snapshot_restore_commit_ms"] += ms
        (refresh_out, candidate_state), event["keyframe_ms"], clone_ms = self.call_stream("keyframe_step", candidate_state, t)
        event["input_payload_clone_ms"] += clone_ms
        state, ms = self.timed(lambda: self.restore(post))
        event["snapshot_restore_commit_ms"] += ms
        event["optional_candidate_computed"] = 1
        # Retain native overlap semantics. Do NOT force first_keyframe=True.
        return refresh_out, state, event, candidate_state

    def commit(self, state, candidate):
        before = {f.name: getattr(state, f.name) for f in fields(state) if f.name not in COMMIT_FIELDS}
        for name in COMMIT_FIELDS:
            value = getattr(candidate, name)
            if value is None:
                raise RuntimeError(f"Candidate field is None: {name}")
            setattr(state, name, self.h.clone_payload(value))
        if any(self.diff(getattr(state, n), v) != 0 for n, v in before.items()):
            raise RuntimeError("GEOM_ONLY changed scheduler or fast-tail fields")

    def r0_audit(self, schedule):
        self.seed(self.args.seed)
        state = self.restore(self.base)
        for t in range(self.T):
            out, state, event, candidate = self.prepare_step(state, t, t in schedule)
            if candidate is None or t + 1 >= self.T:
                continue
            post = self.capture(state)
            discard = self.restore(post)
            commit = self.restore(post)
            self.commit(commit, candidate)
            # Check actual independent storage, including fast-tail fields.
            a = {x.untyped_storage().data_ptr() for f in fields(discard)
                 for x in self.tensors(getattr(discard, f.name)) if x.numel()}
            b = {x.untyped_storage().data_ptr() for f in fields(commit)
                 for x in self.tensors(getattr(commit, f.name)) if x.numel()}
            if a & b:
                raise RuntimeError("Commit/discard state tensors share storage")
            diffs = {n: self.diff(getattr(discard, n), getattr(commit, n)) for n in COMMIT_FIELDS}
            runtime0, runtime1 = self.capture(discard), self.capture(commit)
            # Repeated restoration of the SAME branch must produce the same next output.
            left, right = [], []
            for target, snapshot in ((left, runtime0), (left, runtime0), (right, runtime1)):
                st = self.restore(snapshot)
                result, st, ev, _ = self.prepare_step(st, t + 1, False)
                target.append((self.h.clone_payload(result.radius), self.h.clone_payload(result.rays),
                               self.h.clone_payload(result.confidence), ev["native_restart"]))
            repeat = self.diff(left[0], left[1])
            if repeat > self.args.tolerance:
                raise RuntimeError(f"Same-state replay differs: {repeat} > {self.args.tolerance}")
            result = {
                "status": "passed", "anchor": t,
                "same_current_refresh_output_max_abs_diff": 0.0,
                "same_current_output_basis": "Both hypothetical actions reference the single computed refresh output.",
                "shared_state_tensor_storage_count": len(a & b),
                "commit_field_max_abs_diffs": diffs,
                "same_branch_next_output_repeat_max_abs_diff": repeat,
                "commit_vs_discard_next_output_max_abs_diff": self.diff(left[0][:3], right[0][:3]),
                "next_native_restart_discard": left[0][3], "next_native_restart_commit": right[0][3],
                "scope": "Selected StreamState and audited decoder runtime attributes; not a proof about every possible model cache.",
            }
            self.restore(self.base)
            return result
        raise RuntimeError("R0 inconclusive: all scheduled slots were mandatory restarts or lacked a next frame")

    def run_plan(self, plan, schedule):
        self.seed(self.args.seed)
        state = self.restore(self.base)
        clock = Clocks()
        slots = dict(zip(schedule, map(int, plan)))
        rows = []
        self.sync()
        start = time.perf_counter()
        last_kf, last_geom = 0, 0
        for t in range(self.T):
            out, state, ev, candidate = self.prepare_step(state, t, t in slots)
            did_commit = bool(candidate is not None and slots[t])
            if did_commit:
                _, ms = self.timed(lambda: self.commit(state, candidate))
                ev["snapshot_restore_commit_ms"] += ms
            native = bool(t == 0 or ev["native_restart"])
            if native:
                last_kf = last_geom = t
            elif did_commit:
                last_geom = t
            clock.advance(t, native=native, commit=did_commit)
            if clock.steps_since_kf_reset != t - last_kf or clock.steps_since_geom_commit != t - last_geom:
                raise RuntimeError("Clock transition mismatch")
            metric = self.cf.eval_output(out, self.model, self.paddings, self.padded_shape, self.gt[t])
            rows.append({
                "plan": plan, "frame": t, "scheduled": int(t in slots),
                "planned_commit_bit": slots.get(t, ""), "geom_commit": int(did_commit),
                "output_source": "REFRESH" if candidate is not None or native else "PROP",
                "kf_reset": int(native),
                "scheduler_distance_kf": state.distance_kf,
                "gt_available": int(self.records[t]["gt_available"]),
                **asdict(clock), **ev,
                **{m: float(metric.get(m, float("nan"))) for m in ("absrel", "rmse", "delta1")},
            })
            del candidate, out
        self.sync()
        elapsed = (time.perf_counter() - start) * 1000
        return rows, elapsed


def package_review(root, out, files):
    review = out.parent / (out.name + "_review.zip")
    if review.exists():
        raise FileExistsError(review)
    with zipfile.ZipFile(review, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, "results/" + p.relative_to(out).as_posix())
        for p in sorted(HERE.glob("*")):
            if p.is_file() and p.suffix in (".py", ".json", ".md", ".sh"):
                z.write(p, "runner/" + p.name)
        for rel in files:
            p = root / rel
            if p.is_file():
                z.write(p, "project_sources/" + rel)
        if z.testzip() is not None:
            raise RuntimeError("Review ZIP integrity failure")
    print(f"Review ZIP: {review}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", type=Path, default=HERE.parents[2])
    ap.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/datasets/rgbd_bonn_crowd2"))
    ap.add_argument("--sequence", default="crowd2")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--refresh-frames", default="5,13,21")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--resolution-level", type=int, default=0)
    ap.add_argument("--gt-scale", type=float, default=5000.0)
    ap.add_argument("--max-time-diff", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    args = ap.parse_args()
    root = args.project_root.resolve()
    schedule = [int(x) for x in args.refresh_frames.split(",")]
    if len(schedule) != 3 or schedule != sorted(set(schedule)) or schedule[0] < 1 or schedule[-1] >= args.frames - 1:
        ap.error("Exactly 3 increasing refresh frames are required, all in [1, frames-2]")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = args.out_dir or root / "outputs/r_series/r2a_commit_enum" / f"{args.sequence}_{args.frames}_{stamp}"
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        ap.error(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    pins = json.loads((HERE / "audited_source_hashes.json").read_text(encoding="utf-8"))
    source_files = list(pins) + ["experiments/p1/p1_multihorizon_counterfactual.py",
                               "velodepth/models/velodepth.py", "velodepth/models/decoder_velo.py"]
    status = {"experiment": "R0 + R2-A/K3 fixed-schedule GEOM_ONLY enumeration",
              "sequence": args.sequence, "frames": args.frames, "refresh_frames_zero_based": schedule,
              "weights_frozen": True, "predictor_training": False, "ground_truth_used_for_diagnostics": True,
              "preflight_only": args.preflight_only, "status": "started",
              "commit_fields": list(COMMIT_FIELDS), "runner_version": "1.0"}
    exit_code = 0
    try:
        audit = []
        for rel, expected in pins.items():
            p = root / rel
            actual = text_hash(p) if p.is_file() else None
            audit.append({"path": rel, "expected_normalized_sha256": expected,
                          "actual_normalized_sha256": actual, "matches_audited_source": actual == expected})
        write_json(out / "source_audit.json", audit)
        static_preflight(root)
        status["status"] = "preflight_passed"
        if not args.preflight_only:
            exp = GPUExperiment(root, args)
            with exp.torch.inference_mode():
                # R0 also warms up both candidate and propagation paths; its work is
                # excluded from each instrumented trajectory wall time.
                write_json(out / "r0_runtime_audit.json", exp.r0_audit(schedule))
                all_rows, walls = {}, {}
                for bits in itertools.product("01", repeat=3):
                    plan = "".join(bits)
                    print(f"Running plan {plan}; scheduled frames={schedule}", flush=True)
                    rows, wall = exp.run_plan(plan, schedule)
                    all_rows[plan], walls[plan] = rows, wall
                    write_csv(out / f"frame_metrics_{plan}.csv", rows)
                sums, diagnostic, pairs = summarize(all_rows, schedule)
                for r in sums:
                    r["instrumented_trajectory_wall_ms_including_metrics"] = walls[r["plan"]]
                write_csv(out / "trajectory_summary.csv", sums)
                write_csv(out / "paired_commit_effects.csv", pairs)
                write_json(out / "diagnostic_headroom.json", diagnostic)
                status.update({"status": "completed", "gpu": exp.torch.cuda.get_device_name(0),
                               "torch_version": str(exp.torch.__version__),
                               "gt_matches": sum(int(r["gt_available"]) for r in exp.records),
                               "preprocess_flow_capture_ms_separate": exp.preprocess_flow_capture_ms,
                               "native_restart_policy": "Each trajectory executes its own native restart on the same frame; mandatory restart takes priority over elective commit.",
                               "future_controls": "All 3 fixed-schedule bits executed on the actual evolving state; paired plans differ at one bit only.",
                               "candidate_semantics": "Native keyframe_step from pre-propagation snapshot, including its existing overlap semantics; output used by both commit/discard.",
                               "timing_scope": "Diagnostic replay; shared offline input/flow construction; not online end-to-end latency.",
                               "runtime_helper_source_sha256": text_hash(root / "experiments/p1/p1_multihorizon_counterfactual.py")})
    except Exception:
        status["status"] = "failed"
        (out / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(traceback.format_exc(), file=sys.stderr)
        exit_code = 1
    finally:
        write_json(out / "metadata.json", status)
        package_review(root, out, source_files)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
