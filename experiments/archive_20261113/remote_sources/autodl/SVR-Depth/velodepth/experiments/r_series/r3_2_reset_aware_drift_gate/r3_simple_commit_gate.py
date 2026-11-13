#!/usr/bin/env python3
"""R3.2: reset-aware drift commit-gate experiment for VeloDepth.

All gate decisions use only current frame/state outputs. GT is used only after
selection to evaluate the emitted depth prediction. This experiment controls
persistent geometry-state writeback; it does NOT skip refresh-candidate compute.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
import traceback
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
R2A_DIR = HERE.parent / "r2a_commit_enum"
sys.path.insert(0, str(R2A_DIR))
sys.path.insert(0, str(HERE))
import r2a_commit_enum as r2a
from r3_gate_rules import POLICIES, DISABLED_POLICIES, decide


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    cols = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def normalized_sha256(path: Path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def finite_mean_tensor(x):
    if x is None:
        return float("nan")
    import torch
    y = x.detach().float().reshape(-1)
    y = y[torch.isfinite(y)]
    return float(y.mean().cpu().item()) if y.numel() else float("nan")


def depth_disagreement(pred_candidate, pred_prop):
    """GT-free relative disagreement between two same-frame depth predictions."""
    import numpy as np
    a = np.asarray(pred_candidate, dtype=np.float64)
    b = np.asarray(pred_prop, dtype=np.float64)
    if a.shape != b.shape:
        raise RuntimeError(f"Candidate/prop depth shape mismatch: {a.shape} vs {b.shape}")
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if not valid.any():
        return float("nan"), 0
    rel = np.abs(a[valid] - b[valid]) / np.maximum(np.abs(b[valid]), 1e-8)
    return float(rel.mean()), int(valid.sum())


class R3Experiment(r2a.GPUExperiment):
    def __init__(self, root, args):
        super().__init__(root, args)
        self._last_prop_output = None
        original_propagate = self.stream.propagate_step

        # Capture the propagation output before R2-A's candidate refresh overwrites
        # the local `out` variable. The original stream method and return values
        # remain unchanged.
        def capture_propagate(inputs, t, state, T):
            result = original_propagate(inputs, t, state, T)
            self._last_prop_output = result[0]
            return result

        self.stream.propagate_step = capture_propagate

    def causal_features(self, state, candidate, refresh_out, prop_out):
        if prop_out is None:
            raise RuntimeError("Missing propagation output at optional candidate anchor")
        # Preserve raw confidence summaries for diagnostics only. The two sources
        # are not comparable: keyframe confidence is emitted by confidence_mlp,
        # while the fast path currently uses logdepth_student as logconf_student.
        prop_conf = finite_mean_tensor(prop_out.confidence)
        cand_conf = finite_mean_tensor(refresh_out.confidence)
        prop_depth = self.cf.output_depth(prop_out, self.model, self.paddings, self.padded_shape)
        candidate_depth = self.cf.output_depth(refresh_out, self.model, self.paddings, self.padded_shape)
        residual, residual_valid_pixels = depth_disagreement(candidate_depth, prop_depth)
        return {
            "distance_kf": float(getattr(state, "distance_kf", float("nan"))),
            "flow_magnitude": float(getattr(state, "flow_magnitude", float("nan"))),
            "total_luma_change": float(getattr(state, "total_luma_change", float("nan"))),
            "confidence_prop_mean_raw_diagnostic_only": prop_conf,
            "confidence_candidate_mean_raw_diagnostic_only": cand_conf,
            "confidence_comparison_valid": False,
            "confidence_comparison_reason": "not comparable: keyframe confidence_mlp vs fast-path logconf_student=logdepth_student",
            "depth_disagreement_rel_mean": residual,
            "depth_disagreement_valid_pixels": residual_valid_pixels,
            "feature_source": "current propagation output, current refresh candidate, and current StreamState only; no GT",
        }

    def run_policy(self, policy, schedule, age_threshold, residual_threshold):
        self.seed(self.args.seed)
        state = self.restore(self.base)
        clock = r2a.Clocks()
        slots = set(schedule)
        rows, decision_rows = [], []
        self.sync()
        wall_start = time.perf_counter()
        last_kf = last_geom = 0

        for t in range(self.T):
            self._last_prop_output = None
            out, state, ev, candidate = self.prepare_step(state, t, t in slots)
            action = "NO_OPTIONAL_ACTION"
            features = {
                "distance_kf": float("nan"), "flow_magnitude": float("nan"),
                "total_luma_change": float("nan"),
                "confidence_prop_mean_raw_diagnostic_only": float("nan"),
                "confidence_candidate_mean_raw_diagnostic_only": float("nan"),
                "confidence_comparison_valid": False,
                "confidence_comparison_reason": "not applicable at non-anchor frame",
                "depth_disagreement_rel_mean": float("nan"),
                "depth_disagreement_valid_pixels": 0,
                "feature_source": "not_applicable",
            }
            if candidate is not None:
                features = self.causal_features(state, candidate, out, self._last_prop_output)
                commit = decide(policy, features, age_threshold, residual_threshold)
                action = "COMMIT" if commit else "DISCARD"
                if commit:
                    _, ms = self.timed(lambda: self.commit(state, candidate))
                    ev["snapshot_restore_commit_ms"] += ms
                decision_rows.append({
                    "strategy": policy, "sequence": self.args.sequence,
                    "anchor_frame_zero_based": t, "selected_action": action,
                    "gate_uses_gt": False, "gate_policy": policy,
                    "age_threshold": age_threshold, "residual_threshold": residual_threshold,
                    **features,
                })
            elif t in slots and ev["native_restart"]:
                action = "NATIVE_RESTART_NO_OPTIONAL_ACTION"
                decision_rows.append({
                    "strategy": policy, "sequence": self.args.sequence,
                    "anchor_frame_zero_based": t, "selected_action": action,
                    "gate_uses_gt": False, "gate_policy": policy,
                    "age_threshold": age_threshold, "residual_threshold": residual_threshold,
                    **features,
                })

            did_commit = bool(candidate is not None and action == "COMMIT")
            native = bool(t == 0 or ev["native_restart"])
            if native:
                last_kf = last_geom = t
            elif did_commit:
                last_geom = t
            clock.advance(t, native=native, commit=did_commit)
            if clock.steps_since_kf_reset != t - last_kf or clock.steps_since_geom_commit != t - last_geom:
                raise RuntimeError("Clock transition mismatch")
            # Accuracy metrics are computed only after the causal action is chosen.
            metric = self.cf.eval_output(out, self.model, self.paddings, self.padded_shape, self.gt[t])
            rows.append({
                "strategy": policy, "frame": t, "scheduled": int(t in slots),
                "gate_action": action, "geom_commit": int(did_commit),
                "output_source": "REFRESH" if candidate is not None or native else "PROP",
                "kf_reset": int(native), "scheduler_distance_kf": state.distance_kf,
                "gt_available": int(self.records[t]["gt_available"]),
                **asdict(clock), **ev,
                **{m: float(metric.get(m, float("nan"))) for m in ("absrel", "rmse", "delta1")},
            })
            del candidate, out

        self.sync()
        elapsed = (time.perf_counter() - wall_start) * 1000.0
        return rows, decision_rows, elapsed


def summarize(strategy_rows, schedule, wall_info):
    names = list(strategy_rows)
    if not names:
        raise RuntimeError("No R3 strategy trajectories produced")
    ref = strategy_rows[names[0]]
    for name, rows in strategy_rows.items():
        if len(rows) != len(ref):
            raise RuntimeError(f"Unequal trajectory length: {name}")
        for metric in ("absrel", "rmse", "delta1"):
            a = [math.isfinite(float(r[metric])) for r in ref]
            b = [math.isfinite(float(r[metric])) for r in rows]
            if a != b:
                raise RuntimeError(f"Metric mask mismatch: {name}/{metric}")
    summaries = []
    for name, rows in strategy_rows.items():
        def mean(k):
            vals = [float(r[k]) for r in rows if math.isfinite(float(r[k]))]
            return sum(vals) / len(vals) if vals else float("nan")
        summaries.append({
            "strategy": name, "frames": len(rows),
            "valid_absrel_frames": sum(math.isfinite(float(r["absrel"])) for r in rows),
            "K_requested": len(schedule),
            "refresh_candidates_optional": sum(int(r.get("optional_candidate_computed", 0)) for r in rows),
            "geom_commits_optional": sum(int(r.get("geom_commit", 0)) for r in rows),
            "native_restarts": sum(int(r.get("native_restart", 0)) for r in rows),
            "mean_absrel": mean("absrel"), "mean_rmse": mean("rmse"), "mean_delta1": mean("delta1"),
            "propagation_ms_actual_path": sum(float(r.get("propagation_ms", 0)) for r in rows),
            "keyframe_ms_actual_path": sum(float(r.get("keyframe_ms", 0)) for r in rows),
            "snapshot_restore_commit_ms_actual_path": sum(float(r.get("snapshot_restore_commit_ms", 0)) for r in rows),
            "trajectory_wall_ms_including_metrics": wall_info[name],
            "gate_uses_gt": False if name not in ("AlwaysCommit", "NeverCommit") else False,
        })
    return summaries


def package_review(out_dir, source_files, root):
    review = out_dir.parent / (out_dir.name + "_review.zip")
    if review.exists():
        raise FileExistsError(review)
    with zipfile.ZipFile(review, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file():
                z.write(p, "results/" + p.relative_to(out_dir).as_posix())
        for p in sorted(HERE.glob("*")):
            if p.is_file() and p.suffix in (".py", ".md", ".sh"):
                z.write(p, "runner/" + p.name)
        for p in sorted(R2A_DIR.glob("*")):
            if p.is_file() and p.suffix in (".py", ".json", ".md", ".sh"):
                z.write(p, "runner/r2a_dependency/" + p.name)
        for rel in source_files:
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
    ap.add_argument("--refresh-frames", default="5,14,23")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--resolution-level", type=int, default=0)
    ap.add_argument("--gt-scale", type=float, default=5000.0)
    ap.add_argument("--max-time-diff", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument("--age-threshold", type=float, default=16.0)
    ap.add_argument("--residual-threshold", type=float, default=0.10)
    ap.add_argument("--policies", default=",".join(POLICIES),
                    help="Comma-separated active policies: " + ",".join(POLICIES))
    args = ap.parse_args()

    root = args.project_root.resolve()
    schedule = [int(x) for x in args.refresh_frames.split(",")]
    if len(schedule) != 3 or schedule != sorted(set(schedule)) or schedule[0] < 1 or schedule[-1] >= args.frames - 1:
        ap.error("Exactly 3 increasing refresh frames are required, all in [1, frames-2]")
    if args.age_threshold < 0 or args.residual_threshold < 0:
        ap.error("Gate thresholds must be nonnegative")
    selected_policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    if not selected_policies or len(selected_policies) != len(set(selected_policies)):
        ap.error("--policies must contain one or more unique active policy names")
    unknown_policies = sorted(set(selected_policies) - set(POLICIES))
    if unknown_policies:
        ap.error(f"Unsupported/disabled policies: {unknown_policies}. Active policies: {list(POLICIES)}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = args.out_dir or root / "outputs/r_series/r3_2_reset_aware_drift_gate" / f"{args.sequence}_{args.frames}_{stamp}"
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        ap.error(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    source_files = list(dict.fromkeys(
        list(json.loads((R2A_DIR / "audited_source_hashes.json").read_text(encoding="utf-8"))) + [
            "experiments/p1/p1_multihorizon_counterfactual.py",
            "experiments/p1/p2_3_online_replay_v5_11_action_audit.py",
            "velodepth/models/velodepth.py", "velodepth/models/decoder_velo.py",
        ]
    ))
    meta = {
        "experiment": "R3.2 reset-aware drift commit-gate experiment",
        "sequence": args.sequence, "frames": args.frames,
        "refresh_frames_zero_based": schedule, "K_requested": len(schedule),
        "policies": selected_policies,
        "available_policies": list(POLICIES),
        "disabled_policies": DISABLED_POLICIES,
        "age_threshold": args.age_threshold,
        "residual_threshold": args.residual_threshold, "weights_frozen": True,
        "predictor_training": False, "gate_uses_future_gt": False,
        "gate_uses_gt": False, "online_feature_boundary": "current propagation output, current refresh candidate, and current StreamState only",
        "action_semantics": "At scheduled anchors, refresh output is used for the current frame regardless of COMMIT/DISCARD; the gate controls only the four audited persistent geometry fields for future frames.",
        "cost_semantics": "All strategies compute each scheduled refresh candidate. Post-refresh commit gating does not save candidate inference compute.",
        "status": "started",
    }
    try:
        audit = r2a.static_preflight(root)
        write_json(out / "source_audit.json", audit)
        meta["source_audit_passed"] = True
        if args.preflight_only:
            meta["status"] = "preflight_passed"
        else:
            exp = R3Experiment(root, args)
            with exp.torch.inference_mode():
                write_json(out / "r0_runtime_audit.json", exp.r0_audit(schedule))
                strategy_rows, wall_info, decision_rows = {}, {}, []
                for policy in selected_policies:
                    print(f"Running {policy}; schedule={schedule}; causal GT-free gate", flush=True)
                    rows, decisions, wall = exp.run_policy(policy, schedule, args.age_threshold, args.residual_threshold)
                    strategy_rows[policy] = rows
                    decision_rows.extend(decisions)
                    wall_info[policy] = wall
                    write_csv(out / f"frame_metrics_{policy}.csv", rows)
                write_csv(out / "gate_decisions.csv", decision_rows)
                write_csv(out / "trajectory_summary.csv", summarize(strategy_rows, schedule, wall_info))
                meta.update({
                    "status": "completed", "gpu": exp.torch.cuda.get_device_name(0),
                    "torch_version": str(exp.torch.__version__),
                    "gt_matches": sum(int(r["gt_available"]) for r in exp.records),
                    "preprocess_flow_capture_ms_separate": exp.preprocess_flow_capture_ms,
                    "runtime_helper_source_sha256": normalized_sha256(root / "experiments/p1/p1_multihorizon_counterfactual.py"),
                    "source_audit_passed": True,
                    "same_metric_masks_across_strategies": True,
                    "threshold_selection": "Thresholds are predeclared exploratory values. Decisions use no GT; do not select a deployable threshold from the same sequence by comparing GT metrics. Use feature/action sensitivity first, then cross-sequence validation.",
                    "confidence_gate_status": "disabled until candidate and propagation confidence values have comparable calibrated semantics",
                })
    except Exception:
        meta["status"] = "failed"
        (out / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(traceback.format_exc(), file=sys.stderr)
        write_json(out / "metadata.json", meta)
        package_review(out, source_files, root)
        return 1
    write_json(out / "metadata.json", meta)
    package_review(out, source_files, root)
    print(f"Results directory: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
