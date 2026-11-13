#!/usr/bin/env python3
"""R2-B-CL: dynamic short/long-horizon commit-or-discard diagnostic oracle.

The horizon oracle uses future GT and is strictly a diagnostic upper bound, not
an online-deployable policy. At each scheduled anchor, both action branches are
rolled out from the current policy's own state; the selected state is then used
for the actual continuation and the next anchor is re-evaluated from that state.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
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
import r2a_commit_enum as r2a  # audited R2-A state capture/restore and inference harness

GAMMA = 0.9
HORIZONS = (1, 8)


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


from r2b_math import discounted_gain


def normalized_sha256(path: Path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


class R2BCLExperiment(r2a.GPUExperiment):
    def future_rollout(self, snapshot, anchor, horizon, branch_name, policy_name):
        """Roll out one branch with no optional actions inside the horizon."""
        state = self.restore(snapshot)
        rows = []
        for k in range(1, horizon + 1):
            frame = anchor + k
            if frame >= self.T:
                break
            out, state, ev, candidate = self.prepare_step(state, frame, False)
            if candidate is not None:
                raise RuntimeError("Unexpected optional candidate during fixed-horizon branch rollout")
            metric = self.cf.eval_output(out, self.model, self.paddings, self.padded_shape, self.gt[frame])
            row = {
                "policy": policy_name,
                "anchor_frame_zero_based": anchor,
                "branch": branch_name,
                "future_offset_k": k,
                "frame": frame,
                "gt_available": int(self.records[frame]["gt_available"]),
                "native_restart": int(ev["native_restart"]),
                "propagation_ms": float(ev["propagation_ms"]),
                "keyframe_ms": float(ev["keyframe_ms"]),
                **{m: float(metric.get(m, float("nan"))) for m in ("absrel", "rmse", "delta1")},
            }
            if self.records[frame]["gt_available"]:
                if not math.isfinite(row["absrel"]):
                    raise RuntimeError(
                        f"GT exists but branch AbsRel is non-finite: {policy_name}/{anchor}/{branch_name}/{frame}"
                    )
            rows.append(row)
            del candidate, out
        return rows

    def run_horizon_oracle(self, horizon, schedule):
        """Run a dynamic diagnostic oracle; re-decide at every anchor on own state."""
        self.seed(self.args.seed)
        state = self.restore(self.base)
        slots = set(schedule)
        rows, decisions, branch_rows = [], [], []
        last_kf = last_geom = 0
        self.sync()
        wall_start = time.perf_counter()
        branch_wall_total_ms = 0.0
        branch_frames_total = 0

        for t in range(self.T):
            out, state, ev, candidate = self.prepare_step(state, t, t in slots)
            # Evaluate current output before counterfactual rollouts mutate model runtime.
            metric = self.cf.eval_output(out, self.model, self.paddings, self.padded_shape, self.gt[t])
            chosen_action = "NO_OPTIONAL_ACTION"
            gain_sum = float("nan")
            valid_pairs = 0
            branch_wall_ms = 0.0
            d_loss_sum = c_loss_sum = float("nan")
            d_snapshot = c_snapshot = None

            if candidate is not None:
                # Freeze the post-propagation state. The current refresh output is shared
                # by Commit and Discard; only the four audited persistent fields differ.
                branch_start = time.perf_counter()
                post_snapshot = self.capture(state)
                discard_state = self.restore(post_snapshot)
                d_snapshot = self.capture(discard_state)
                commit_state = self.restore(post_snapshot)
                self.commit(commit_state, candidate)
                c_snapshot = self.capture(commit_state)

                d_rows = self.future_rollout(d_snapshot, t, horizon, "DISCARD", f"H{horizon}_Oracle_CL")
                c_rows = self.future_rollout(c_snapshot, t, horizon, "COMMIT", f"H{horizon}_Oracle_CL")
                branch_frames_total += len(d_rows) + len(c_rows)
                branch_rows.extend(d_rows)
                branch_rows.extend(c_rows)

                d_by_k = {int(r["future_offset_k"]): float(r["absrel"]) for r in d_rows}
                c_by_k = {int(r["future_offset_k"]): float(r["absrel"]) for r in c_rows}
                d_loss_sum = sum(GAMMA ** k * v for k, v in d_by_k.items() if math.isfinite(v))
                c_loss_sum = sum(GAMMA ** k * v for k, v in c_by_k.items() if math.isfinite(v))
                try:
                    gain_sum, valid_pairs = discounted_gain(d_by_k, c_by_k, horizon, GAMMA)
                    # Tie goes to DISCARD. The future GT choice is explicitly diagnostic.
                    did_choose_commit = gain_sum > 0.0
                    chosen_action = "COMMIT" if did_choose_commit else "DISCARD"
                except RuntimeError as exc:
                    if "No valid paired future GT frames" not in str(exc):
                        raise
                    # If the horizon contains no valid GT, no oracle evidence exists;
                    # fail closed to DISCARD and preserve the missing-evidence record.
                    gain_sum, valid_pairs = float("nan"), 0
                    chosen_action = "DISCARD_NO_VALID_GT"
                    did_choose_commit = False
                chosen_snapshot = c_snapshot if did_choose_commit else d_snapshot
                # Restore the chosen branch's full runtime snapshot before continuing the
                # actual trajectory. At the next anchor the oracle is re-evaluated anew.
                state = self.restore(chosen_snapshot)
                branch_wall_ms = (time.perf_counter() - branch_start) * 1000.0
                branch_wall_total_ms += branch_wall_ms
                decisions.append({
                    "policy": f"H{horizon}_Oracle_CL",
                    "sequence": self.args.sequence,
                    "anchor_frame_zero_based": t,
                    "horizon": horizon,
                    "gamma": GAMMA,
                    "discounted_gain_discard_minus_commit": gain_sum,
                    "discounted_discard_loss_sum": d_loss_sum,
                    "discounted_commit_loss_sum": c_loss_sum,
                    "valid_future_pairs": valid_pairs,
                    "selected_action": chosen_action,
                    "tie_policy": "DISCARD",
                    "oracle_uses_future_gt": True,
                    "branch_rollout_ms": branch_wall_ms,
                    "branch_future_frames_evaluated": len(d_rows) + len(c_rows),
                    "native_restarts_discard_branch": sum(int(r["native_restart"]) for r in d_rows),
                    "native_restarts_commit_branch": sum(int(r["native_restart"]) for r in c_rows),
                })
            elif t in slots and ev["native_restart"]:
                chosen_action = "NATIVE_RESTART_NO_OPTIONAL_ACTION"

            did_commit = bool(candidate is not None and chosen_action == "COMMIT")
            native = bool(t == 0 or ev["native_restart"])
            if native:
                last_kf = last_geom = t
            elif did_commit:
                last_geom = t
            clock = r2a.Clocks() if t == 0 else clock
            clock.advance(t, native=native, commit=did_commit)
            if clock.steps_since_kf_reset != t - last_kf or clock.steps_since_geom_commit != t - last_geom:
                raise RuntimeError("Clock transition mismatch in dynamic oracle trajectory")

            rows.append({
                "strategy": f"H{horizon}_Oracle_CL",
                "frame": t,
                "scheduled": int(t in slots),
                "oracle_horizon": horizon if t in slots and candidate is not None else "",
                "oracle_action": chosen_action,
                "oracle_future_gain_sum": gain_sum,
                "oracle_future_valid_pairs": valid_pairs,
                "oracle_branch_rollout_ms": branch_wall_ms,
                "oracle_discounted_discard_loss_sum": d_loss_sum,
                "oracle_discounted_commit_loss_sum": c_loss_sum,
                "geom_commit": int(did_commit),
                "output_source": "REFRESH" if candidate is not None or native else "PROP",
                "kf_reset": int(native),
                "scheduler_distance_kf": state.distance_kf,
                "gt_available": int(self.records[t]["gt_available"]),
                **asdict(clock),
                **ev,
                **{m: float(metric.get(m, float("nan"))) for m in ("absrel", "rmse", "delta1")},
            })
            del candidate, out

        self.sync()
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        extra = {
            "strategy": f"H{horizon}_Oracle_CL",
            "instrumented_trajectory_wall_ms_including_oracle_rollouts": wall_ms,
            "oracle_branch_rollout_ms_total": branch_wall_total_ms,
            "oracle_branch_future_frames_evaluated": branch_frames_total,
        }
        return rows, decisions, branch_rows, extra


def summarize_strategies(strategy_rows, schedule, wall_info):
    names = list(strategy_rows)
    if not names:
        raise RuntimeError("No strategy trajectories produced")
    reference = strategy_rows[names[0]]
    for name, rows in strategy_rows.items():
        if len(rows) != len(reference):
            raise RuntimeError(f"Unequal trajectory lengths for {name}")
        for metric in ("absrel", "rmse", "delta1"):
            ref_mask = [math.isfinite(float(r[metric])) for r in reference]
            mask = [math.isfinite(float(r[metric])) for r in rows]
            if mask != ref_mask:
                raise RuntimeError(f"Metric mask differs across strategies: {name}/{metric}")
    out = []
    for name, rows in strategy_rows.items():
        def mean(metric):
            vals = [float(r[metric]) for r in rows if math.isfinite(float(r[metric]))]
            return sum(vals) / len(vals) if vals else float("nan")
        row = {
            "strategy": name,
            "frames": len(rows),
            "valid_absrel_frames": sum(math.isfinite(float(r["absrel"])) for r in rows),
            "K_requested": len(schedule),
            "refresh_candidates_optional": sum(int(r.get("optional_candidate_computed", 0)) for r in rows),
            "geom_commits_optional": sum(int(r.get("geom_commit", 0)) for r in rows),
            "native_restarts": sum(int(r.get("native_restart", 0)) for r in rows),
            "mean_absrel": mean("absrel"),
            "mean_rmse": mean("rmse"),
            "mean_delta1": mean("delta1"),
            "propagation_ms_actual_path": sum(float(r.get("propagation_ms", 0)) for r in rows),
            "keyframe_ms_actual_path": sum(float(r.get("keyframe_ms", 0)) for r in rows),
            "snapshot_restore_commit_ms_actual_path": sum(float(r.get("snapshot_restore_commit_ms", 0)) for r in rows),
        }
        row.update(wall_info.get(name, {}))
        out.append(row)
    return out


def package_review(out_dir: Path, source_files, project_root: Path):
    review = out_dir.parent / (out_dir.name + "_review.zip")
    if review.exists():
        raise FileExistsError(review)
    with zipfile.ZipFile(review, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file():
                zf.write(p, "results/" + p.relative_to(out_dir).as_posix())
        for p in sorted(HERE.glob("*")):
            if p.is_file() and p.suffix in (".py", ".md", ".sh"):
                zf.write(p, "runner/" + p.name)
        for p in sorted(R2A_DIR.glob("*")):
            if p.is_file() and p.suffix in (".py", ".json", ".md", ".sh"):
                zf.write(p, "runner/r2a_dependency/" + p.name)
        for rel in source_files:
            src = project_root / rel
            if src.is_file():
                zf.write(src, "project_sources/" + rel)
        if zf.testzip() is not None:
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
    args = ap.parse_args()

    root = args.project_root.resolve()
    schedule = [int(x) for x in args.refresh_frames.split(",")]
    if len(schedule) != 3 or schedule != sorted(set(schedule)) or schedule[0] < 1 or schedule[-1] >= args.frames - 1:
        ap.error("Exactly 3 increasing refresh frames are required, all in [1, frames-2]")
    if max(HORIZONS) >= min(b - a for a, b in zip(schedule, schedule[1:])):
        ap.error("Horizon must be shorter than the gap between scheduled anchors to avoid future optional actions inside lookahead")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = args.out_dir or root / "outputs/r_series/r2b_cl" / f"{args.sequence}_{args.frames}_{stamp}"
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        ap.error(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    source_files = list(dict.fromkeys(
        list(json.loads((R2A_DIR / "audited_source_hashes.json").read_text(encoding="utf-8"))) + [
            "experiments/p1/p1_multihorizon_counterfactual.py",
            "experiments/p1/p2_3_online_replay_v5_11_action_audit.py",
            "velodepth/models/velodepth.py",
            "velodepth/models/decoder_velo.py",
        ]
    ))
    meta = {
        "experiment": "R2-B-CL dynamic H1/H8 commit-discard oracle diagnostic",
        "sequence": args.sequence,
        "frames": args.frames,
        "refresh_frames_zero_based": schedule,
        "K_requested": len(schedule),
        "horizons": list(HORIZONS),
        "gamma": GAMMA,
        "weights_frozen": True,
        "predictor_training": False,
        "oracle_uses_future_gt": True,
        "online_deployable": False,
        "state_recursion": "Each oracle re-evaluates at each scheduled anchor on its own selected state; after each choice, the selected branch state is restored and used for actual continuation.",
        "lookahead_policy": "No elective actions inside H window; native restarts follow normal stream semantics. H must be shorter than anchor spacing.",
        "tie_policy": "DISCARD",
        "status": "started",
    }
    try:
        audit = r2a.static_preflight(root)
        write_json(out / "source_audit.json", audit)
        meta["source_audit_passed"] = True
        if args.preflight_only:
            meta["status"] = "preflight_passed"
        if not args.preflight_only:
            exp = R2BCLExperiment(root, args)
            with exp.torch.inference_mode():
                write_json(out / "r0_runtime_audit.json", exp.r0_audit(schedule))
                strategy_rows, wall_info = {}, {}
                branch_rows, decision_rows = [], []

                for plan, name in (("000", "Fixed_000"), ("111", "Fixed_111")):
                    print(f"Running baseline {name}; schedule={schedule}", flush=True)
                    rows, wall = exp.run_plan(plan, schedule)
                    for row in rows:
                        row["strategy"] = name
                    strategy_rows[name] = rows
                    wall_info[name] = {"trajectory_wall_ms_including_metrics": wall,
                                       "oracle_branch_rollout_ms_total": 0.0,
                                       "oracle_branch_future_frames_evaluated": 0}
                    write_csv(out / f"frame_metrics_{name}.csv", rows)

                for horizon in HORIZONS:
                    name = f"H{horizon}_Oracle_CL"
                    print(f"Running {name}; schedule={schedule}; future-GT diagnostic only", flush=True)
                    rows, decisions, branches, extra = exp.run_horizon_oracle(horizon, schedule)
                    strategy_rows[name] = rows
                    decision_rows.extend(decisions)
                    branch_rows.extend(branches)
                    wall_info[name] = extra
                    write_csv(out / f"frame_metrics_{name}.csv", rows)

                summaries = summarize_strategies(strategy_rows, schedule, wall_info)
                write_csv(out / "trajectory_summary.csv", summaries)
                write_csv(out / "oracle_decisions.csv", decision_rows)
                write_csv(out / "oracle_branch_metrics.csv", branch_rows)
                meta.update({
                    "status": "completed",
                    "gpu": exp.torch.cuda.get_device_name(0),
                    "torch_version": str(exp.torch.__version__),
                    "gt_matches": sum(int(r["gt_available"]) for r in exp.records),
                    "preprocess_flow_capture_ms_separate": exp.preprocess_flow_capture_ms,
                    "runtime_helper_source_sha256": normalized_sha256(root / "experiments/p1/p1_multihorizon_counterfactual.py"),
                    "native_restart_policy": "Native restart takes priority and suppresses elective commit on that frame.",
                    "candidate_semantics": "At an anchor, the current refresh output is shared by both actions; only audited COMMIT_FIELDS differ in persistent state.",
                    "cost_interpretation": "Oracle branch rollout costs are reported separately. These future-GT diagnostics are not online latency or equal-cost deployable policies.",
                    "same_metric_masks_across_strategies": True,
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
