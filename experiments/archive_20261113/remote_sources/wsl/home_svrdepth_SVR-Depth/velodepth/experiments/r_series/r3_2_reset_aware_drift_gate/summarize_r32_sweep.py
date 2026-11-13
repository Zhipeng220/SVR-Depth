#!/usr/bin/env python3
"""Aggregate R3.2 closed-loop threshold sweeps into auditable CSV/Markdown files."""
from __future__ import annotations
import csv
import json
import re
import sys
from pathlib import Path

POLICY_ORDER = {"ResidualGate": 0, "DriftGate": 1, "AgeDriftGate": 2}


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def as_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def action_code(action):
    if action == "COMMIT": return "C"
    if action == "DISCARD": return "D"
    if action == "NATIVE_RESTART_NO_OPTIONAL_ACTION": return "N"
    return "X"


def main(root: Path):
    metric_rows, action_rows = [], []
    for summary_path in sorted(root.glob("*/thr_*/trajectory_summary.csv")):
        run_dir = summary_path.parent
        seq = run_dir.parent.name
        match = re.fullmatch(r"thr_(\d+p\d+)", run_dir.name)
        if not match:
            continue
        threshold = float(match.group(1).replace("p", "."))
        meta_path = run_dir / "metadata.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        summary = read_csv(summary_path)
        decisions_path = run_dir / "gate_decisions.csv"
        decisions = read_csv(decisions_path) if decisions_path.exists() else []
        for row in summary:
            policy = row.get("strategy", "")
            if policy not in POLICY_ORDER:
                continue
            ds = sorted(
                [d for d in decisions if d.get("strategy") == policy],
                key=lambda d: int(d.get("anchor_frame_zero_based", 0)),
            )
            pattern = "".join(action_code(d.get("selected_action", "")) for d in ds)
            commits = sum(1 for d in ds if d.get("selected_action") == "COMMIT")
            metric_rows.append({
                "sequence": seq,
                "residual_threshold": f"{threshold:.2f}",
                "policy": policy,
                "pattern_9_37_65": pattern,
                "selected_commit_count": commits,
                "frames": row.get("frames", ""),
                "valid_absrel_frames": row.get("valid_absrel_frames", ""),
                "K_requested": row.get("K_requested", ""),
                "refresh_candidates_optional": row.get("refresh_candidates_optional", ""),
                "native_restarts": row.get("native_restarts", ""),
                "mean_absrel": row.get("mean_absrel", ""),
                "mean_rmse": row.get("mean_rmse", ""),
                "mean_delta1": row.get("mean_delta1", ""),
                "trajectory_wall_ms_including_metrics": row.get("trajectory_wall_ms_including_metrics", ""),
                "gate_uses_gt": row.get("gate_uses_gt", ""),
                "run_dir": str(run_dir),
            })
        for d in decisions:
            if d.get("strategy") not in POLICY_ORDER:
                continue
            action_rows.append({
                "sequence": seq,
                "residual_threshold": f"{threshold:.2f}",
                "policy": d.get("strategy", ""),
                "anchor_frame_zero_based": d.get("anchor_frame_zero_based", ""),
                "selected_action": d.get("selected_action", ""),
                "distance_kf": d.get("distance_kf", ""),
                "flow_magnitude": d.get("flow_magnitude", ""),
                "depth_disagreement_rel_mean": d.get("depth_disagreement_rel_mean", ""),
                "depth_disagreement_valid_pixels": d.get("depth_disagreement_valid_pixels", ""),
                "gate_uses_gt": d.get("gate_uses_gt", ""),
                "run_dir": str(run_dir),
            })
    metric_rows.sort(key=lambda r: (r["sequence"], float(r["residual_threshold"]), POLICY_ORDER.get(r["policy"], 99)))
    action_rows.sort(key=lambda r: (r["sequence"], float(r["residual_threshold"]), POLICY_ORDER.get(r["policy"], 99), int(r["anchor_frame_zero_based"])))
    if not metric_rows:
        raise SystemExit(f"No sweep results found under {root}; expected */thr_*/trajectory_summary.csv")
    write_csv(root / "r32_sensitivity_summary.csv", metric_rows)
    write_csv(root / "r32_sensitivity_actions.csv", action_rows)
    lines = [
        "# R3.2 DriftGate / AgeDriftGate sensitivity summary", "",
        f"- Result root: `{root}`",
        "- Thresholds are exploratory; do not select a deployable threshold from held-out GT metrics.",
        "- `ResidualGate` preserves the original R3 direction as a control.",
        "- `DriftGate` commits when residual >= threshold.",
        "- `AgeDriftGate` commits only when distance_kf >= 16 and residual >= threshold.",
        "- All policies compute the same refresh candidates; no refresh-inference saving is claimed.", "",
        "| Sequence | Threshold | Policy | Pattern (9/37/65) | Commits | AbsRel | RMSE | δ1 |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for r in metric_rows:
        lines.append(
            f"| {r['sequence']} | {float(r['residual_threshold']):.2f} | {r['policy']} | "
            f"`{r['pattern_9_37_65']}` | {r['selected_commit_count']} | "
            f"{as_float(r['mean_absrel']):.6f} | {as_float(r['mean_rmse']):.6f} | {as_float(r['mean_delta1']):.6f} |"
        )
    (root / "R3_2敏感性结果摘要.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary: {root / 'r32_sensitivity_summary.csv'}")
    print(f"Actions: {root / 'r32_sensitivity_actions.csv'}")
    print(f"Report: {root / 'R3_2敏感性结果摘要.md'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python summarize_r32_sweep.py <sensitivity_root>")
    main(Path(sys.argv[1]).resolve())
