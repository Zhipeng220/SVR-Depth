#!/usr/bin/env python3
"""Aggregate R3 ResidualGate threshold-sweep outputs; no threshold is selected here."""
from pathlib import Path
import csv, json, sys

root = Path(sys.argv[1]).resolve()
summary_rows = []
action_rows = []
for out in sorted(root.rglob("thr_*")):
    if not out.is_dir():
        continue
    meta_path = out / "metadata.json"
    summary_path = out / "trajectory_summary.csv"
    decisions_path = out / "gate_decisions.csv"
    if not (meta_path.exists() and summary_path.exists() and decisions_path.exists()):
        continue
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    threshold = float(meta["residual_threshold"])
    sequence = str(meta.get("sequence", "unknown"))
    with summary_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            row["sequence"] = sequence
            row["residual_threshold"] = threshold
            row["output_dir"] = out.name
            summary_rows.append(row)
    with decisions_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            row["sequence"] = sequence
            row["residual_threshold"] = threshold
            row["output_dir"] = out.name
            action_rows.append(row)

def write(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

write(root / "residual_threshold_sweep_summary.csv", summary_rows)
write(root / "residual_threshold_sweep_actions.csv", action_rows)
print(f"Threshold runs summarized: {len(summary_rows)}")
print(f"Decision rows collected: {len(action_rows)}")
print("Files: residual_threshold_sweep_summary.csv, residual_threshold_sweep_actions.csv")
print("Reminder: choose thresholds by GT-free feature/action stability first; do not tune on this smoke sequence's GT metrics.")
