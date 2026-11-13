#!/usr/bin/env python3
"""Recompute descriptive M05 effects from an existing CSV without GPU inference."""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "p1"))
from cvpr_m05_formal import summarize_effects


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--effects-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True,
                   help="New directory; existing results are never overwritten")
    args = p.parse_args()
    source = args.effects_csv.absolute()
    out = args.out_dir.absolute()
    with source.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV has missing or duplicate headers")
        rows = list(reader)
    if not rows:
        raise ValueError("No anchor effects in CSV")
    fields = [x for x in reader.fieldnames if x.startswith("H4_") and x.endswith("_AbsRel")]
    fields.append("T_minus_O_H1_H4_window_AbsRel")
    for row in rows:
        for field in fields:
            value = row.get(field)
            if value not in (None, "") and not math.isfinite(float(value)):
                raise ValueError(f"Non-finite value in {field}")
    summary = summarize_effects(rows)
    result = {
        "scope": "offline descriptive recalculation; no model inference or invariant revalidation",
        "input_csv": str(source), "input_csv_sha256": digest(source),
        "summary_implementation_sha256": digest(ROOT / "experiments/p1/cvpr_m05_formal.py"),
        "rows": len(rows), "effects": summary,
    }
    report = ["# M05 offline descriptive resummary", "",
              "No inference or invariant checks were run. Original files remain unchanged.", "",
              f"Source: {source}", f"SHA256: {result['input_csv_sha256']}", "",
              "| Seed | A-native sequence-equal mean T minus O H4 AbsRel |",
              "|---:|---:|"]
    for row in summary["per_seed_primary"]:
        v = row["A_native_primary_sequence_equal_mean_T_minus_O_H4"]
        report.append(f"| {row['seed']} | {'NA' if v is None else format(v, '.8f')} |")
    report.extend(["", "Descriptive result only; anchors are not independent scenes.", ""])
    out.mkdir(parents=True, exist_ok=False)
    (out / "m05_offline_resummary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Offline resummary written to {out}")


if __name__ == "__main__":
    main()
