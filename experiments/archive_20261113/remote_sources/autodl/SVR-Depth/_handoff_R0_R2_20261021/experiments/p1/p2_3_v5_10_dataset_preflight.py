#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inventory Bonn RGB-D sequences for P2.3 v5.10 external validation.

No model imports, no GPU use. An external candidate must not be one of the
already-used Crowd2/Balloon2/Crowd3 sequences, must contain the requested number
of existing RGB frames, and must have enough existing depth GT files matched
within the configured timestamp tolerance on the evaluated prefix.
"""
import argparse
import bisect
import csv
import re
from pathlib import Path

USED_DEFAULT = ("crowd2", "balloon2", "crowd3")


def read_index(root: Path, filename: str):
    path = root / filename
    if not path.is_file():
        return []
    rows = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            ts = float(parts[0])
        except ValueError:
            continue
        rel = " ".join(parts[1:])
        full = root / rel
        rows.append((ts, rel, full.is_file()))
    rows.sort(key=lambda x: x[0])
    return rows


def sequence_name(path: Path):
    name = path.name
    prefix = "rgbd_bonn_"
    return name[len(prefix):] if name.startswith(prefix) else name


def resembles_previously_used(seq: str, used):
    # Catch common renamed copies such as crowd2_copy, crowd2-v2, or copy_crowd2.
    for old in used:
        if seq == old:
            return True
        if re.search(rf"(^|[^a-z0-9]){re.escape(old)}($|[^a-z0-9])", seq):
            return True
    return False


def inspect_sequence(path: Path, frames: int, tolerance: float, min_matches: int, used):
    seq = sequence_name(path)
    rgb_all = read_index(path, "rgb.txt")
    depth_all = read_index(path, "depth.txt")
    rgb_existing = [r for r in rgb_all if r[2]]
    depth_existing = [r for r in depth_all if r[2]]
    selected = rgb_existing[:frames]
    depth_ts = [r[0] for r in depth_existing]
    matched = 0
    diffs = []
    for ts, _, _ in selected:
        i = bisect.bisect_left(depth_ts, ts)
        near = []
        if i < len(depth_ts):
            near.append(depth_ts[i])
        if i > 0:
            near.append(depth_ts[i-1])
        if near:
            dt = min(abs(x-ts) for x in near)
            if dt <= tolerance:
                matched += 1
                diffs.append(dt)

    is_used = resembles_previously_used(seq.lower(), {x.lower() for x in used})
    enough_rgb = len(selected) >= frames
    enough_gt = matched >= min_matches
    has_indexes = (path / "rgb.txt").is_file() and (path / "depth.txt").is_file()
    eligible = bool(has_indexes and not is_used and enough_rgb and enough_gt)
    reasons = []
    if is_used:
        reasons.append("previously_used_sequence")
    if not has_indexes:
        reasons.append("missing_rgb_or_depth_index")
    if not enough_rgb:
        reasons.append(f"only_{len(selected)}_existing_rgb_frames")
    if not enough_gt:
        reasons.append(f"only_{matched}_GT_matches_below_{min_matches}")
    if eligible:
        reasons.append("eligible_external_holdout")
    return {
        "sequence": seq,
        "data_root": str(path.resolve()),
        "rgb_index_rows": len(rgb_all),
        "rgb_files_exist": len(rgb_existing),
        "depth_index_rows": len(depth_all),
        "depth_files_exist": len(depth_existing),
        "eval_frames_requested": frames,
        "eval_rgb_frames_available": len(selected),
        "gt_matches_prefix": matched,
        "gt_match_fraction_prefix": matched / len(selected) if selected else 0.0,
        "mean_abs_gt_time_diff_s": sum(diffs) / len(diffs) if diffs else "",
        "gt_tolerance_s": tolerance,
        "minimum_gt_matches_required": min_matches,
        "previously_used": is_used,
        "eligible_external_holdout": eligible,
        "status_reason": ";".join(reasons),
    }


def main():
    ap = argparse.ArgumentParser(description="P2.3 v5.10 Bonn sequence inventory (CPU preflight)")
    ap.add_argument("--dataset-root", default="/root/autodl-tmp/datasets")
    ap.add_argument("--frames", type=int, default=86)
    ap.add_argument("--max-time-diff", type=float, default=0.02)
    ap.add_argument("--min-gt-matches", type=int, default=80)
    ap.add_argument("--used-sequences", default="crowd2,balloon2,crowd3")
    ap.add_argument("--out-csv", default="outputs/p2_3_v5_10_external/p2_3_v5_10_preflight.csv")
    args = ap.parse_args()
    if args.frames < 1 or args.min_gt_matches < 0 or args.min_gt_matches > args.frames:
        ap.error("Require frames >= 1 and 0 <= min-gt-matches <= frames.")
    root = Path(args.dataset_root)
    used = {x.strip() for x in args.used_sequences.split(",") if x.strip()}
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and
                  ((p / "rgb.txt").is_file() or (p / "depth.txt").is_file()))
    rows = [inspect_sequence(p, args.frames, args.max_time_diff, args.min_gt_matches, used)
            for p in dirs]
    rows.sort(key=lambda r: (not r["eligible_external_holdout"], r["sequence"]))
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sequence", "data_root", "rgb_index_rows", "rgb_files_exist",
        "depth_index_rows", "depth_files_exist", "eval_frames_requested",
        "eval_rgb_frames_available", "gt_matches_prefix", "gt_match_fraction_prefix",
        "mean_abs_gt_time_diff_s", "gt_tolerance_s", "minimum_gt_matches_required",
        "previously_used", "eligible_external_holdout", "status_reason",
    ]
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Dataset root: {root}")
    print(f"Sequences inspected: {len(rows)}")
    eligible = [r for r in rows if r["eligible_external_holdout"]]
    for r in rows:
        status = "ELIGIBLE" if r["eligible_external_holdout"] else "SKIP"
        print(f"[{status}] {r['sequence']}: RGB={r['rgb_files_exist']}, "
              f"GT-match first {args.frames}={r['gt_matches_prefix']}/{min(args.frames, r['rgb_files_exist'])}; "
              f"{r['status_reason']}")
    print(f"Eligible unseen sequences: {len(eligible)}")
    print(f"Wrote: {out}")
    if not eligible:
        print("NO ELIGIBLE EXTERNAL SEQUENCE: do not run GPU replay; add/download a new suitable sequence first.")


if __name__ == "__main__":
    main()
