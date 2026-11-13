#!/usr/bin/env python3
"""GT4: build a one-to-one RGB/depth timestamp manifest for a Bonn sequence.

This script does not run model inference and does not need GT1 predictions.
It preserves every RGB index and matches existing depth files monotonically,
maximizing match count first and minimizing total absolute timestamp error second.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

DEFAULT_TOLERANCE_MS = 20.0
EPS = 1e-9


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        if not fieldnames:
            f.write("\n")
            return
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def resolve_path(root: Path, raw_path: str, subdir: str):
    p = Path(raw_path)
    if p.is_absolute():
        candidates = [p]
    else:
        candidates = [root / p, root / subdir / p, root / subdir / p.name, root / p.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def read_index(root: Path, filename: str, subdir: str):
    index_path = root / filename
    if not index_path.is_file():
        raise FileNotFoundError(f"索引不存在：{index_path}")
    rows = []
    for line_number, line in enumerate(index_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            timestamp = float(fields[0])
        except ValueError:
            continue
        rows.append({
            "timestamp": timestamp,
            "raw_path": fields[1],
            "path": resolve_path(root, fields[1], subdir),
            "line_number": line_number,
        })
    if not rows:
        raise RuntimeError(f"索引没有有效记录：{index_path}")
    return rows


def ordered_one_to_one_match(rgb_records, depth_records, tolerance_ms):
    # A physical depth image can appear only once. Keep its first index occurrence.
    path_to_record = {}
    duplicate_path_rows = 0
    for rec in depth_records:
        if rec["path"] is None:
            continue
        key = str(rec["path"])
        if key in path_to_record:
            duplicate_path_rows += 1
            continue
        path_to_record[key] = rec
    valid_depths = sorted(path_to_record.values(), key=lambda r: (r["timestamp"], r["line_number"]))
    rgb_times = [float(r["timestamp"]) for r in rgb_records]
    depth_times = [float(r["timestamp"]) for r in valid_depths]
    n, m = len(rgb_records), len(valid_depths)
    count = [[0] * (m + 1) for _ in range(n + 1)]
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    action = [[0] * (m + 1) for _ in range(n + 1)]  # 1=skip RGB, 2=skip depth, 3=match
    for i in range(1, n + 1):
        rt = rgb_times[i - 1]
        for j in range(1, m + 1):
            best_count = count[i - 1][j]
            best_cost = cost[i - 1][j]
            best_action = 1
            c2, d2 = count[i][j - 1], cost[i][j - 1]
            if c2 > best_count or (c2 == best_count and d2 < best_cost - EPS):
                best_count, best_cost, best_action = c2, d2, 2
            delta_ms = abs(depth_times[j - 1] - rt) * 1000.0
            if delta_ms <= tolerance_ms + EPS:
                c3 = count[i - 1][j - 1] + 1
                d3 = cost[i - 1][j - 1] + delta_ms
                if (c3 > best_count or
                    (c3 == best_count and d3 < best_cost - EPS) or
                    (c3 == best_count and abs(d3 - best_cost) <= EPS)):
                    best_count, best_cost, best_action = c3, d3, 3
            count[i][j], cost[i][j], action[i][j] = best_count, best_cost, best_action
    matches = {}
    i, j = n, m
    while i > 0 and j > 0:
        a = action[i][j]
        if a == 3:
            matches[i - 1] = j - 1
            i -= 1; j -= 1
        elif a == 1:
            i -= 1
        elif a == 2:
            j -= 1
        else:
            raise RuntimeError(f"DP 回溯异常：i={i}, j={j}, action={a}")
    assigned = [matches[k] for k in sorted(matches)]
    if len(assigned) != len(set(assigned)):
        raise AssertionError("一对一约束失败：GT 被重复分配")
    if any(assigned[k] >= assigned[k + 1] for k in range(len(assigned) - 1)):
        raise AssertionError("单调约束失败：GT 顺序逆序")
    for ri, di in matches.items():
        dt = abs(rgb_times[ri] - depth_times[di]) * 1000.0
        if dt > tolerance_ms + 1e-6:
            raise AssertionError(f"时间差超容差：RGB={ri}, dt={dt:.6f}ms")
    return valid_depths, matches, duplicate_path_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence-root", required=True)
    ap.add_argument("--sequence-name", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-time-diff-ms", type=float, default=DEFAULT_TOLERANCE_MS)
    args = ap.parse_args()
    t0 = time.time()
    root = Path(args.sequence_root).expanduser().resolve()
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"GT4 pair builder: {args.sequence_name}", flush=True)
    print("Sequence root:", root, flush=True)
    print("Tolerance (ms):", args.max_time_diff_ms, flush=True)
    if args.max_time_diff_ms <= 0:
        raise ValueError("时间差阈值必须为正数")
    rgb = read_index(root, "rgb.txt", "rgb")
    depth = read_index(root, "depth.txt", "depth")
    if any(rgb[i]["timestamp"] <= rgb[i-1]["timestamp"] for i in range(1, len(rgb))):
        raise RuntimeError("rgb.txt 时间戳非严格递增")
    missing_rgb = [r for r in rgb if r["path"] is None]
    missing_depth = [r for r in depth if r["path"] is None]
    write_csv(out / "gt4_missing_depth_files.csv", [
        {"depth_index_line": r["line_number"], "timestamp": r["timestamp"], "raw_path": r["raw_path"]}
        for r in missing_depth
    ], ["depth_index_line", "timestamp", "raw_path"])
    if missing_rgb:
        write_csv(out / "gt4_missing_rgb_files.csv", [
            {"rgb_index": i, "rgb_index_line": r["line_number"], "timestamp": r["timestamp"], "raw_path": r["raw_path"]}
            for i, r in enumerate(rgb) if r["path"] is None
        ])
        raise RuntimeError(f"发现 {len(missing_rgb)} 个 RGB 文件缺失；已写出 gt4_missing_rgb_files.csv，停止配对")
    valid_depths, matches, duplicate_path_rows = ordered_one_to_one_match(rgb, depth, args.max_time_diff_ms)
    used_depth_indices = set(matches.values())
    pair_rows = []
    unmatched_rows = []
    used_deltas = []
    for rgb_idx, rr in enumerate(rgb):
        depth_idx = matches.get(rgb_idx)
        dr = valid_depths[depth_idx] if depth_idx is not None else None
        signed = (dr["timestamp"] - rr["timestamp"]) * 1000.0 if dr else ""
        delta = abs(signed) if dr else ""
        if dr:
            used_deltas.append(float(delta))
        row = {
            "frame_index_zero_based": rgb_idx,
            "prediction_slot": rgb_idx,
            "rgb_timestamp": rr["timestamp"],
            "rgb_path": str(rr["path"]),
            "depth_timestamp": dr["timestamp"] if dr else "",
            "depth_path": str(dr["path"]) if dr else "",
            "signed_delta_ms": signed,
            "abs_delta_ms": delta,
            "gt_match_status": "MATCHED" if dr else "NO_GT_WITHIN_TOLERANCE_OR_UNASSIGNED",
            "gt_raw_index_line": dr["line_number"] if dr else "",
            "gt_reused": False,
        }
        pair_rows.append(row)
        if dr is None:
            unmatched_rows.append({"frame_index_zero_based": rgb_idx, "rgb_timestamp": rr["timestamp"], "rgb_path": str(rr["path"]), "reason": "No unique monotonic GT assignment within tolerance"})
    unused_rows = [
        {"depth_timestamp": r["timestamp"], "depth_path": str(r["path"]), "depth_index_line": r["line_number"]}
        for j, r in enumerate(valid_depths) if j not in used_depth_indices
    ]
    write_csv(out / "gt4_pairs.csv", pair_rows)
    write_csv(out / "gt4_unmatched_rgb.csv", unmatched_rows, ["frame_index_zero_based", "rgb_timestamp", "rgb_path", "reason"])
    write_csv(out / "gt4_unused_gt.csv", unused_rows, ["depth_timestamp", "depth_path", "depth_index_line"])
    summary = {
        "experiment": "GT4 sequence-specific one-to-one RGB-GT timestamp pairing",
        "sequence_name": args.sequence_name,
        "sequence_root": str(root),
        "rgb_index_records": len(rgb),
        "rgb_missing_files": len(missing_rgb),
        "depth_index_records": len(depth),
        "existing_unique_depth_images": len(valid_depths),
        "missing_depth_files": len(missing_depth),
        "duplicate_depth_path_index_rows": duplicate_path_rows,
        "matched_pairs": len(matches),
        "unmatched_rgb_frames": len(unmatched_rows),
        "unused_valid_gt_images": len(unused_rows),
        "max_time_diff_ms": args.max_time_diff_ms,
        "max_observed_abs_delta_ms": max(used_deltas) if used_deltas else None,
        "mean_observed_abs_delta_ms": sum(used_deltas) / len(used_deltas) if used_deltas else None,
        "one_to_one_unique_gt": len({r["depth_path"] for r in pair_rows if r["gt_match_status"] == "MATCHED"}) == len(matches),
        "rgb_timestamp_strictly_increasing": True,
        "gt_timestamp_strictly_increasing_after_matching": all(
            float(pair_rows[i]["depth_timestamp"]) < float(pair_rows[j]["depth_timestamp"])
            for i, j in zip(sorted(matches), sorted(matches)[1:])
        ),
        "runtime_sec": time.time() - t0,
    }
    (out / "gt4_pair_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("Pair CSV:", out / "gt4_pairs.csv", flush=True)
    print("Pair summary:", out / "gt4_pair_summary.json", flush=True)
    if not summary["one_to_one_unique_gt"] or not summary["gt_timestamp_strictly_increasing_after_matching"]:
        raise RuntimeError("配对不变量失败")
    if len(matches) < 9:
        raise RuntimeError("有效配对少于 9 帧，无法形成至少 3 个三帧片段")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"GT4 pair builder FAILED: {exc!r}", file=sys.stderr, flush=True)
        raise
