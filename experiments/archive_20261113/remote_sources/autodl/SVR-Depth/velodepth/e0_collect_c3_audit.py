#!/usr/bin/env python3
"""Collect a compact provenance bundle for SVR-Depth C3 / GT4-A2 runs.

Run from any location:
  python e0_collect_c3_audit.py --root /root/autodl-tmp/SVR-Depth/velodepth

The script does not train, run inference, load checkpoints, or modify source files.
It writes an audit folder and a compact ZIP under <root>/outputs/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

TEXT_EXTS = {".py", ".sh", ".md", ".json", ".log", ".txt", ".csv", ".yaml", ".yml", ".toml"}
CHECKPOINT_EXTS = {".pt", ".pth", ".ckpt"}
SKIP_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".avi", ".npy", ".npz", ".safetensors", ".bin"}
NAME_RE = re.compile(r"c3|gt4[_-]?a2|continuous.*repair|repair.*continuous", re.I)
CONTENT_RE = re.compile(
    r"C3|update_policy|optimizer\s*\.\s*step|steps_per_mode|train_cycles|"
    r"unique.{0,20}target|supervised.{0,20}(frame|target)|propagation.{0,20}frame|"
    r"full.?cycle|gt4.?a2|training.{0,20}summary|checkpoint_sha256",
    re.I,
)
MAX_TEXT_SCAN = 12 * 1024 * 1024
MAX_PACKAGE_FILE = 24 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_TEXT_SCAN or path.suffix.lower() not in TEXT_EXTS:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError):
        return None


def git_info(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, cmd in {
        "git_root": ["git", "rev-parse", "--show-toplevel"],
        "git_commit": ["git", "rev-parse", "HEAD"],
        "git_status_short": ["git", "status", "--short"],
    }.items():
        try:
            p = subprocess.run(cmd, cwd=root, text=True, capture_output=True, timeout=5)
            result[key] = (p.stdout.strip() if p.returncode == 0 else "NOT_AVAILABLE")
        except Exception:
            result[key] = "NOT_AVAILABLE"
    return result


def json_summary(path: Path) -> str:
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return ""
    fields = []
    wanted = {
        "experiment", "status", "verdict", "update_policy", "steps_per_mode", "steps",
        "optimizer_steps", "train_cycles", "refresh_interval", "future_frames_per_anchor",
        "unique_gt_targets", "unique_target_count", "frame_exposures", "supervised_frame_exposures",
        "checkpoint", "checkpoint_sha256", "script_sha256", "source_hashes", "protocol",
    }
    def walk(x: Any, prefix: str = ""):
        if isinstance(x, dict):
            for k, v in x.items():
                pathkey = f"{prefix}.{k}" if prefix else k
                if k in wanted:
                    val = json.dumps(v, ensure_ascii=False, sort_keys=True)
                    fields.append(f"{pathkey}={val[:600]}")
                if isinstance(v, (dict, list)):
                    walk(v, pathkey)
        elif isinstance(x, list):
            for i, v in enumerate(x[:20]):
                if isinstance(v, (dict, list)):
                    walk(v, f"{prefix}[{i}]")
    walk(obj)
    return " | ".join(fields)


def csv_summary(path: Path) -> str:
    try:
        if path.stat().st_size > MAX_TEXT_SCAN:
            return "CSV_TOO_LARGE_TO_PARSE"
        with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not reader.fieldnames:
            return f"rows={len(rows)} no_header"
        bits = [f"rows={len(rows)}", "columns=" + ",".join(reader.fieldnames[:30])]
        for key in ("step", "optimizer_step", "sample_id", "anchor_index", "refresh_cycle", "update_policy"):
            if key in reader.fieldnames:
                vals = [r.get(key, "") for r in rows]
                nonempty = [v for v in vals if v != ""]
                if nonempty:
                    bits.append(f"unique_{key}={len(set(nonempty))}")
                    if key in {"step", "optimizer_step"}:
                        try:
                            nums = [int(float(v)) for v in nonempty]
                            bits.append(f"{key}_minmax={min(nums)}..{max(nums)}")
                        except Exception:
                            pass
        return " ; ".join(bits)
    except Exception as e:
        return f"CSV_PARSE_ERROR={type(e).__name__}:{e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="VeloDepth repository root")
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    if not (root / "experiments" / "p1").is_dir() or not (root / "outputs").is_dir():
        print(f"ERROR: {root} does not look like the VeloDepth project root (need experiments/p1 and outputs).", file=sys.stderr)
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "outputs" / f"e0_c3_audit_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    payload = out_dir / "review_payload"
    payload.mkdir()
    zip_path = root / "outputs" / f"e0_c3_audit_{stamp}_review.zip"

    scan_roots = [root / "experiments" / "p1", root / "outputs"]
    # Include top-level project notes, but not archives or arbitrary data.
    root_notes = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".md", ".txt", ".json"}]
    files: list[Path] = []
    for base in scan_roots:
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if out_dir in p.parents or p == zip_path:
                continue
            if p.suffix.lower() in TEXT_EXTS or p.suffix.lower() in CHECKPOINT_EXTS:
                files.append(p)
    files.extend(root_notes)
    files = sorted(set(files))

    # Candidate directories: explicitly named C3/GT4-A2 folders, plus parents of text files with protocol hits.
    candidate_dirs: set[Path] = set()
    for base in scan_roots:
        for d in [base, *[x for x in base.rglob("*") if x.is_dir()]]:
            if out_dir in d.parents or d == out_dir:
                continue
            if NAME_RE.search(str(d.relative_to(root))):
                candidate_dirs.add(d)

    hits: dict[Path, list[tuple[int, str]]] = {}
    candidates: set[Path] = set()
    for p in files:
        rel = str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
        name_match = bool(NAME_RE.search(rel))
        text = safe_read_text(p)
        line_hits: list[tuple[int, str]] = []
        if text is not None:
            for i, line in enumerate(text.splitlines(), 1):
                if CONTENT_RE.search(line):
                    line_hits.append((i, line[:500]))
                    if len(line_hits) >= 120:
                        break
        if name_match or line_hits:
            candidates.add(p)
            if line_hits:
                hits[p] = line_hits
                if p.is_relative_to(root / "outputs"):
                    candidate_dirs.add(p.parent)

    # Expand named/content-hit output directories by one directory tree, while excluding large raw assets.
    expanded: set[Path] = set(candidates)
    for d in list(candidate_dirs):
        if not d.exists() or not d.is_dir():
            continue
        for p in d.rglob("*"):
            if not p.is_file() or out_dir in p.parents or p == zip_path:
                continue
            if p.suffix.lower() in SKIP_EXTS:
                continue
            if p.suffix.lower() in TEXT_EXTS or p.suffix.lower() in CHECKPOINT_EXTS:
                expanded.add(p)

    # Avoid packaging generic non-C3 reports merely because they contain the letters C3 in unrelated text.
    # Keep all source hits and named GT4-A2/C3 paths; keep output hit files and their nearby run artifacts.
    selected: set[Path] = set()
    for p in expanded:
        rel = str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
        if p.is_relative_to(root / "experiments" / "p1"):
            selected.add(p)
        elif NAME_RE.search(rel):
            selected.add(p)
        elif p.is_relative_to(root / "outputs") and (p in candidates or any(p.is_relative_to(d) for d in candidate_dirs)):
            selected.add(p)
        elif p in root_notes and (p in candidates or hits.get(p)):
            selected.add(p)

    inventory_rows = []
    copied = []
    for p in sorted(selected):
        try:
            st = p.stat()
            digest = sha256_file(p) if st.st_size <= 2 * 1024**3 else "SKIPPED_GT2GB"
            rel = str(p.relative_to(root))
            content = safe_read_text(p)
            tokens = []
            if NAME_RE.search(rel): tokens.append("name_match")
            if p in hits: tokens.append("content_match")
            detail = ""
            if content is not None and p.suffix.lower() == ".json":
                detail = json_summary(p)
            elif content is not None and p.suffix.lower() == ".csv":
                detail = csv_summary(p)
            inventory_rows.append({
                "relative_path": rel,
                "size_bytes": st.st_size,
                "mtime_local": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                "sha256": digest,
                "match_reason": ";".join(tokens),
                "summary": detail[:2500],
            })
            # Include small text evidence and small repair-head checkpoints, not raw data/prediction arrays.
            should_copy = p.suffix.lower() in TEXT_EXTS or (p.suffix.lower() in CHECKPOINT_EXTS and st.st_size <= MAX_PACKAGE_FILE)
            if should_copy and st.st_size <= MAX_PACKAGE_FILE:
                dst = payload / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
                copied.append(rel)
        except OSError as e:
            inventory_rows.append({"relative_path": str(p), "size_bytes": "ERROR", "mtime_local": "", "sha256": "", "match_reason": "ERROR", "summary": str(e)})

    inventory_path = out_dir / "inventory.tsv"
    with inventory_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "size_bytes", "mtime_local", "sha256", "match_reason", "summary"], delimiter="\t")
        writer.writeheader(); writer.writerows(inventory_rows)

    hits_path = out_dir / "protocol_hits.txt"
    with hits_path.open("w", encoding="utf-8") as f:
        f.write(f"ROOT: {root}\nGenerated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        for p in sorted(hits):
            f.write(f"=== {p.relative_to(root) if p.is_relative_to(root) else p} ===\n")
            for lineno, line in hits[p]:
                f.write(f"{lineno}: {line}\n")
            f.write("\n")

    git = git_info(root)
    c3_named = [str(p.relative_to(root)) for p in selected if NAME_RE.search(str(p.relative_to(root)))]
    update_policy_hits = [str(p.relative_to(root)) for p in hits if re.search(r"update_policy", safe_read_text(p) or "", re.I)]
    optimizer_hits = [str(p.relative_to(root)) for p in hits if re.search(r"optimizer\s*\.\s*step", safe_read_text(p) or "", re.I)]
    report = [
        "# E0 C3 provenance collection",
        "",
        f"- Project root: `{root}`",
        f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Candidate files inventoried: **{len(inventory_rows)}**",
        f"- Evidence files copied into review payload: **{len(copied)}**",
        f"- Paths with C3/GT4-A2-like names: **{len(c3_named)}**",
        f"- Files with `update_policy` hits: **{len(update_policy_hits)}**",
        f"- Files with `optimizer.step` hits: **{len(optimizer_hits)}**",
        "",
        "## Git provenance",
    ]
    report.extend([f"- {k}: `{v.replace(chr(10), ' | ')[:1000]}`" for k, v in git.items()])
    report += ["", "## Named C3 / GT4-A2 paths"]
    report.extend([f"- `{p}`" for p in c3_named] or ["- None found by filename."])
    report += ["", "## `update_policy` matches"]
    report.extend([f"- `{p}`" for p in update_policy_hits] or ["- None found in scanned source/text files."])
    report += ["", "## `optimizer.step` matches"]
    report.extend([f"- `{p}`" for p in optimizer_hits] or ["- None found in scanned source/text files."])
    report += [
        "",
        "## Important interpretation",
        "This collector does not run training or inference. It inventories source/config/log/CSV/checkpoint provenance so the C3 protocol can be identified without guessing from filenames. Compare script SHA256, actual optimizer-step records, cycle/frame coverage, unique supervised targets, checkpoint hashes, evaluation CSVs, refresh interval, and reset behavior before treating any C3 result as verified.",
        "",
        "## Files",
        "- `inventory.tsv`: paths, sizes, mtimes, SHA256, and JSON/CSV summaries.",
        "- `protocol_hits.txt`: lines matching C3/update-policy/training protocol terms.",
        "- `review_payload/`: selected small text evidence and small repair-head checkpoints; no RGB/GT/prediction arrays are copied.",
    ]
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in out_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(out_dir.parent))
    print("E0 audit collection complete (no training/inference was run).")
    print(f"Report: {out_dir / 'report.md'}")
    print(f"Inventory: {inventory_path}")
    print(f"Review ZIP: {zip_path}")
    print("Upload the Review ZIP back for the next provenance check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
