#!/usr/bin/env python3
"""Whitelist packager for GT4-A2-C2 continuity replay; excludes model checkpoints."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[2]

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/gt4a2_crowd2_continuity")
    args = ap.parse_args()
    out = Path(args.out_dir).resolve()
    archive = ROOT / "outputs/GT4_A2_Crowd2_连续状态复测_review.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)

    output_names = [
        "gt4a2_c2_preflight.json", "gt4a2_c2_preflight.log", "gt4a2_c2_run.log",
        "gt4a2_c2_protocol.json", "gt4a2_c2_short_window_cycle_means.csv",
        "gt4a2_c2_short_window_target_losses.csv", "gt4a2_c2_target_loss_comparison.csv",
        "gt4a2_c2_continuous_frame_metrics.csv", "gt4a2_c2_sequence_summary.csv",
        "gt4a2_c2_frame_offset_summary.csv", "gt4a2_c2_continuity_summary.json",
        "GT4_A2_Crowd2_连续状态复测报告.md",
    ]
    source_paths = [
        ROOT / "experiments/p1/gt4a2_c2_continuity_retest.py",
        ROOT / "experiments/p1/run_gt4a2_c2_continuity_retest.sh",
        ROOT / "experiments/p1/gt4a2_continuity_package_review.py",
        ROOT / "experiments/p1/gt4a2_exp1_train_eval.py",
        ROOT / "experiments/p1/gt2_fixed_clip_repair_train.py",
        ROOT / "velodepth/models/decoder_velo.py",
        ROOT / "velodepth/models/persistent_feature_repair.py",
    ]
    entries, missing = [], []
    for name in output_names:
        p = out / name
        if p.is_file():
            entries.append((p, f"{out.relative_to(ROOT)}/{name}"))
        elif name not in ("gt4a2_c2_run.log",):
            missing.append(f"{out}/{name}")
    for p in source_paths:
        if p.is_file():
            entries.append((p, str(p.relative_to(ROOT))))
        else:
            missing.append(str(p.relative_to(ROOT)))

    summary = out / "gt4a2_c2_continuity_summary.json"
    manifest = {
        "experiment": "GT4-A2-C2 Crowd2 continuous-state replay",
        "status": "COMPLETE" if summary.is_file() and not missing else "INCOMPLETE",
        "missing_files": missing,
        "included_files": [
            {"path": arc, "size_bytes": p.stat().st_size, "sha256": sha(p)}
            for p, arc in entries
        ],
        "excluded": [
            "pretrained model.safetensors", "C/D final checkpoint binaries",
            "full-resolution predictions", "feature caches", "per-pixel arrays",
            "original full dataset files"
        ],
    }
    manifest_path = out / "gt4a2_c2_package_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    entries.append((manifest_path, f"{out.relative_to(ROOT)}/{manifest_path.name}"))

    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p, arc in entries:
            z.write(p, arc)
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"ZIP integrity failed: {bad}")
    print("Package status:", manifest["status"])
    print("Missing files:", len(missing))
    for item in missing:
        print("  MISSING:", item)
    print("Review ZIP:", archive)
    print("Size MiB:", round(archive.stat().st_size / 1024**2, 3))
    print("ZIP integrity: PASS")
    return 0 if summary.is_file() and not missing else 1

if __name__ == "__main__":
    raise SystemExit(main())
