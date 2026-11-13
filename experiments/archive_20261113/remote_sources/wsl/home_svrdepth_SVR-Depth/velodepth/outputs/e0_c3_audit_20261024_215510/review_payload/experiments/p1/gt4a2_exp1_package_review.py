#!/usr/bin/env python3
"""Whitelist packager for GT4-A2 Experiment 1; no recursive output collection."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[2]

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/gt4a2_exp1")
    args = ap.parse_args()
    out = Path(args.out_dir).resolve()
    archive = ROOT / "outputs/GT4_A2_Experiment1_review.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    names = [
        "gt4a2_exp1_preflight.json", "gt4a2_exp1_preflight.log", "gt4a2_exp1_run.log",
        "gt4a2_exp1_config.json", "gt4a2_exp1_anchor_manifest.csv", "gt4a2_exp1_sample_order.csv",
        "gt4a2_exp1_initial_head.pt", "gt4a2_exp1_C_final_step1000.pt", "gt4a2_exp1_D_final_step1000.pt",
        "gt4a2_exp1_C_training_curve.csv", "gt4a2_exp1_D_training_curve.csv",
        "gt4a2_exp1_C_training_summary.json", "gt4a2_exp1_D_training_summary.json",
        "gt4a2_exp1_full_sequence_frame_metrics.csv", "gt4a2_exp1_cycle_summary.csv",
        "gt4a2_exp1_sequence_summary.csv", "gt4a2_exp1_keyframe_checks.csv",
        "gt4a2_exp1_decision.json", "gt4a2_exp1_summary.json", "GT4_A2_Experiment1_report.md",
        "gt4a2_exp1_crowd2_pairs_used.csv", "gt4a2_exp1_crowd3_pairs_used.csv", "gt4a2_exp1_balloon2_pairs_used.csv",
    ]
    source_paths = [
        ROOT / "experiments/p1/gt4a2_exp1_train_eval.py",
        ROOT / "experiments/p1/run_gt4a2_exp1.sh",
        ROOT / "experiments/p1/gt4a2_exp1_package_review.py",
        ROOT / "experiments/p1/gt2_fixed_clip_repair_train.py",
        ROOT / "velodepth/models/decoder_velo.py",
        ROOT / "velodepth/models/persistent_feature_repair.py",
    ]
    entries = []
    missing = []
    for name in names:
        p = out / name
        if p.is_file():
            entries.append((p, f"outputs/gt4a2_exp1/{name}"))
        else:
            missing.append(f"outputs/gt4a2_exp1/{name}")
    for p in source_paths:
        if p.is_file():
            entries.append((p, str(p.relative_to(ROOT))))
        else:
            missing.append(str(p.relative_to(ROOT)))
    manifest = {
        "experiment": "GT4-A2 Experiment 1 compact review bundle",
        "status": "COMPLETE" if (out / "gt4a2_exp1_summary.json").is_file() and not missing else "INCOMPLETE",
        "missing_files": missing,
        "included_files": [{"path": arc, "size_bytes": p.stat().st_size, "sha256": sha(p)} for p, arc in entries],
        "excluded": ["model.safetensors", "full-resolution images", "per-pixel prediction arrays", "feature caches", "temporary files"],
    }
    manifest_path = out / "gt4a2_exp1_package_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    entries.append((manifest_path, "outputs/gt4a2_exp1/gt4a2_exp1_package_manifest.json"))
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p, arc in entries:
            z.write(p, arc)
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"ZIP integrity failed: {bad}")
    print("Bundle status:", manifest["status"])
    print("Missing files:", len(missing))
    if missing:
        for name in missing:
            print("  MISSING:", name)
    print("Archive:", archive)
    print("Archive size MiB:", round(archive.stat().st_size / 1024**2, 2))
    print("ZIP integrity: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
