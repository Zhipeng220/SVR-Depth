#!/usr/bin/env python3
"""Package GT4 outputs, logs, checkpoints, and exact source files for review."""
from __future__ import annotations
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/gt4_validation"
ARCHIVE = ROOT / "outputs/GT4_review.zip"


def sha256(p: Path):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    required = []
    for seq in ("crowd3", "balloon2"):
        d = OUT / seq
        required += [d / n for n in (
            "gt4_pair_build.log", "gt4_pairs.csv", "gt4_pair_summary.json",
            "gt4_run.log", "gt4_config.json", "gt4_summary.json",
            "gt4_frame_metrics.csv", "gt4_branch_checks.csv", "GT4_report.md",
            "gt4_pair_csv_used.csv",
        )]
    source_files = [
        ROOT / "experiments/p1/gt4_build_sequence_pairs.py",
        ROOT / "experiments/p1/gt4_cross_sequence_eval.py",
        ROOT / "experiments/p1/run_gt4.sh",
        ROOT / "experiments/p1/gt4_package_review.py",
        ROOT / "experiments/p1/gt3_heldout_eval.py",
        ROOT / "experiments/p1/gt2_fixed_clip_repair_train.py",
        ROOT / "velodepth/models/decoder_velo.py",
        ROOT / "velodepth/models/persistent_feature_repair.py",
        ROOT / "outputs/gt2_validation/gt2_summary.json",
        ROOT / "outputs/gt2_validation/gt2_C_repair_head.pt",
        ROOT / "outputs/gt2_validation/gt2_D_repair_head.pt",
        ROOT / "outputs/gt1a1_validation/gt1a1_pairs.csv",
    ]
    missing = [str(p) for p in required + source_files if not p.is_file()]
    if not OUT.exists():
        OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file() and p.resolve() != ARCHIVE.resolve():
                arc = str(p.relative_to(ROOT))
                z.write(p, arcname=arc)
                entries.append({"path": arc, "size": p.stat().st_size, "sha256": sha256(p)})
        for p in source_files:
            if p.is_file():
                arc = str(p.relative_to(ROOT))
                if not any(x["path"] == arc for x in entries):
                    z.write(p, arcname=arc)
                    entries.append({"path": arc, "size": p.stat().st_size, "sha256": sha256(p)})
        manifest = {
            "experiment": "GT4 cross-sequence evaluation review bundle",
            "complete": not missing,
            "missing_required_files": missing,
            "files": entries,
        }
        z.writestr("outputs/gt4_validation/GT4_review_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    with zipfile.ZipFile(ARCHIVE) as z:
        bad = z.testzip()
    print("Archive:", ARCHIVE)
    print("Size bytes:", ARCHIVE.stat().st_size)
    print("ZIP integrity:", "PASS" if bad is None else f"FAIL: {bad}")
    print("Files included:", len(entries) + 1)
    if missing:
        print("PACKAGE STATUS: INCOMPLETE (existing logs were still packaged)")
        for p in missing:
            print("MISSING:", p)
    else:
        print("PACKAGE STATUS: COMPLETE")

if __name__ == "__main__":
    main()
