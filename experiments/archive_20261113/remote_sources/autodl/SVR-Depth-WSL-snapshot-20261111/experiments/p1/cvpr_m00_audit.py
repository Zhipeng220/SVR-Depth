#!/usr/bin/env python3
"""CVPR M00 read-only audit for SVR-Depth.

This tool does not import torch, initialize CUDA, train, run inference, modify
model weights, or touch source files. It inventories the frozen P_multi setup
and emits an auditable report for deciding whether M05 development can start.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()

PINNED_SOURCES = {
    "experiments/p1/gt4a2_y04_fair_multiscene_fix2.py":
        "4972e676e8f4b7c867615546eac34dcebce96c6b3e8cc0b5ad51e91fce14d291",
    "experiments/p1/gt4a2_exp1_train_eval.py":
        "c24441cb405465c1a16a86127b51aade8f9af69b5d368f43d830d0e759ee3d7f",
    "experiments/p1/gt2_fixed_clip_repair_train.py":
        "bf7cd2021b1d0bbf66d75502dba6afed87f4c8ff3ef58ff7b18f56ac28044192",
    "experiments/p1/gt4a2_c3_closed_loop.py":
        "76327208acac4543b2a87ccba8b7af435e4c5ce65267b42052e51b435cbad7f5",
    "velodepth/models/velodepth.py":
        "739c5f27a749174987b9a36edaac32fb98070e87dbef0a0a52f7430d61149479",
    "velodepth/models/encoder_velo.py":
        "3c5aa2a3a6dce03b68373953037fe8a0313a2cab36d66e5e3b88a56ec4a7ba8a",
    "velodepth/models/decoder_velo.py":
        "35ca12629accd1a2a77595f505165fa17dffab07c763a8b639604285b94bdb08",
    "velodepth/models/persistent_feature_repair.py":
        "6dfc8b0c5565ea9139b8c266c5ef337b3309ce0fde2f365428fc44390b84117c",
}

EXPECTED_P_CHECKPOINT_SHA256 = (
    "77a86cdf523f776608b6b591d6c6552002b90398e85d00d3a3fe29afe14a0a24"
)

PAIR_CSVS = {
    "crowd2": "outputs/gt4a2_exp1/gt4a2_exp1_crowd2_pairs_used.csv",
    "crowd3": "outputs/gt4a2_exp1/gt4a2_exp1_crowd3_pairs_used.csv",
    "balloon2": "outputs/gt4a2_exp1/gt4a2_exp1_balloon2_pairs_used.csv",
}
DATASET_PROTOCOL = "outputs/y11_confirmation_data/dataset_protocol.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cmd_output(cmd: list[str], timeout: int = 15) -> dict:
    try:
        p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True,
                           timeout=timeout, check=False)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(),
                "stderr": p.stderr.strip()}
    except Exception as e:
        return {"returncode": None, "stdout": "", "stderr": repr(e)}


def inspect_git() -> dict:
    return {
        "root": cmd_output(["git", "rev-parse", "--show-toplevel"]),
        "commit": cmd_output(["git", "rev-parse", "HEAD"]),
        "branch": cmd_output(["git", "branch", "--show-current"]),
        "status": cmd_output(["git", "status", "--short"]),
    }


def inspect_sources() -> list[dict]:
    rows = []
    for rel, expected in PINNED_SOURCES.items():
        p = ROOT / rel
        exists = p.is_file()
        actual = sha256(p) if exists else ""
        rows.append({
            "path": rel, "exists": exists, "actual_sha256": actual,
            "expected_sha256": expected,
            "match": (actual == expected) if exists else False,
            "note": "" if not exists or actual == expected else
                    "source differs from the pinned audit snapshot; inspect before M05",
        })
    return rows


def inspect_pair_csvs() -> list[dict]:
    rows = []
    for seq, rel in PAIR_CSVS.items():
        p = ROOT / rel
        item = {"sequence": seq, "path": rel, "exists": p.is_file(),
                "bytes": p.stat().st_size if p.is_file() else 0,
                "sha256": sha256(p) if p.is_file() else "",
                "rows": None, "header": []}
        if p.is_file():
            try:
                with p.open("r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.reader(f)
                    item["header"] = next(reader, [])
                    item["rows"] = sum(1 for _ in reader)
            except Exception as e:
                item["read_error"] = repr(e)
        rows.append(item)
    return rows


def flatten_strings(obj, prefix=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(flatten_strings(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(flatten_strings(v, f"{prefix}[{i}]"))
    elif isinstance(obj, str):
        out.append({"key": prefix, "value": obj})
    return out


def inspect_dataset_protocol() -> dict:
    p = ROOT / DATASET_PROTOCOL
    if not p.is_file():
        return {"path": DATASET_PROTOCOL, "exists": False}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        strings = flatten_strings(obj)
        return {
            "path": DATASET_PROTOCOL, "exists": True,
            "bytes": p.stat().st_size, "sha256": sha256(p),
            "top_level_keys": list(obj.keys()) if isinstance(obj, dict) else [],
            "string_entries": strings[:300],
            "parse_error": None,
        }
    except Exception as e:
        return {"path": DATASET_PROTOCOL, "exists": True,
                "bytes": p.stat().st_size, "sha256": sha256(p),
                "parse_error": repr(e)}


def checkpoint_candidates(explicit: str | None) -> list[dict]:
    candidates: list[Path] = []
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = ROOT / p
        if p.is_file():
            candidates.append(p.resolve())
        else:
            return [{"path": str(p), "exists": False, "error": "explicit checkpoint not found"}]

    if not candidates:
        roots = [
            ROOT / "outputs/gt4a2_y04_validation_fix3_20261025_225902",
            ROOT / "outputs",
        ]
        seen = set()
        for base in roots:
            if not base.exists():
                continue
            for p in base.rglob("*"):
                if not p.is_file() or p.suffix.lower() not in (".pt", ".pth", ".ckpt"):
                    continue
                rel = p.relative_to(ROOT).as_posix()
                # Keep the first root restricted to Y04-Fix3; broad outputs scan
                # only for filenames that look like final P checkpoints.
                if base.name == "outputs" and not any(
                    token in p.name.lower() for token in ("p_multi", "p-multi", "persistent", "final_step1000")
                ):
                    continue
                if rel not in seen:
                    candidates.append(p)
                    seen.add(rel)
                if len(candidates) >= 30:
                    break
            if len(candidates) >= 30:
                break

    rows = []
    for p in candidates[:30]:
        try:
            actual = sha256(p)
            rows.append({
                "path": p.relative_to(ROOT).as_posix(),
                "exists": True, "bytes": p.stat().st_size,
                "sha256": actual,
                "matches_expected_P_sha256": actual == EXPECTED_P_CHECKPOINT_SHA256,
            })
        except Exception as e:
            rows.append({"path": str(p), "exists": True, "error": repr(e)})
    return rows


def inspect_runtime() -> dict:
    nvidia = cmd_output(["nvidia-smi"], timeout=10)
    snapshot = os.environ.get("VDEEP_SNAPSHOT", "")
    snapshot_info = {"env": snapshot, "exists": bool(snapshot)}
    if snapshot:
        p = Path(snapshot).expanduser()
        snapshot_info.update({
            "resolved": str(p.resolve()),
            "config_json": (p / "config.json").is_file(),
            "model_safetensors": (p / "model.safetensors").is_file(),
        })
        if (p / "model.safetensors").is_file():
            snapshot_info["model_safetensors_sha256"] = sha256(p / "model.safetensors")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(ROOT),
        "torch_spec_found_without_import": importlib.util.find_spec("torch") is not None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi": nvidia,
        "VDEEP_SNAPSHOT": snapshot_info,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default=os.environ.get("SVRDEPTH_P_CHECKPOINT", ""))
    args = ap.parse_args()

    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=False)

    runtime = inspect_runtime()
    git = inspect_git()
    sources = inspect_sources()
    pairs = inspect_pair_csvs()
    protocol = inspect_dataset_protocol()
    checkpoints = checkpoint_candidates(args.checkpoint)

    source_exists = all(x["exists"] for x in sources)
    source_match = all(x["match"] for x in sources)
    pair_complete = all(x["exists"] and (x.get("rows") or 0) > 0 for x in pairs)
    protocol_ok = protocol.get("exists", False) and not protocol.get("parse_error")
    checkpoint_match = any(x.get("matches_expected_P_sha256") for x in checkpoints)
    snapshot = runtime["VDEEP_SNAPSHOT"]
    snapshot_ok = snapshot.get("config_json", False) and snapshot.get("model_safetensors", False)

    checks = {
        "repository_root": (ROOT / "velodepth" / "models").is_dir(),
        "all_pinned_sources_exist": source_exists,
        "all_pinned_sources_match_reference": source_match,
        "three_training_pair_manifests_available": pair_complete,
        "Y11_dataset_protocol_readable": protocol_ok,
        "P_multi_checkpoint_sha256_verified": checkpoint_match,
        "VeloDepth_snapshot_complete": snapshot_ok,
        "torch_package_present": runtime["torch_spec_found_without_import"],
    }
    # Audit is intentionally non-destructive and never launches training/inference.
    # READY means M00 has enough provenance to begin a developer M05 canary;
    # source mismatches remain a hard review item even if files exist.
    ready = all([
        checks["repository_root"], checks["all_pinned_sources_exist"],
        checks["all_pinned_sources_match_reference"],
        checks["three_training_pair_manifests_available"],
        checks["Y11_dataset_protocol_readable"],
        checks["P_multi_checkpoint_sha256_verified"],
        checks["VeloDepth_snapshot_complete"],
        checks["torch_package_present"],
    ])
    report = {
        "experiment": "CVPR-M00-read-only-audit",
        "created_at": datetime.now().astimezone().isoformat(),
        "read_only": True,
        "training_performed": False,
        "model_inference_performed": False,
        "cuda_initialized_by_audit": False,
        "expected_P_multi_checkpoint_sha256": EXPECTED_P_CHECKPOINT_SHA256,
        "checks": checks,
        "readiness": "READY_FOR_M05_DEV_CANARY" if ready else "INCOMPLETE_REQUIRES_REVIEW",
        "runtime": runtime,
        "git": git,
        "sources": sources,
        "pair_manifests": pairs,
        "dataset_protocol": protocol,
        "P_multi_checkpoint_candidates": checkpoints,
        "notes": [
            "Source hash mismatch is reported, not overwritten or repaired.",
            "The expected P checkpoint hash is taken from CVPR plan v2.1.",
            "Y09/Y11/N-series recordings already inspected remain development data.",
            "This audit does not certify independent test data or statistical power.",
        ],
    }

    (out / "m00_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(out / "source_hash_audit.csv", sources)
    write_csv(out / "pair_manifest_audit.csv", pairs)
    write_csv(out / "checkpoint_candidates.csv", checkpoints)

    lines = [
        "# CVPR M00 Read-only Audit",
        "",
        f"- Created: {report['created_at']}",
        f"- Readiness: **{report['readiness']}**",
        "- Training: NOT RUN",
        "- Model inference: NOT RUN",
        "- CUDA initialization by audit: NOT RUN",
        "",
        "## Checks",
        "",
    ]
    for k, v in checks.items():
        lines.append(f"- {'PASS' if v else 'REVIEW'} — `{k}`")
    lines += [
        "",
        "## Required next action",
        "",
        "Do not start M05 formal runs until the P checkpoint hash, source identity, "
        "dataset protocol, and state-field semantics have been reconciled.",
        "",
        "## Files",
        "",
        "- `m00_audit.json`",
        "- `source_hash_audit.csv`",
        "- `pair_manifest_audit.csv`",
        "- `checkpoint_candidates.csv`",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    review_zip = out / "cvpr_m00_review.zip"
    with zipfile.ZipFile(review_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in ("m00_audit.json", "source_hash_audit.csv",
                     "pair_manifest_audit.csv", "checkpoint_candidates.csv", "REPORT.md"):
            z.write(out / name, name)
    with zipfile.ZipFile(review_zip) as z:
        if z.testzip() is not None:
            raise RuntimeError("M00 review ZIP CRC verification failed")

    print(f"[M00] readiness={report['readiness']}", flush=True)
    for k, v in checks.items():
        print(f"[M00] {'PASS' if v else 'REVIEW'} {k}", flush=True)
    print(f"[M00] report={out / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
