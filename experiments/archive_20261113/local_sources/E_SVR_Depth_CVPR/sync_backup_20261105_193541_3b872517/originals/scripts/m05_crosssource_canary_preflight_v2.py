#!/usr/bin/env python3
"""Read-only preflight for adapting M05 to a CrossSource-v2 checkpoint.

This script performs no model inference, training, data writes, or checkpoint writes.
It verifies the E4-C Best checkpoint, pinned VeloDepth snapshot/code, M05 development
pair manifests, helper APIs, CUDA availability, and deterministic canary cycles.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import inspect
import os
import sys
from pathlib import Path

EXPECTED_BASE_SHA = "615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e"
EXPECTED_CHECKPOINT_SHA = "0447fbcc25d1ea93d5a80e7f8a7a3b38ac911a8465951009da53988e712754c0"
EXPECTED_INIT_HEAD_SHA = "fbb01f8d81da722fc73c34dff7b833800d32270f1ca3d7616830ce869c77c16f"
EXPECTED_PARAMS = 170496

PAIR_FILES = {
    "crowd2": ("gt4a2_exp1_crowd2_pairs_used.csv", "f7bf9226edcedf6c05d63938e29f92bf9364878c9291dbd79172c973c296f350"),
    "crowd3": ("gt4a2_exp1_crowd3_pairs_used.csv", "9f4da8360b8f6b3039330dc7fe9dba6dd7219d9427e774b89006b09269fa17ad"),
    "balloon2": ("gt4a2_exp1_balloon2_pairs_used.csv", "afbd46ebed192aaf69f580822993fb7c756c5656de0595f68143993859918b06"),
}

EXPECTED_SOURCE_SHA = {
    "experiments/crosssource_v2/legacy_helpers/gt2_fixed_clip_repair_train.py": "50a8da294c87ff5ee43131d0c04fddd9227fa0ea76041a354d86c9f703a267b3",  # intentional NumPy writable-array fix used by E4-C
    "experiments/crosssource_v2/legacy_helpers/gt4a2_exp1_train_eval.py": "911da6b1a8e0bfa21471d86816de823ababf561ba8a5b20df52d0d51a97dfd83",
    "velodepth/models/velodepth.py": "739c5f27a749174987b9a36edaac32fb98070e87dbef0a0a52f7430d61149479",
    "velodepth/models/decoder_velo.py": "35ca12629accd1a2a77595f505165fa17dffab07c763a8b639604285b94bdb08",
    "velodepth/models/encoder_velo.py": "3c5aa2a3a6dce03b68373953037fe8a0313a2cab36d66e5e3b88a56ec4a7ba8a",
    "velodepth/models/persistent_feature_repair.py": "6dfc8b0c5565ea9139b8c266c5ef337b3309ce0fde2f365428fc44390b84117c",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def locate_project_root(start: Path) -> Path:
    candidates = [start, *start.parents]
    for p in start.rglob("train_crosssource_v2.py"):
        if p.parent.name == "scripts":
            candidates.append(p.parent.parent)
    seen = set()
    for root in candidates:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        if (root / "scripts/train_crosssource_v2.py").is_file() and (
            root / "experiments/crosssource_v2/legacy_helpers/gt2_fixed_clip_repair_train.py"
        ).is_file():
            return root
    raise FileNotFoundError(
        "Cannot locate project root containing scripts/train_crosssource_v2.py and "
        "experiments/crosssource_v2/legacy_helpers. Run from the CrossSource-v2 repo root."
    )


def locate_pair_root(project_root: Path) -> Path:
    candidates = []
    env = os.environ.get("M05_PAIR_ROOT", "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    candidates += [
        project_root / "outputs/gt4a2_exp1",
        project_root / "outputs/legacy_runtime/outputs/gt4a2_exp1",
        Path("/root/autodl-tmp/SVR-Depth/velodepth/outputs/gt4a2_exp1"),
        Path("/root/autodl-tmp/SVR-Depth-CVPR/outputs/legacy_runtime/outputs/gt4a2_exp1"),
    ]
    for root in candidates:
        root = root.expanduser().resolve()
        if all((root / name).is_file() for name, _ in PAIR_FILES.values()):
            return root
    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Cannot find all three locked M05 pair CSVs. Set M05_PAIR_ROOT to their common "
        f"directory. Tried:\n{tried}"
    )


def locate_snapshot() -> Path:
    env = os.environ.get("VDEEP_SNAPSHOT", "").strip()
    candidates = [Path(env).expanduser()] if env else []
    snapshot_name = "2b2ee9fe44d0b261e736653042cecd7693f99c8c"
    hf_roots = [
        Path(os.environ.get("HF_HOME", "")).expanduser() if os.environ.get("HF_HOME") else None,
        Path.home() / "hf_cache",
        Path.home() / ".cache/huggingface",
        Path("/root/autodl-tmp/hf_cache"),
    ]
    for hf in hf_roots:
        if hf is not None:
            candidates.append(hf / "hub/models--lpiccinelli--velodepth/snapshots" / snapshot_name)
    for p in candidates:
        p = p.resolve()
        weights = p / "model.safetensors"
        if weights.is_file() and sha256(weights) == EXPECTED_BASE_SHA:
            return p
    raise FileNotFoundError(
        "Cannot locate the pinned VeloDepth snapshot with the expected model.safetensors SHA. "
        "Set VDEEP_SNAPSHOT to the exact snapshot directory."
    )


def main() -> int:
    import torch

    cwd = Path.cwd().resolve()
    project = locate_project_root(cwd)
    helper_dir = project / "experiments/crosssource_v2/legacy_helpers"
    checkpoint = project / "outputs/crosssource_v2_E4_20261104_C_s20261103_1000step/best_checkpoint.pt"

    print("========== M05 CROSSOURCE CANARY PREFLIGHT ==========")
    print("Project root:", project)

    # Pin the active helper implementations before importing them.
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(helper_dir))
    for rel, expected in EXPECTED_SOURCE_SHA.items():
        path = project / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"SOURCE HASH BLOCK: {rel}: {actual} != {expected}")
        print(f"[PASS] source SHA {rel}: {actual}")

    if not checkpoint.is_file():
        raise FileNotFoundError(f"E4-C Best checkpoint missing: {checkpoint}")
    cp_sha = sha256(checkpoint)
    if cp_sha != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError(f"E4-C Best checkpoint SHA mismatch: {cp_sha}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity = ckpt.get("identity", {})
    # The E4-C checkpoint records the exact CrossSource helper provenance.
    # In particular, gt2_fixed_clip_repair_train.py includes the one-line
    # writable NumPy copy fix (SHA 50a8...), not the pre-fix archive version.
    recorded_sources = identity.get("source_sha256", {})
    for rel in (
        "experiments/crosssource_v2/legacy_helpers/gt2_fixed_clip_repair_train.py",
        "experiments/crosssource_v2/legacy_helpers/gt4a2_exp1_train_eval.py",
    ):
        recorded = recorded_sources.get(rel)
        expected = EXPECTED_SOURCE_SHA[rel]
        if recorded != expected:
            raise RuntimeError(
                f"Checkpoint source provenance mismatch for {rel}: "
                f"checkpoint={recorded!r}, expected={expected!r}"
            )
    print("[PASS] checkpoint records the same CrossSource helper hashes")

    expected_identity = {
        "protocol": "SVR-Depth-CrossSource-v2",
        "method": "P_multi",
        "seed": 20261103,
        "initialization_seed": 20261103,
        "steps": 1000,
        "window_frames": 20,
        "tbptt_frames": 4,
        "snapshot_sha256": EXPECTED_BASE_SHA,
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise RuntimeError(f"Checkpoint identity mismatch: {key}={identity.get(key)!r}")
    if int(ckpt.get("step", -1)) != 700:
        raise RuntimeError(f"Expected E4-C Best step 700, got {ckpt.get('step')}")
    if ckpt.get("init_head_sha256") != EXPECTED_INIT_HEAD_SHA:
        raise RuntimeError("E4-C Best initialization hash differs from M00 audit")
    head_state = ckpt.get("head_state")
    if not isinstance(head_state, dict) or not head_state:
        raise RuntimeError("CrossSource checkpoint is missing head_state")
    nparams = sum(v.numel() for v in head_state.values() if isinstance(v, torch.Tensor))
    if nparams != EXPECTED_PARAMS:
        raise RuntimeError(f"Repair-head parameter count mismatch: {nparams}")
    print("[PASS] E4-C Best checkpoint identity / SHA / step / init hash / head_state")
    print("  checkpoint SHA:", cp_sha)
    print("  init_head_sha256:", ckpt["init_head_sha256"])
    print("  head parameters:", nparams)

    snapshot = locate_snapshot()
    snapshot_sha = sha256(snapshot / "model.safetensors")
    if snapshot_sha != EXPECTED_BASE_SHA:
        raise RuntimeError("Pinned snapshot SHA mismatch")
    print("[PASS] VeloDepth snapshot:", snapshot)

    # Import CrossSource-v2's helper pair. Import gt2 first so exp1 reuses this exact module.
    gt2 = importlib.import_module("gt2_fixed_clip_repair_train")
    exp1 = importlib.import_module("gt4a2_exp1_train_eval")
    if Path(gt2.__file__).resolve() != (helper_dir / "gt2_fixed_clip_repair_train.py").resolve():
        raise RuntimeError(f"Wrong gt2 helper imported: {gt2.__file__}")
    if Path(exp1.__file__).resolve() != (helper_dir / "gt4a2_exp1_train_eval.py").resolve():
        raise RuntimeError(f"Wrong exp1 helper imported: {exp1.__file__}")
    print("[PASS] helper imports:", gt2.__file__)
    print("[PASS] helper imports:", exp1.__file__)

    fast_sig = inspect.signature(gt2.fast_step)
    one_sig = inspect.signature(exp1.one_fast_step)
    if "confidence" in fast_sig.parameters or "confidence_state" in fast_sig.parameters:
        raise RuntimeError(f"Unexpected confidence input in fast_step: {fast_sig}")
    one_src = inspect.getsource(exp1.one_fast_step)
    if "gt2.fast_step(model, context, 1, radius, features)" not in one_src.replace("\n", " "):
        compact = " ".join(one_src.split())
        if "gt2.fast_step(model, context, 1, radius, features)" not in compact:
            raise RuntimeError("one_fast_step does not call the audited fast_step signature")
    print("[PASS] state I/O API:")
    print("  fast_step", fast_sig)
    print("  one_fast_step", one_sig)
    print("  incoming confidence is not consumed by fixed fast_step")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; preflight cannot approve a CUDA M05 canary")
    print("[PASS] CUDA:", torch.cuda.get_device_name(0))

    pair_root = locate_pair_root(project)
    print("Pair manifest root:", pair_root)
    eligible_by_sequence = {}
    for sequence, (filename, expected_hash) in PAIR_FILES.items():
        path = pair_root / filename
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Pair manifest hash mismatch: {filename}: {actual_hash}")
        rows = exp1.read_manifest(path, training=False)
        n_full = len(rows) // 16
        eligible = []
        for cycle in range(n_full):
            start = cycle * 16
            targets = [start + age + 4 for age in (1, 4, 8)]
            if all(
                str(rows[i].get("gt_match_status", "")).strip().upper() == "MATCHED"
                for i in targets
            ):
                eligible.append(cycle)
        if not eligible:
            raise RuntimeError(f"No eligible K16 cycle in {sequence}")
        chosen = eligible[len(eligible) // 2]  # deterministic median eligible cycle
        eligible_by_sequence[sequence] = {
            "frames": len(rows),
            "full_K16_cycles": n_full,
            "eligible_cycles": len(eligible),
            "canary_cycle": chosen,
            "anchor_indices": [chosen * 16 + age for age in (1, 4, 8)],
            "H4_GT_indices": [chosen * 16 + age + 4 for age in (1, 4, 8)],
        }
        print(
            f"[PASS] {sequence}: frames={len(rows)} full_cycles={n_full} "
            f"eligible={len(eligible)} selected_cycle={chosen}"
        )

    print("\n[M05 CROSSOURCE CANARY PREFLIGHT PASS]")
    print("No training performed; no model inference performed; no files written.")
    print("Selected development cycles:")
    for name, detail in eligible_by_sequence.items():
        print(f"  {name}: {detail}")
    print("Next: run a single-checkpoint, development-only M05 Canary; do NOT launch formal M05 or Full4 yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
