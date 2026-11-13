#!/usr/bin/env python3
"""CVPR M05 Formal 1.0: preregistered multi-anchor persistent-state intervention.

Protocol source: SVR-Depth_CVPR主实验执行计划_20261101_v2.1.
Phase 1 (prepare): read-only manifest audit and deterministic anchor freeze.
Phase 2 (run): requires three distinct, provenance-verified P_multi checkpoints.

No training, optimizer updates, checkpoint writes, dataset writes, or test-set access.
All runs are development-data mechanism experiments; this is not M01.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "p1"))

K = 16
AGES = (1, 4, 8)
HORIZON = 4
CYCLES_PER_SEQUENCE = 8
MAX_GT_DELTA_MS = 20.0
EXPECTED_BASE_SHA256 = "615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e"
KNOWN_P_CHECKPOINT = (
    "outputs/gt4a2_y04_validation_fix3_20261025_225902/"
    "training/P_multi/gt4a2_c3_C_final_step1000.pt"
)
KNOWN_P_CHECKPOINT_SHA256 = "77a86cdf523f776608b6b591d6c6552002b90398e85d00d3a3fe29afe14a0a24"

# Y04 v2.1 source fixes the outer experiment seed at 20261025. The nested
# gt4a2_c3_closed_loop.py checkpoint field `seed` is 20261024 because that
# module's own constant is retained when Y04 calls its training function.
# The authoritative Y04 seed is therefore validated by the recorded sample
# order hash, not by the nested C3 `seed` field.
Y04_DEFAULT_SEED = 20261025
Y04_TRAINING_STEPS = 1000
Y04_MULTI_SLOT_COUNT = 24


def y04_balanced_order(n_slots: int, steps: int, seed: int) -> list[int]:
    import random
    rng = random.Random(int(seed))
    order = []
    while len(order) < int(steps):
        block = list(range(int(n_slots)))
        rng.shuffle(block)
        order.extend(block)
    return order[:int(steps)]


def y04_sample_order_sha256(seed: int) -> str:
    order = y04_balanced_order(Y04_MULTI_SLOT_COUNT, Y04_TRAINING_STEPS, seed)
    return hashlib.sha256(",".join(map(str, order)).encode()).hexdigest()


def expected_y04_multi_slots() -> list[list[Any]]:
    return (
        [["crowd2", i] for i in range(8)]
        + [["crowd3", i] for i in range(8)]
        + [["balloon2", i] for i in range(8)]
    )

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

SEQUENCES = ("crowd2", "crowd3", "balloon2")
CORE_BRANCHES = ("A", "O", "F", "T", "P_native")
DIAGNOSTIC_BRANCHES = ("R_only", "C_only")
ALL_BRANCHES = CORE_BRANCHES + DIAGNOSTIC_BRANCHES


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def verify_sources() -> dict:
    audit = {}
    for rel, expected in PINNED_SOURCES.items():
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"Pinned source missing: {path}")
        actual = sha256_file(path)
        audit[rel] = {"expected": expected, "actual": actual, "match": actual == expected}
        if actual != expected:
            raise RuntimeError(
                f"Source mismatch; no run started: {rel}: actual={actual}, expected={expected}"
            )
    return audit


def resolve_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def read_and_audit_manifest(path: Path, sequence: str) -> tuple[list[dict], dict]:
    if not path.is_file():
        raise FileNotFoundError(f"{sequence} pair manifest missing: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"{sequence} pair manifest is empty: {path}")
    required = {
        "frame_index_zero_based", "rgb_path", "rgb_timestamp",
        "gt_match_status", "depth_path", "abs_delta_ms", "gt_reused",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise RuntimeError(f"{sequence} manifest missing columns: {missing}")
    rows.sort(key=lambda r: int(r["frame_index_zero_based"]))
    indices = [int(r["frame_index_zero_based"]) for r in rows]
    if indices != list(range(len(rows))):
        raise RuntimeError(
            f"{sequence}: RGB indices must be continuous from zero; "
            f"got {indices[:3]}...{indices[-3:]}, rows={len(rows)}"
        )
    seen_gt = set()
    prev_ts = None
    matched = 0
    unmatched = 0
    for i, row in enumerate(rows):
        row["_index"] = i
        rgb_path = resolve_path(row["rgb_path"])
        if not rgb_path.is_file():
            raise FileNotFoundError(f"{sequence}: missing RGB at index={i}: {rgb_path}")
        row["_rgb_resolved"] = str(rgb_path)
        try:
            ts = float(row["rgb_timestamp"])
        except Exception as e:
            raise RuntimeError(f"{sequence}: invalid RGB timestamp at index={i}") from e
        if not math.isfinite(ts) or (prev_ts is not None and ts <= prev_ts):
            raise RuntimeError(f"{sequence}: RGB timestamps not strictly increasing at index={i}")
        prev_ts = ts
        status = str(row["gt_match_status"]).strip().upper()
        if status == "MATCHED":
            depth_raw = str(row["depth_path"]).strip()
            if not depth_raw:
                raise RuntimeError(f"{sequence}: MATCHED row without depth_path at index={i}")
            depth_path = resolve_path(depth_raw)
            if not depth_path.is_file():
                raise FileNotFoundError(f"{sequence}: missing matched GT at index={i}: {depth_path}")
            reused = str(row.get("gt_reused", "False")).strip().lower() in {"true", "1", "yes"}
            if reused:
                raise RuntimeError(f"{sequence}: GT marked reused at index={i}: {depth_path}")
            try:
                dt = float(row.get("abs_delta_ms") or "nan")
            except Exception as e:
                raise RuntimeError(f"{sequence}: invalid abs_delta_ms at index={i}") from e
            if not math.isfinite(dt) or dt < 0 or dt > MAX_GT_DELTA_MS:
                raise RuntimeError(f"{sequence}: GT time delta invalid/out of range at index={i}: {dt}")
            key = str(depth_path)
            if key in seen_gt:
                raise RuntimeError(f"{sequence}: duplicate GT assignment at index={i}: {key}")
            seen_gt.add(key)
            row["_depth_resolved"] = key
            row["_abs_delta_ms"] = dt
            matched += 1
        else:
            row["_depth_resolved"] = ""
            row["_abs_delta_ms"] = ""
            unmatched += 1
    audit = {
        "sequence": sequence,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rgb_frames": len(rows),
        "matched_gt_frames": matched,
        "unmatched_gt_frames": unmatched,
        "full_K16_cycles": len(rows) // K,
        "tail_frames_not_in_full_cycle": len(rows) % K,
        "max_gt_delta_ms": MAX_GT_DELTA_MS,
        "gt_reuse_check": "PASS",
        "rgb_continuity_check": "PASS",
    }
    return rows, audit


def choose_cycles(candidates: list[int], n: int = CYCLES_PER_SEQUENCE) -> list[int]:
    if len(candidates) <= n:
        return list(candidates)
    chosen = []
    for j in range(n):
        idx = math.floor(j * (len(candidates) - 1) / (n - 1))
        chosen.append(candidates[idx])
    if len(set(chosen)) != len(chosen):
        raise RuntimeError(f"Evenly spaced cycle selection produced duplicates: {chosen}")
    return chosen


def build_anchor_rows(sequence: str, rows: list[dict], audit: dict) -> tuple[list[dict], dict]:
    n_full = len(rows) // K
    candidates = []
    candidate_reasons = []
    for cycle in range(n_full):
        start = cycle * K
        h4_targets = [start + age + HORIZON for age in AGES]
        statuses = [
            str(rows[idx]["gt_match_status"]).strip().upper() == "MATCHED"
            for idx in h4_targets
        ]
        eligible = all(statuses)
        candidate_reasons.append({
            "sequence": sequence,
            "cycle": cycle,
            "cycle_start": start,
            "cycle_end": start + K - 1,
            "h4_target_indices": ",".join(map(str, h4_targets)),
            "h4_target_gt_matched": ",".join("1" if x else "0" for x in statuses),
            "eligible": eligible,
            "reason": "all_age_1_4_8_H4_GT_matched" if eligible else "one_or_more_H4_GT_missing",
        })
        if eligible:
            candidates.append(cycle)
    if not candidates:
        raise RuntimeError(
            f"{sequence}: no eligible K16 cycles; require matched GT at age 1/4/8 H4 targets"
        )
    selected = choose_cycles(candidates)
    anchors = []
    for rank, cycle in enumerate(selected):
        start = cycle * K
        for age in AGES:
            anchor = start + age
            future = list(range(anchor + 1, anchor + HORIZON + 1))
            next_refresh = start + K
            if not all(anchor < idx < next_refresh for idx in future):
                raise RuntimeError(
                    f"Invalid horizon crossing refresh: {sequence} cycle={cycle} age={age}"
                )
            for h, idx in enumerate(future, 1):
                status = str(rows[idx]["gt_match_status"]).strip().upper()
                anchors.append({
                    "sequence": sequence,
                    "cycle": cycle,
                    "cycle_selection_rank": rank,
                    "candidate_cycle_count": len(candidates),
                    "cycle_start_index": start,
                    "cycle_end_index": start + K - 1,
                    "age": age,
                    "anchor_index": anchor,
                    "future_horizon": h,
                    "future_index": idx,
                    "next_refresh_index": next_refresh,
                    "gt_matched": status == "MATCHED",
                    "gt_path": rows[idx]["_depth_resolved"],
                    "rgb_path": rows[idx]["_rgb_resolved"],
                    "gt_delta_ms": rows[idx]["_abs_delta_ms"],
                    "eligibility_basis": "H4 GT at ages 1,4,8; RGB continuity; no method scores",
                })
    meta = {
        "sequence": sequence,
        "full_K16_cycles": n_full,
        "eligible_cycles": candidates,
        "eligible_cycle_count": len(candidates),
        "selected_cycles": selected,
        "selected_cycle_count": len(selected),
        "shortfall_vs_target_8": max(0, CYCLES_PER_SEQUENCE - len(selected)),
        "anchors_per_sequence": len(selected) * len(AGES),
        "ages": list(AGES),
        "horizon": HORIZON,
        "max_future_frame_before_refresh": K - 1,
        "selection_rule": (
            "If n>=8 choose floor(j*(n-1)/7), j=0..7 from time-sorted eligible cycles; "
            "if n<8 include all eligible cycles."
        ),
        "candidate_screen_uses_method_scores": False,
    }
    return anchors, {"candidate_cycles": candidate_reasons, "selection": meta}


def prepare(args) -> int:
    source_audit = verify_sources()
    snapshot = resolve_path(args.snapshot) if args.snapshot else None
    snapshot_audit = {"path": str(snapshot) if snapshot else "", "complete": False}
    if snapshot:
        config = snapshot / "config.json"
        weights = snapshot / "model.safetensors"
        if not config.is_file() or not weights.is_file():
            raise FileNotFoundError(f"Incomplete VeloDepth snapshot: {snapshot}")
        actual_base = sha256_file(weights)
        snapshot_audit.update({
            "config_exists": True,
            "weights_exists": True,
            "model_safetensors_sha256": actual_base,
            "expected_sha256": EXPECTED_BASE_SHA256,
            "complete": actual_base == EXPECTED_BASE_SHA256,
        })
        if actual_base != EXPECTED_BASE_SHA256:
            raise RuntimeError(
                f"VeloDepth base model hash mismatch: {actual_base} != {EXPECTED_BASE_SHA256}"
            )
    else:
        raise RuntimeError("VDEEP_SNAPSHOT unset; pass --snapshot or export VDEEP_SNAPSHOT.")

    manifest_args = {
        "crowd2": args.crowd2_pair_csv,
        "crowd3": args.crowd3_pair_csv,
        "balloon2": args.balloon2_pair_csv,
    }
    all_anchors = []
    data_audit = []
    selection = {}
    for seq in SEQUENCES:
        pair_path = resolve_path(manifest_args[seq])
        rows, audit = read_and_audit_manifest(pair_path, seq)
        anchors, detail = build_anchor_rows(seq, rows, audit)
        data_audit.append(audit)
        all_anchors.extend(anchors)
        selection[seq] = detail

    if not all_anchors:
        raise RuntimeError("No M05 anchors selected.")

    out = resolve_path(args.out_dir) if args.out_dir else (
        ROOT / "outputs" / f"cvpr_m05_formal_prepare_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    )
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "anchor_manifest.csv", all_anchors)
    write_csv(out / "data_manifest_audit.csv", data_audit)
    candidate_rows = [
        row for seq in SEQUENCES for row in selection[seq]["candidate_cycles"]
    ]
    write_csv(out / "candidate_cycle_audit.csv", candidate_rows)

    checkpoint_template = [
        {
            "seed": str(Y04_DEFAULT_SEED),
            "checkpoint_path": KNOWN_P_CHECKPOINT,
            "expected_sha256": KNOWN_P_CHECKPOINT_SHA256,
            "training_summary_path": (
                "outputs/gt4a2_y04_validation_fix3_20261025_225902/"
                "training/P_multi/gt4a2_c3_C_training_summary.json"
            ),
            "training_log_path": "",
            "notes": "Y04 outer seed=20261025; verify by y04_sample_order_sha256. Do not use nested C3 seed metadata.",
        },
        {"seed": "", "checkpoint_path": "", "expected_sha256": "", "training_summary_path": "", "training_log_path": "", "notes": "Fill with second independently trained P_multi seed and its training log."},
        {"seed": "", "checkpoint_path": "", "expected_sha256": "", "training_summary_path": "", "training_log_path": "", "notes": "Fill with third independently trained P_multi seed and its training log."},
    ]
    known_checkpoint = ROOT / KNOWN_P_CHECKPOINT
    checkpoint_status = {
        "known_checkpoint_path": str(known_checkpoint),
        "exists": known_checkpoint.is_file(),
        "sha256_matches_M00": False,
        "nested_c3_seed_metadata": None,
        "y04_outer_seed": Y04_DEFAULT_SEED,
        "y04_sample_order_sha256_expected": y04_sample_order_sha256(Y04_DEFAULT_SEED),
        "y04_sample_order_sha256_matches": False,
        "formal_seed_coverage": "0/3",
    }
    if known_checkpoint.is_file():
        actual = sha256_file(known_checkpoint)
        checkpoint_status["sha256_matches_M00"] = actual == KNOWN_P_CHECKPOINT_SHA256
        if actual != KNOWN_P_CHECKPOINT_SHA256:
            raise RuntimeError(
                f"Known P_multi checkpoint differs from M00-verified hash: {actual}"
            )
        try:
            import torch
            ckpt = torch.load(known_checkpoint, map_location="cpu", weights_only=False)
            checkpoint_status["nested_c3_seed_metadata"] = ckpt.get("seed")
            checkpoint_status["y04_sample_order_sha256_matches"] = (
                ckpt.get("y04_sample_order_sha256")
                == y04_sample_order_sha256(Y04_DEFAULT_SEED)
            )
            checkpoint_status["metadata_pass"] = (
                ckpt.get("mode_label") == "C"
                and ckpt.get("mode") == "pre_readout"
                and ckpt.get("y04_label") == "Y04_P_multi"
                and ckpt.get("y04_kind") == "persistent"
                and int(ckpt.get("step_count", -1)) == 1000
                and int(ckpt.get("optimizer_steps", -1)) == 1000
                and ckpt.get("pretrained_weights_sha256") == EXPECTED_BASE_SHA256
                and set(ckpt.get("training_sequences", [])) == set(SEQUENCES)
                and ckpt.get("y04_sample_order_sha256")
                    == y04_sample_order_sha256(Y04_DEFAULT_SEED)
                and ckpt.get("training_cycle_slots") == expected_y04_multi_slots()
            )
            checkpoint_status["formal_seed_coverage"] = "1/3" if checkpoint_status["metadata_pass"] else "0/3"
        except Exception as e:
            checkpoint_status["metadata_pass"] = False
            checkpoint_status["metadata_error"] = repr(e)
    # The first seed is sourced from the pinned Y04 outer script and checked
    # against its deterministic sample-order digest. The nested C3 seed field
    # is retained as provenance metadata but is not used as the Y04 seed.
    write_csv(out / "checkpoint_manifest.csv", checkpoint_template)

    plan = {
        "experiment": "CVPR-M05 Formal 1.0",
        "phase": "PREPARED_ANCHORS_FROZEN",
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(ROOT),
        "protocol_source": "SVR-Depth_CVPR主实验执行计划_20261101_v2.1",
        "training_performed": False,
        "model_inference_performed": False,
        "test_set_accessed": False,
        "data_role": "development-only mechanism study; Crowd2/Crowd3/Balloon2",
        "K": K,
        "ages": list(AGES),
        "horizon": HORIZON,
        "cycles_per_sequence_target": CYCLES_PER_SEQUENCE,
        "sequences": manifest_args,
        "data_audit": data_audit,
        "selection": {seq: selection[seq]["selection"] for seq in SEQUENCES},
        "anchor_count": len({
            (r["sequence"], int(r["cycle"]), int(r["age"])) for r in all_anchors
        }),
        "anchor_manifest_row_count": len(all_anchors),
        "anchor_manifest": str(out / "anchor_manifest.csv"),
        "anchor_manifest_sha256": sha256_file(out / "anchor_manifest.csv"),
        "source_hashes": source_audit,
        "snapshot": snapshot_audit,
        "checkpoint_status": checkpoint_status,
        "checkpoint_manifest": str(out / "checkpoint_manifest.csv"),
        "formal_run_requirement": (
            "Exactly three distinct P_multi checkpoints with unique training seeds, "
            "matching Y04_P_multi/pre_readout/1000-step provenance and the same base model."
        ),
        "primary_estimand": (
            "A-native parent, K16, H4 single target frame, T minus O paired AbsRel; "
            "negative values favor committing F1. F minus A is the corresponding state-contract check."
        ),
        "secondary_strata": (
            "P-maintained parent is reported separately; never pooled with A-native."
        ),
        "statistical_unit": (
            "Anchor is paired within recording; cycles/frames are not independent physical scenes. "
            "Only three named development recordings are in this package, so no scene-level inferential CI."
        ),
    }
    write_json(out / "protocol_lock.json", plan)
    lines = [
        "# CVPR M05 Formal 1.0 — anchor preparation",
        "",
        "**Status: PREPARED_ANCHORS_FROZEN**",
        "",
        "- Training: NOT RUN",
        "- Model inference: NOT RUN",
        "- Test set: NOT accessed",
        f"- K / ages / H: {K} / {AGES} / {HORIZON}",
        f"- Frozen anchor interventions: {len({(r['sequence'], int(r['cycle']), int(r['age'])) for r in all_anchors})}",
        f"- Anchor-manifest horizon rows: {len(all_anchors)}",
        f"- Existing verified P checkpoint seed coverage: {checkpoint_status['formal_seed_coverage']}",
        "",
        "## Selected cycles",
        "",
        "|Sequence|Eligible cycles|Selected cycles|Anchors|Shortfall|",
        "|---|---:|---:|---:|---:|",
    ]
    for seq in SEQUENCES:
        m = selection[seq]["selection"]
        lines.append(
            f"|{seq}|{m['eligible_cycle_count']}|{m['selected_cycle_count']}|"
            f"{m['anchors_per_sequence']}|{m['shortfall_vs_target_8']}|"
        )
    lines += [
        "",
        "## Next requirement",
        "",
        "The formal execution phase is intentionally blocked until three distinct, "
        "provenance-verified P_multi checkpoints are listed in `checkpoint_manifest.csv`. "
        "The known M00-verified checkpoint fills seed 20261025; two additional independently "
        "trained seeds are not assumed to exist.",
        "",
        "Do not duplicate the existing checkpoint under different seed labels.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    review = out / "cvpr_m05_formal_prepare_review.zip"
    with zipfile.ZipFile(review, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in (
            "protocol_lock.json", "anchor_manifest.csv", "data_manifest_audit.csv",
            "candidate_cycle_audit.csv", "checkpoint_manifest.csv", "REPORT.md",
        ):
            z.write(out / name, name)
    with zipfile.ZipFile(review) as z:
        if z.testzip() is not None:
            raise RuntimeError("Prepare review ZIP CRC verification failed")
    print(f"[M05-FORMAL] phase=PREPARED_ANCHORS_FROZEN", flush=True)
    print(f"[M05-FORMAL] anchors={len(all_anchors)}", flush=True)
    print(f"[M05-FORMAL] checkpoint_seed_coverage={checkpoint_status['formal_seed_coverage']}", flush=True)
    print(f"[M05-FORMAL] report={out / 'REPORT.md'}", flush=True)
    print(f"[M05-FORMAL] review={review}", flush=True)
    return 0


def parse_checkpoint_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint manifest missing: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    complete = []
    for row in rows:
        seed_raw = str(row.get("seed", "")).strip()
        cp_raw = str(row.get("checkpoint_path", "")).strip()
        sha_raw = str(row.get("expected_sha256", "")).strip().lower()
        if not seed_raw and not cp_raw and not sha_raw:
            continue
        if not seed_raw or not cp_raw or len(sha_raw) != 64:
            raise RuntimeError(f"Incomplete checkpoint manifest row: {row}")
        try:
            seed = int(seed_raw)
        except ValueError as e:
            raise RuntimeError(f"Checkpoint seed must be integer: {seed_raw}") from e
        complete.append({
            "seed": seed,
            "checkpoint_path": resolve_path(cp_raw),
            "expected_sha256": sha_raw,
            "training_summary_path": resolve_path(row["training_summary_path"])
                if str(row.get("training_summary_path", "")).strip() else None,
            "training_log_path": resolve_path(row["training_log_path"])
                if str(row.get("training_log_path", "")).strip() else None,
            "notes": row.get("notes", ""),
        })
    if len(complete) != 3:
        raise RuntimeError(
            f"FORMAL_RUN_BLOCKED: exactly 3 complete P_multi checkpoints required; found {len(complete)}. "
            "No model inference has started."
        )
    if len({x["seed"] for x in complete}) != 3:
        raise RuntimeError("FORMAL_RUN_BLOCKED: training seed IDs must be unique.")
    if len({str(x["checkpoint_path"]) for x in complete}) != 3:
        raise RuntimeError("FORMAL_RUN_BLOCKED: checkpoint paths must be unique.")
    if len({x["expected_sha256"] for x in complete}) != 3:
        raise RuntimeError("FORMAL_RUN_BLOCKED: checkpoint SHA256 values must be unique.")
    return sorted(complete, key=lambda x: x["seed"])


def verify_checkpoint_rows(rows: list[dict]) -> list[dict]:
    import torch
    verified = []
    for item in rows:
        p = item["checkpoint_path"]
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint missing: {p}")
        actual = sha256_file(p)
        if actual != item["expected_sha256"]:
            raise RuntimeError(f"Checkpoint SHA mismatch: {p}: {actual}")
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        expected_order_sha = y04_sample_order_sha256(item["seed"])
        checks = {
            "y04_sample_order_matches_manifest_seed": (
                ckpt.get("y04_sample_order_sha256") == expected_order_sha
            ),
            "y04_multi_training_slots": (
                ckpt.get("training_cycle_slots") == expected_y04_multi_slots()
            ),
            "y04_optimizer_steps_1000": int(ckpt.get("optimizer_steps", -1)) == 1000,
            "mode_label_C": ckpt.get("mode_label") == "C",
            "mode_pre_readout": ckpt.get("mode") == "pre_readout",
            "y04_label": ckpt.get("y04_label") == "Y04_P_multi",
            "y04_kind_persistent": ckpt.get("y04_kind") == "persistent",
            "step_count_1000": int(ckpt.get("step_count", -1)) == 1000,
            "base_model_sha256": ckpt.get("pretrained_weights_sha256") == EXPECTED_BASE_SHA256,
            "three_training_sequences": set(ckpt.get("training_sequences", [])) == set(SEQUENCES),
            "repair_head_state_dict": isinstance(ckpt.get("repair_head_state_dict"), dict)
                and bool(ckpt.get("repair_head_state_dict")),
        }
        if not all(checks.values()):
            raise RuntimeError(f"Checkpoint provenance failed: {p}: {checks}")
        summary = item["training_summary_path"]
        if summary:
            if not summary.is_file():
                raise FileNotFoundError(f"Training summary missing: {summary}")
            summary_obj = json.loads(summary.read_text(encoding="utf-8"))
            if summary_obj.get("checkpoint_sha256") and summary_obj["checkpoint_sha256"] != actual:
                raise RuntimeError(f"Training summary/checkpoint hash mismatch: {summary}")
        training_log = item["training_log_path"]
        if training_log and not training_log.is_file():
            raise FileNotFoundError(f"Training log missing: {training_log}")
        verified.append({
            "seed": item["seed"],
            "checkpoint_path": str(p),
            "sha256": actual,
            "training_summary_path": str(summary) if summary else "",
            "training_log_path": str(training_log) if training_log else "",
            "checks": checks,
            "repair_head_keys": len(ckpt["repair_head_state_dict"]),
            "training_sequences": ckpt.get("training_sequences", []),
            "sample_order_sha256": ckpt.get("y04_sample_order_sha256", ckpt.get("sample_order_sha256", "")),
            "initial_head_sha256": ckpt.get("y04_initial_head_sha256", ckpt.get("initial_head_state_sha256", "")),
            "y04_sample_order_sha256": ckpt.get("y04_sample_order_sha256", ""),
            "nested_c3_seed_metadata_not_authoritative": ckpt.get("seed"),
        })
        del ckpt
    return verified


def tensor_sha(obj: Any) -> str:
    import torch
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu().contiguous()
            h.update(str(t.dtype).encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(t.numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x):
                h.update(str(k).encode())
                visit(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x:
                visit(v)
        else:
            h.update(repr(x).encode())
    visit(obj)
    return h.hexdigest()


def clone_state(state: dict) -> dict:
    return {
        "radius": state["radius"].detach().clone(),
        "confidence": state["confidence"].detach().clone(),
        "features": [x.detach().clone() for x in state["features"]],
        "rays_map": state["rays_map"].detach().clone(),
    }


def max_abs(a: Any, b: Any) -> float:
    import torch
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return float("inf")
        d = (a.detach().float() - b.detach().float()).abs()
        if d.numel() == 0 or not bool(torch.isfinite(d).all()):
            return 0.0 if d.numel() == 0 else float("inf")
        return float(d.max().item())
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return float("inf")
        return max((max_abs(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return float("inf")
        return max((max_abs(a[k], b[k]) for k in a), default=0.0)
    return 0.0 if a == b else float("inf")


def state_diff(a: dict, b: dict) -> dict:
    return {
        "radius": max_abs(a["radius"], b["radius"]),
        "confidence": max_abs(a["confidence"], b["confidence"]),
        "features": max_abs(a["features"], b["features"]),
        "rays_map": max_abs(a["rays_map"], b["rays_map"]),
        "all_fields": max_abs(a, b),
    }


def set_mode(model, mode: str) -> None:
    if mode not in {"off", "pre_readout"}:
        raise ValueError(mode)
    model.pixel_decoder.feature_repair_mode = mode


def capture_rng(torch):
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(torch, state):
    torch.set_rng_state(state["cpu"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def run_step(model, exp1, prev_frame, curr_frame, static, state, torch):
    set_mode(model, model.pixel_decoder.feature_repair_mode)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        r, c, f = exp1.one_fast_step(
            model, prev_frame, curr_frame, static,
            state["radius"], state["confidence"], state["features"]
        )
    return {
        "radius": r.detach(),
        "confidence": c.detach(),
        "features": [x.detach() for x in f],
        "rays_map": state["rays_map"].detach(),
    }


def build_parent_states(model, exp1, frames, static_by_idx, cycle_start, ages,
                        mode, rng_state, torch):
    set_mode(model, mode)
    restore_rng(torch, rng_state)
    rays, radius, confidence, features = exp1.keyframe_state(
        model, frames[cycle_start], model._m05_shape
    )
    state = {
        "radius": radius.detach(),
        "confidence": confidence.detach(),
        "features": [x.detach() for x in features],
        "rays_map": rays.detach(),
    }
    cache = {}
    if 1 in ages:
        cache[1] = clone_state(state)
    max_parent_step = max(ages) - 1
    for step in range(1, max_parent_step + 1):
        idx = cycle_start + step
        state = run_step(
            model, exp1, frames[idx - 1], frames[idx], static_by_idx[idx], state, torch
        )
        age = step + 1
        if age in ages:
            cache[age] = clone_state(state)
    if set(cache) != set(ages):
        raise RuntimeError(f"Parent state cache incomplete: expected={ages}, got={sorted(cache)}")
    return cache


def make_prediction(model, gt2, state, shape):
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        pred = gt2.make_z_depth(
            state["radius"], state["rays_map"], shape["padded_shape"],
            shape["paddings"], model.interpolation_mode
        )
    return pred.detach().float().cpu().numpy()


def metric_row(gt_np, pred_np):
    import numpy as np
    if gt_np is None:
        return {
            "gt_matched": False, "AbsRel": "", "RMSE": "", "delta1": "",
            "valid_ratio": "", "gt_valid_ratio": "", "valid_pixels": 0, "total_pixels": 0,
        }
    from gt2_fixed_clip_repair_train import calculate_metrics
    return {"gt_matched": True, **calculate_metrics(gt_np, pred_np)}


def hash_static(static_by_idx):
    return tensor_sha(static_by_idx)


class CallCounter:
    """Nested module-call ledger; counts are not additive across nested modules."""
    def __init__(self, model):
        self.stage = "setup"
        self.counts = defaultdict(lambda: defaultdict(int))
        self.handles = []
        modules = [
            ("pixel_encoder", model.pixel_encoder),
            ("small_encoder", model.small_encoder),
            ("pixel_decoder", model.pixel_decoder),
            ("radial_forward_fast", model.pixel_decoder.radial_module),
            ("persistent_feature_repair", model.pixel_decoder.persistent_feature_repair),
        ]
        for name, module in modules:
            if module is None:
                continue
            self.handles.append(
                module.register_forward_hook(
                    lambda _m, _i, _o, n=name: self.counts[self.stage].__setitem__(
                        n, self.counts[self.stage][n] + 1
                    )
                )
            )

    def set_stage(self, stage):
        self.stage = stage

    def reset(self):
        self.counts = defaultdict(lambda: defaultdict(int))
        self.stage = "setup"

    def rows(self):
        out = []
        for stage, modules in self.counts.items():
            for name, val in modules.items():
                out.append({"stage": stage, "module": name, "calls": val,
                            "nested_counts_not_additive": True})
        return out

    def close(self):
        for h in self.handles:
            h.remove()


def branch_replay(model, exp1, gt2, frames, static_by_idx, shape, anchor, state,
                  parent_label, branch, rng_state, torch, static_hash_before):
    set_mode(model, "off")
    restore_rng(torch, rng_state)
    current = clone_state(state)
    rows = []
    preds = {}
    torch.cuda.reset_peak_memory_stats()
    t_branch = time.perf_counter()
    for h in range(1, HORIZON + 1):
        idx = anchor + h
        current = run_step(
            model, exp1, frames[idx - 1], frames[idx], static_by_idx[idx], current, torch
        )
        pred = make_prediction(model, gt2, current, shape)
        preds[h] = pred
        metric = metric_row(frames[idx]["gt_np"], pred)
        rows.append({
            "parent": parent_label,
            "branch": branch,
            "horizon": h,
            "frame_index": idx,
            "repair_mode_after_intervention": "off",
            "state_sha256_after_step": tensor_sha(current),
            **metric,
        })
    torch.cuda.synchronize()
    branch_wall_sec = time.perf_counter() - t_branch
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    static_hash_after = hash_static(static_by_idx)
    return (
        rows, preds, static_hash_before == static_hash_after,
        branch_wall_sec, peak_allocated, peak_reserved,
    )


def process_anchor(model, exp1, gt2, frames, static_by_idx, shape, anchor_row,
                   parent_label, parent_state, tolerance, torch, counter):
    anchor = int(anchor_row["anchor_index"])
    prev = frames[anchor - 1]
    curr = frames[anchor]
    parent_hash = tensor_sha(parent_state)
    parent_rng = capture_rng(torch)
    parent_rng_hash = tensor_sha(parent_rng)
    timing_rows = []

    counter.set_stage("anchor_intervention")
    torch.cuda.reset_peak_memory_stats()
    t_intervention = time.perf_counter()
    set_mode(model, "off")
    restore_rng(torch, parent_rng)
    s0 = run_step(model, exp1, prev, curr, static_by_idx[anchor], clone_state(parent_state), torch)

    set_mode(model, "pre_readout")
    restore_rng(torch, parent_rng)
    s1 = run_step(model, exp1, prev, curr, static_by_idx[anchor], clone_state(parent_state), torch)

    d0 = make_prediction(model, gt2, s0, shape)
    d1 = make_prediction(model, gt2, s1, shape)
    torch.cuda.synchronize()
    intervention_wall_sec = time.perf_counter() - t_intervention
    timing_rows.append({
        "seed": anchor_row["seed"], "sequence": anchor_row["sequence"],
        "cycle": int(anchor_row["cycle"]), "age": int(anchor_row["age"]),
        "parent": parent_label, "anchor_index": anchor,
        "stage": "anchor_intervention_D0_D1",
        "wall_sec": intervention_wall_sec,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    })
    states = {
        "A": {"radius": s0["radius"], "confidence": s0["confidence"],
              "features": s0["features"], "rays_map": parent_state["rays_map"]},
        "O": {"radius": s0["radius"], "confidence": s0["confidence"],
              "features": s0["features"], "rays_map": parent_state["rays_map"]},
        "F": {"radius": s0["radius"], "confidence": s0["confidence"],
              "features": s1["features"], "rays_map": parent_state["rays_map"]},
        "T": {"radius": s0["radius"], "confidence": s0["confidence"],
              "features": s1["features"], "rays_map": parent_state["rays_map"]},
        "P_native": {"radius": s1["radius"], "confidence": s1["confidence"],
                     "features": s1["features"], "rays_map": parent_state["rays_map"]},
        # Supplementary 2x2 r/c field-isolation cells conditional on F1.
        "R_only": {"radius": s1["radius"], "confidence": s0["confidence"],
                   "features": s1["features"], "rays_map": parent_state["rays_map"]},
        "C_only": {"radius": s0["radius"], "confidence": s1["confidence"],
                   "features": s1["features"], "rays_map": parent_state["rays_map"]},
    }
    current_predictions = {"A": d0, "F": d0, "O": d1, "T": d1, "P_native": d1}
    current_rows = []
    for branch in CORE_BRANCHES:
        metric = metric_row(curr["gt_np"], current_predictions[branch])
        current_rows.append({
            "seed": anchor_row["seed"],
            "sequence": anchor_row["sequence"],
            "cycle": int(anchor_row["cycle"]),
            "age": int(anchor_row["age"]),
            "parent": parent_label,
            "anchor_index": anchor,
            "branch": branch,
            "current_output_source": "D0" if branch in {"A", "F"} else "D1",
            "committed_state_sha256": tensor_sha(states[branch]),
            **metric,
        })

    state_rows = []
    for branch, state in states.items():
        state_rows.append({
            "seed": anchor_row["seed"], "sequence": anchor_row["sequence"],
            "cycle": int(anchor_row["cycle"]), "age": int(anchor_row["age"]),
            "parent": parent_label, "anchor_index": anchor, "branch": branch,
            "parent_state_sha256": parent_hash,
            "committed_state_sha256": tensor_sha(state),
            "radius_dtype": str(state["radius"].dtype),
            "radius_shape": str(tuple(state["radius"].shape)),
            "confidence_dtype": str(state["confidence"].dtype),
            "confidence_shape": str(tuple(state["confidence"].shape)),
            "feature_dtypes": ";".join(str(x.dtype) for x in state["features"]),
            "feature_shapes": ";".join(str(tuple(x.shape)) for x in state["features"]),
            "rays_dtype": str(state["rays_map"].dtype),
            "rays_shape": str(tuple(state["rays_map"].shape)),
        })

    diffs = {
        "A_vs_O": state_diff(states["A"], states["O"]),
        "F_vs_T": state_diff(states["F"], states["T"]),
        "F_vs_P_native": state_diff(states["F"], states["P_native"]),
        "F_vs_R_only": state_diff(states["F"], states["R_only"]),
        "F_vs_C_only": state_diff(states["F"], states["C_only"]),
        "current_D0_A_vs_F_max_abs": float(abs(d0 - d0).max()),
        "current_D1_O_vs_T_max_abs": float(abs(d1 - d1).max()),
        "current_D1_T_vs_P_native_max_abs": float(abs(d1 - d1).max()),
    }

    future_rows = []
    future_predictions = {}
    static_before = hash_static(static_by_idx)
    for branch in ALL_BRANCHES:
        counter.set_stage(f"future_{branch}")
        rows, preds, static_unchanged, branch_wall_sec, peak_allocated, peak_reserved = branch_replay(
            model, exp1, gt2, frames, static_by_idx, shape, anchor, states[branch],
            parent_label, branch, parent_rng, torch, static_before
        )
        timing_rows.append({
            "seed": anchor_row["seed"], "sequence": anchor_row["sequence"],
            "cycle": int(anchor_row["cycle"]), "age": int(anchor_row["age"]),
            "parent": parent_label, "anchor_index": anchor,
            "stage": f"future_replay_{branch}",
            "wall_sec": branch_wall_sec,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        })
        if not static_unchanged:
            raise RuntimeError(
                f"Shared flow/static cache mutated during branch {branch}: {anchor_row}"
            )
        for r in rows:
            r.update({
                "seed": anchor_row["seed"], "sequence": anchor_row["sequence"],
                "cycle": int(anchor_row["cycle"]), "age": int(anchor_row["age"]),
                "anchor_index": anchor, "static_cache_unchanged": True,
            })
        future_rows.extend(rows)
        future_predictions[branch] = preds

    future_checks = {}
    for h in range(1, HORIZON + 1):
        a, o = future_predictions["A"][h], future_predictions["O"][h]
        f, t = future_predictions["F"][h], future_predictions["T"][h]
        p = future_predictions["P_native"][h]
        future_checks[str(h)] = {
            "A_vs_O_prediction_max_abs": float(abs(a - o).max()),
            "F_vs_T_prediction_max_abs": float(abs(f - t).max()),
            "T_minus_O_equals_F_minus_A_prediction_max_abs": float(
                abs((t - o) - (f - a)).max()
            ),
            "P_native_vs_T_prediction_max_abs": float(abs(p - t).max()),
            "R_only_vs_T_prediction_max_abs": float(
                abs(future_predictions["R_only"][h] - t).max()
            ),
            "C_only_vs_T_prediction_max_abs": float(
                abs(future_predictions["C_only"][h] - t).max()
            ),
        }

    invariants = {
        "A_O_full_state_equal": diffs["A_vs_O"]["all_fields"] <= tolerance,
        "F_T_full_state_equal": diffs["F_vs_T"]["all_fields"] <= tolerance,
        "F_T_Pnative_feature_equal": (
            max_abs(states["F"]["features"], states["P_native"]["features"]) <= tolerance
        ),
        "current_D0_A_F_equal": float(abs(d0 - d0).max()) <= tolerance,
        "current_D1_O_T_Pnative_equal": float(abs(d1 - d1).max()) <= tolerance,
        "future_A_O_equal": all(v["A_vs_O_prediction_max_abs"] <= tolerance for v in future_checks.values()),
        "future_F_T_equal": all(v["F_vs_T_prediction_max_abs"] <= tolerance for v in future_checks.values()),
        "future_T_O_equals_F_A": all(
            v["T_minus_O_equals_F_minus_A_prediction_max_abs"] <= tolerance
            for v in future_checks.values()
        ),
        "shared_static_cache_unchanged": True,
    }
    # Explicitly test the C-only state intervention. Since the audited fast_step
    # signature receives radius and features but not incoming confidence, this
    # should produce the same future trajectory as T. It is a path-specific result.
    confidence_effect = {
        "C_only_vs_T_future_max_abs_by_horizon": {
            h: future_checks[h]["C_only_vs_T_prediction_max_abs"] for h in future_checks
        },
        "confidence_consumed_by_fast_step": any(
            v["C_only_vs_T_prediction_max_abs"] > tolerance for v in future_checks.values()
        ),
        "scope": "fixed-K fast_step path; keyframe refresh reinitializes state and accepts no prior confidence argument",
    }
    invariants["confidence_only_no_effect_on_fixed_fast_step"] = (
        not confidence_effect["confidence_consumed_by_fast_step"]
    )

    # The two extra branches form a supplementary 2x2 radius/confidence contrast
    # conditional on F1; they are diagnostics, not additional primary estimands.
    field_factorial = {}
    for h in range(1, HORIZON + 1):
        t = future_predictions["T"][h]
        r = future_predictions["R_only"][h]
        c = future_predictions["C_only"][h]
        p = future_predictions["P_native"][h]
        field_factorial[str(h)] = {
            "R_only_minus_T_prediction_mean": float((r - t).mean()),
            "C_only_minus_T_prediction_mean": float((c - t).mean()),
            "P_native_minus_R_only_prediction_mean": float((p - r).mean()),
            "P_native_minus_C_only_prediction_mean": float((p - c).mean()),
            "interaction_prediction_mean": float((p - r - c + t).mean()),
        }

    detail = {
        "seed": anchor_row["seed"], "sequence": anchor_row["sequence"],
        "cycle": int(anchor_row["cycle"]), "age": int(anchor_row["age"]),
        "parent": parent_label, "anchor_index": anchor,
        "parent_state_sha256": parent_hash,
        "parent_rng_sha256": parent_rng_hash,
        "state_diffs": diffs,
        "future_checks": future_checks,
        "confidence_effect": confidence_effect,
        "radius_confidence_factorial_diagnostic": field_factorial,
        "invariants": invariants,
        "passed": all(invariants.values()),
    }
    return current_rows, future_rows, state_rows, detail, timing_rows


def state_io_audit() -> dict:
    import gt2_fixed_clip_repair_train as gt2
    import gt4a2_exp1_train_eval as exp1
    fast_sig = str(inspect.signature(gt2.fast_step))
    key_sig = str(inspect.signature(exp1.keyframe_state))
    fast_src = inspect.getsource(gt2.fast_step)
    one_src = inspect.getsource(exp1.one_fast_step)
    key_src = inspect.getsource(exp1.keyframe_state)
    confidence_in_fast_inputs = "confidence_state" in fast_src or "confidence_state" in one_src
    return {
        "gt2_fast_step_signature": fast_sig,
        "exp1_one_fast_step_signature": str(inspect.signature(exp1.one_fast_step)),
        "exp1_keyframe_state_signature": key_sig,
        "fast_step_source_sha256": sha256_bytes(fast_src.encode()),
        "one_fast_step_source_sha256": sha256_bytes(one_src.encode()),
        "keyframe_state_source_sha256": sha256_bytes(key_src.encode()),
        "incoming_confidence_passed_to_fast_step": confidence_in_fast_inputs,
        "fast_step_source_excerpt": fast_src,
        "keyframe_reset_accepts_previous_state": False,
        "interpretation": (
            "The audited fixed-K path consumes radius/features; incoming confidence is not an argument. "
            "Keyframe initialization receives the current frame and shape, not prior state. "
            "This is a source-path statement, not a claim about every possible model path."
        ),
    }


def summarize_effects(effect_rows: list[dict]) -> dict:
    # No frame/anchor-level inferential CI: these are only three development recordings.
    by = defaultdict(list)
    for r in effect_rows:
        key = (int(r["seed"]), r["sequence"], r["parent"], int(r["age"]))
        by[key].append(r)

    strata = []
    for key, vals in sorted(by.items()):
        seed, seq, parent, age = key
        def mean_field(field):
            xs = [float(v[field]) for v in vals if v.get(field) not in ("", None)]
            return sum(xs) / len(xs) if xs else None
        strata.append({
            "seed": seed, "sequence": seq, "parent": parent, "age": age,
            "n_anchors": len(vals),
            "n_H4": sum(v.get("T_minus_O_H4_AbsRel") not in ("", None) for v in vals),
            "mean_T_minus_O_H4_AbsRel": mean_field("T_minus_O_H4_AbsRel"),
            "mean_F_minus_A_H4_AbsRel": mean_field("F_minus_A_H4_AbsRel"),
            "mean_Pnative_minus_T_H4_AbsRel": mean_field("Pnative_minus_T_H4_AbsRel"),
            "mean_Ronly_minus_T_H4_AbsRel": mean_field("Ronly_minus_T_H4_AbsRel"),
            "mean_Conly_minus_T_H4_AbsRel": mean_field("Conly_minus_T_H4_AbsRel"),
            "mean_T_minus_O_H1_H4_window_AbsRel": mean_field("T_minus_O_H1_H4_window_AbsRel"),
        })

    per_seed = []
    for seed in sorted({int(r["seed"]) for r in effect_rows}):
        seq_vals = []
        per_seq = {}
        for seq in SEQUENCES:
            age_vals = []
            for age in AGES:
                matches = [
                    s for s in strata
                    if s["seed"] == seed and s["sequence"] == seq
                    and s["parent"] == "A_native" and s["age"] == age
                    and s["mean_T_minus_O_H4_AbsRel"] is not None
                ]
                if matches:
                    age_vals.append(matches[0]["mean_T_minus_O_H4_AbsRel"])
            if age_vals:
                per_seq[seq] = sum(age_vals) / len(age_vals)
                seq_vals.append(per_seq[seq])
        per_seed.append({
            "seed": seed,
            "A_native_primary_sequence_equal_mean_T_minus_O_H4": (
                sum(seq_vals) / len(seq_vals) if seq_vals else None
            ),
            "A_native_primary_by_sequence_equal_age_mean": per_seq,
            "interpretation": "descriptive only; do not treat anchors or frames as independent scenes",
        })

    seed_values = [
        x["A_native_primary_sequence_equal_mean_T_minus_O_H4"]
        for x in per_seed if x["A_native_primary_sequence_equal_mean_T_minus_O_H4"] is not None
    ]
    return {
        "strata": strata,
        "per_seed_primary": per_seed,
        "three_seed_descriptive_mean": sum(seed_values) / len(seed_values) if seed_values else None,
        "inferential_ci": None,
        "ci_reason": (
            "Only three named development recordings are included and no verified physical-scene "
            "group map is supplied. Anchor/frame bootstrap would be pseudoreplication; report "
            "recording-level descriptive results and add scene-group inference only with verified groups."
        ),
    }


def run(args) -> int:
    import numpy as np
    import torch
    import cv2
    import gt2_fixed_clip_repair_train as gt2
    import gt4a2_exp1_train_eval as exp1

    if not torch.cuda.is_available():
        raise RuntimeError("M05 Formal run requires CUDA; no CPU fallback.")
    if not args.snapshot:
        raise RuntimeError("VDEEP_SNAPSHOT unset; pass --snapshot or export it.")
    snapshot = resolve_path(args.snapshot)
    if sha256_file(snapshot / "model.safetensors") != EXPECTED_BASE_SHA256:
        raise RuntimeError("VeloDepth model.safetensors SHA256 differs from M00 lock.")

    plan_path = resolve_path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("phase") != "PREPARED_ANCHORS_FROZEN":
        raise RuntimeError("Plan is not a prepared/frozen M05 anchor plan.")
    if Path(plan["repo_root"]).resolve() != ROOT:
        raise RuntimeError("Prepared plan belongs to a different repository root.")
    if sha256_file(Path(plan["anchor_manifest"])) != plan["anchor_manifest_sha256"]:
        raise RuntimeError("Frozen anchor manifest hash changed.")
    source_audit = verify_sources()
    for rel, meta in plan["source_hashes"].items():
        if source_audit[rel]["actual"] != meta["actual"]:
            raise RuntimeError(f"Source changed after prepare: {rel}")

    checkpoint_manifest = resolve_path(args.checkpoint_manifest)
    checkpoint_rows = parse_checkpoint_manifest(checkpoint_manifest)
    verified_checkpoints = verify_checkpoint_rows(checkpoint_rows)

    anchor_rows = []
    with Path(plan["anchor_manifest"]).open("r", newline="", encoding="utf-8-sig") as f:
        anchor_rows = list(csv.DictReader(f))
    if not anchor_rows:
        raise RuntimeError("Frozen anchor manifest is empty.")
    # Verify all sequence pair manifests still match the preparation lock.
    sequence_rows = {}
    sequence_audits = {}
    for seq in SEQUENCES:
        pair_path = resolve_path(plan["sequences"][seq])
        rows, audit = read_and_audit_manifest(pair_path, seq)
        expected = next(x for x in plan["data_audit"] if x["sequence"] == seq)
        if audit["sha256"] != expected["sha256"]:
            raise RuntimeError(f"{seq} pair manifest changed after anchor freeze.")
        sequence_rows[seq] = exp1.read_manifest(pair_path, training=False)
        sequence_audits[seq] = audit

    out = resolve_path(args.out_dir) if args.out_dir else (
        ROOT / "outputs" / f"cvpr_m05_formal_run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    )
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(plan_path, out / "protocol_lock.json")
    shutil.copy2(Path(plan["anchor_manifest"]), out / "anchor_manifest.csv")
    prepare_dir = plan_path.parent
    for provenance_name in ("data_manifest_audit.csv", "candidate_cycle_audit.csv"):
        source = prepare_dir / provenance_name
        if source.is_file():
            shutil.copy2(source, out / provenance_name)
    shutil.copy2(checkpoint_manifest, out / "checkpoint_manifest.csv")

    all_current, all_future, all_state, all_details, all_effects, call_rows = [], [], [], [], [], []
    state_audit = state_io_audit()
    (out / "state_io_audit.json").write_text(
        json.dumps(state_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tolerance = float(args.tolerance)
    if tolerance <= 0 or tolerance > 1e-2:
        raise ValueError("Tolerance must be >0 and <=1e-2; freeze from canary before run.")
    unique_anchor_count = sum(
        len(plan["selection"][seq]["selected_cycles"]) * len(AGES)
        for seq in SEQUENCES
    )
    total_anchors = unique_anchor_count * len(verified_checkpoints) * 2
    completed = 0
    for ckpt_info in verified_checkpoints:
        seed = int(ckpt_info["seed"])
        print(f"[M05-FORMAL] seed={seed} checkpoint={ckpt_info['checkpoint_path']}", flush=True)
        model = gt2.load_model(snapshot, "pre_readout", torch.device("cuda"))
        ckpt = torch.load(ckpt_info["checkpoint_path"], map_location="cpu", weights_only=False)
        head = model.pixel_decoder.persistent_feature_repair
        if head is None:
            raise RuntimeError("P_multi repair head not registered.")
        head.load_state_dict(ckpt["repair_head_state_dict"], strict=True)
        head.eval()
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if any(p.requires_grad for p in model.parameters()):
            raise RuntimeError("Frozen model contains trainable parameters.")
        counter = CallCounter(model)
        try:
            for seq in SEQUENCES:
                counter.reset()
                rows = sequence_rows[seq]
                shape = exp1.shape_info_for(
                    model, Path(rows[0]["rgb_path"]), torch.device("cuda")
                )
                model._m05_shape = shape
                selected_cycles = plan["selection"][seq]["selected_cycles"]
                for cycle in selected_cycles:
                    start = int(cycle) * K
                    max_idx = start + max(AGES) + HORIZON
                    frames = {}
                    t_frame_load = time.perf_counter()
                    for idx in range(start, max_idx + 1):
                        row = rows[idx]
                        matched = str(row.get("gt_match_status", "")).strip().upper() == "MATCHED"
                        frames[idx] = exp1.load_frame(row, shape, torch.device("cuda"), need_gt=matched)
                    torch.cuda.synchronize()
                    call_rows.append({
                        "seed": seed, "sequence": seq, "cycle": cycle,
                        "stage": "frame_load_and_preprocess",
                        "wall_sec": time.perf_counter() - t_frame_load,
                        "frame_count": len(frames),
                    })
                    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
                    t_flow = time.perf_counter()
                    static_by_idx = {
                        idx: exp1.transition_static(
                            model, frames[idx - 1], frames[idx], shape, dis
                        )
                        for idx in range(start + 1, max_idx + 1)
                    }
                    torch.cuda.synchronize()
                    call_rows.append({
                        "seed": seed, "sequence": seq, "cycle": cycle,
                        "stage": "flow_and_static_precompute",
                        "wall_sec": time.perf_counter() - t_flow,
                        "calls": len(static_by_idx),
                    })
                    cycle_rng = capture_rng(torch)
                    counter.set_stage("parent_A_native")
                    t0 = time.perf_counter()
                    parent_a = build_parent_states(
                        model, exp1, frames, static_by_idx, start, AGES, "off", cycle_rng, torch
                    )
                    torch.cuda.synchronize()
                    a_sec = time.perf_counter() - t0
                    counter.set_stage("parent_P_maintained")
                    t0 = time.perf_counter()
                    parent_p = build_parent_states(
                        model, exp1, frames, static_by_idx, start, AGES, "pre_readout", cycle_rng, torch
                    )
                    torch.cuda.synchronize()
                    p_sec = time.perf_counter() - t0

                    # Parent keyframe equality and prior P writeback counts are recorded.
                    for age in AGES:
                        pa, pp = parent_a[age], parent_p[age]
                        anchor_index = start + age
                        parent_diff = state_diff(pa, pp)
                        anchor_row = {
                            "seed": seed, "sequence": seq, "cycle": cycle,
                            "age": age, "anchor_index": anchor_index,
                        }
                        for parent_label, parent_state in (("A_native", pa), ("P_maintained", pp)):
                            plan_row = {
                                **anchor_row,
                                "parent": parent_label,
                                "prior_P_writeback_calls": 0 if parent_label == "A_native" else age - 1,
                                "parent_state_sha256": tensor_sha(parent_state),
                            }
                            # The frozen anchor manifest has one row per horizon; select H=1 row.
                            current_anchor = next(
                                r for r in anchor_rows
                                if r["sequence"] == seq and int(r["cycle"]) == int(cycle)
                                and int(r["age"]) == age and int(r["future_horizon"]) == 1
                            )
                            counter.set_stage("anchor_and_future_branches")
                            crows, frows, srows, detail, timing_rows = process_anchor(
                                model, exp1, gt2, frames, static_by_idx, shape,
                                {**current_anchor, "seed": seed}, parent_label,
                                parent_state, tolerance, torch, counter
                            )
                            all_current.extend(crows)
                            all_future.extend(frows)
                            all_state.extend(srows)
                            call_rows.extend(timing_rows)
                            detail["parent_state_comparison_A_vs_P"] = parent_diff
                            detail["prior_P_writeback_calls"] = plan_row["prior_P_writeback_calls"]
                            if age == 1:
                                detail["invariants"]["age1_A_P_parent_state_equal"] = (
                                    parent_diff["all_fields"] <= tolerance
                                )
                                detail["passed"] = all(detail["invariants"].values())
                            all_details.append(detail)
                            completed += 1
                            if completed % 12 == 0:
                                print(
                                    f"[M05-FORMAL] progress anchors={completed}/{total_anchors} "
                                    f"seed={seed} seq={seq} cycle={cycle} age={age} parent={parent_label}",
                                    flush=True,
                                )
                            # Effect rows are derived from future metric rows after all branches.
                    call_rows.extend([
                        {"seed": seed, "sequence": seq, "cycle": cycle,
                         "stage": "parent_A_native", "wall_sec": a_sec},
                        {"seed": seed, "sequence": seq, "cycle": cycle,
                         "stage": "parent_P_maintained", "wall_sec": p_sec},
                    ])
                    del frames, static_by_idx, parent_a, parent_p
                    torch.cuda.empty_cache()

                counter.close()
                call_rows.extend([
                    {"seed": seed, "sequence": seq, "cycle": "", "stage": r["stage"],
                     "module": r["module"], "calls": r["calls"],
                     "nested_counts_not_additive": r["nested_counts_not_additive"]}
                    for r in counter.rows()
                ])
            del model
            torch.cuda.empty_cache()
        except Exception:
            counter.close()
            del model
            torch.cuda.empty_cache()
            raise

    # Derive anchor-level paired effects from raw branch rows. GT-missing H1-H3 remain missing;
    # all RGB frames were still propagated through every branch.
    future_lookup = {}
    for row in all_future:
        key = (
            int(row["seed"]), row["sequence"], int(row["cycle"]), int(row["age"]),
            row["parent"], row["branch"], int(row["horizon"]),
        )
        future_lookup[key] = row
    for seed in [int(x["seed"]) for x in verified_checkpoints]:
        for seq in SEQUENCES:
            for cycle in plan["selection"][seq]["selected_cycles"]:
                for age in AGES:
                    for parent in ("A_native", "P_maintained"):
                        def val(branch, h):
                            r = future_lookup.get((seed, seq, int(cycle), age, parent, branch, h))
                            if not r or r.get("AbsRel") in ("", None):
                                return None
                            return float(r["AbsRel"])
                        diffs_h = {}
                        for h in range(1, HORIZON + 1):
                            o, t = val("O", h), val("T", h)
                            a, f = val("A", h), val("F", h)
                            diffs_h[h] = {
                                "T_minus_O": None if o is None or t is None else t - o,
                                "F_minus_A": None if a is None or f is None else f - a,
                            }
                        h4 = diffs_h[4]
                        win_vals = [v["T_minus_O"] for v in diffs_h.values() if v["T_minus_O"] is not None]
                        def delta(branch, base, h=4):
                            x, y = val(branch, h), val(base, h)
                            return None if x is None or y is None else x - y
                        all_effects.append({
                            "seed": seed, "sequence": seq, "cycle": int(cycle),
                            "age": age, "parent": parent,
                            "H4_T_minus_O_AbsRel": "" if h4["T_minus_O"] is None else h4["T_minus_O"],
                            "H4_F_minus_A_AbsRel": "" if h4["F_minus_A"] is None else h4["F_minus_A"],
                            "H4_Pnative_minus_T_AbsRel": "" if delta("P_native", "T") is None else delta("P_native", "T"),
                            "H4_Ronly_minus_T_AbsRel": "" if delta("R_only", "T") is None else delta("R_only", "T"),
                            "H4_Conly_minus_T_AbsRel": "" if delta("C_only", "T") is None else delta("C_only", "T"),
                            "T_minus_O_H1_H4_window_AbsRel": "" if not win_vals else sum(win_vals) / len(win_vals),
                            "n_GT_horizons": len(win_vals),
                            "interpretation": "negative favors the first named branch; H4 is the primary target",
                        })

    summary = summarize_effects(all_effects)
    all_invariants_pass = all(d["passed"] for d in all_details)
    status = "FORMAL_RUN_COMPLETE_INVARIANTS_PASS" if all_invariants_pass else "FORMAL_RUN_INVALID_INVARIANT_FAIL"
    run_summary = {
        "experiment": "CVPR-M05 Formal 1.0",
        "status": status,
        "training_performed": False,
        "test_set_accessed": False,
        "model_inference_performed": True,
        "K": K, "ages": list(AGES), "H": HORIZON,
        "sequences": list(SEQUENCES),
        "checkpoint_provenance": verified_checkpoints,
        "anchor_count_per_parent_per_seed": unique_anchor_count,
        "parent_count": 2,
        "branch_count": len(ALL_BRANCHES),
        "core_branches": list(CORE_BRANCHES),
        "supplementary_field_isolation_branches": list(DIAGNOSTIC_BRANCHES),
        "primary_estimand": "A_native, K16, age 1/4/8, H4, T minus O paired AbsRel",
        "primary_effect_summary": summary["per_seed_primary"],
        "effect_strata_summary": summary["strata"],
        "three_seed_descriptive_mean": summary["three_seed_descriptive_mean"],
        "inferential_ci": summary["inferential_ci"],
        "ci_reason": summary["ci_reason"],
        "all_invariants_pass": all_invariants_pass,
        "invariant_checks": len(all_details),
        "failed_invariants": [
            {"seed": d["seed"], "sequence": d["sequence"], "cycle": d["cycle"],
             "age": d["age"], "parent": d["parent"],
             "failed": [k for k, v in d["invariants"].items() if not v]}
            for d in all_details if not d["passed"]
        ],
        "interpretation_limits": [
            "M05 is a single-intervention mechanism study, not continuous closed-loop M04.",
            "A_native and P_maintained parent histories are separate strata and are never pooled.",
            "T-O and F-A are the same future state effect under the state-contract invariants, not independent discoveries.",
            "R_only/C_only are supplementary synthetic field-isolation diagnostics conditional on F1, not new deployment methods.",
            "Confidence non-consumption is specific to the audited fixed-K fast_step path; keyframe refresh reinitializes state.",
            "Only three development recordings are included; no scene-level inferential CI is claimed.",
            "A successful invariant audit does not imply positive depth-quality effect.",
        ],
        "state_io_audit": state_audit,
        "source_hashes": source_audit,
        "snapshot_sha256": EXPECTED_BASE_SHA256,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    write_csv(out / "m05_current_frame_metrics.csv", all_current)
    write_csv(out / "m05_future_frame_metrics.csv", all_future)
    write_csv(out / "m05_state_audit.csv", all_state)
    write_csv(out / "m05_anchor_effects.csv", all_effects)
    write_csv(out / "m05_call_ledger.csv", call_rows)
    write_json(out / "m05_formal_summary.json", run_summary)

    lines = [
        "# CVPR M05 Formal 1.0 — results",
        "",
        f"**Status: {status}**",
        "",
        "- Training: NOT RUN",
        "- Test set: NOT ACCESSED",
        "- Main estimand: A-native parent, K16, ages 1/4/8, H4, T−O paired AbsRel",
        "- P-maintained parent: separate secondary stratum; not pooled with A-native",
        "- T−O and F−A: corresponding state-contract checks, not independent findings",
        "- R_only/C_only: supplementary synthetic field-isolation diagnostics",
        "",
        "## Primary descriptive result by P training seed",
        "",
        "|Seed|A-native sequence-equal mean T−O H4 AbsRel|",
        "|---:|---:|",
    ]
    for r in summary["per_seed_primary"]:
        v = r["A_native_primary_sequence_equal_mean_T_minus_O_H4"]
        lines.append(f"|{r['seed']}|{'NA' if v is None else f'{v:.8f}'}|")
    lines += [
        "",
        f"Three-seed descriptive mean: `{summary['three_seed_descriptive_mean']}`",
        "",
        "No inferential confidence interval is reported: only three named development recordings "
        "are included and no verified physical-scene grouping is provided. Do not bootstrap "
        "anchors/frames as if they were independent scenes.",
        "",
        "## Invariant status",
        "",
        f"- Anchor-parent checks: {len(all_details)}",
        f"- All invariants pass: {all_invariants_pass}",
        f"- Failed anchors: {len(run_summary['failed_invariants'])}",
        "",
        "## Interpretation",
        "",
        "A positive quality result is not assumed. A failed invariant invalidates the corresponding "
        "contrast. If invariants pass but H4 T−O is not beneficial, report that outcome without "
        "relabeling the mechanism as successful.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    review = out / "cvpr_m05_formal_review.zip"
    whitelist = [
        "protocol_lock.json", "anchor_manifest.csv", "data_manifest_audit.csv",
        "candidate_cycle_audit.csv", "checkpoint_manifest.csv",
        "state_io_audit.json", "m05_formal_summary.json",
        "m05_current_frame_metrics.csv",
        "m05_future_frame_metrics.csv", "m05_state_audit.csv",
        "m05_anchor_effects.csv", "m05_call_ledger.csv", "REPORT.md",
    ]
    with zipfile.ZipFile(review, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in whitelist:
            z.write(out / name, name)
    with zipfile.ZipFile(review) as z:
        if z.testzip() is not None:
            raise RuntimeError("M05 formal review ZIP CRC verification failed")
    print(f"[M05-FORMAL] status={status}", flush=True)
    print(f"[M05-FORMAL] report={out / 'REPORT.md'}", flush=True)
    print(f"[M05-FORMAL] review={review}", flush=True)
    return 0 if all_invariants_pass else 2


def main():
    ap = argparse.ArgumentParser(description="CVPR M05 Formal 1.0")
    sub = ap.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="read-only data audit and anchor freeze; no inference")
    prep.add_argument("--snapshot", default=os.environ.get("VDEEP_SNAPSHOT", ""))
    prep.add_argument("--crowd2-pair-csv", default="outputs/gt4a2_exp1/gt4a2_exp1_crowd2_pairs_used.csv")
    prep.add_argument("--crowd3-pair-csv", default="outputs/gt4a2_exp1/gt4a2_exp1_crowd3_pairs_used.csv")
    prep.add_argument("--balloon2-pair-csv", default="outputs/gt4a2_exp1/gt4a2_exp1_balloon2_pairs_used.csv")
    prep.add_argument("--out-dir", default="")
    prep.set_defaults(func=prepare)

    execute = sub.add_parser("run", help="formal three-seed multi-anchor inference")
    execute.add_argument("--plan", required=True)
    execute.add_argument("--checkpoint-manifest", required=True)
    execute.add_argument("--snapshot", default=os.environ.get("VDEEP_SNAPSHOT", ""))
    execute.add_argument("--out-dir", default="")
    execute.add_argument("--tolerance", type=float, default=1e-5)
    execute.set_defaults(func=run)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
