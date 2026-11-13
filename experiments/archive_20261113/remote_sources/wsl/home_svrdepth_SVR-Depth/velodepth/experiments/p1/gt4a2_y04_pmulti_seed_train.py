#!/usr/bin/env python3
"""Train one additional Y04 P_multi seed for CVPR M05 Formal 1.0.

This runner reuses the pinned Y04-Fix3 implementation and its audited helpers.
It trains ONLY the P_multi arm (24 multi-scene cycle slots, 1000 updates),
leaving the original Y04 script and all C3/Q03 helper source files unchanged.

The authoritative outer seed is recorded as y04_outer_seed and is independently
checkable from y04_sample_order_sha256. The nested C3 checkpoint `seed` field
remains the historical C3 module seed (20261024) by design; it is not the Y04
outer training seed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
P1 = ROOT / "experiments" / "p1"
if str(P1) not in sys.path:
    sys.path.insert(0, str(P1))

import gt4a2_y04_fair_multiscene as y04  # noqa: E402

RUNNER_VERSION = "M05-Pmulti-SeedRunner-1.0"
ALLOWED_SEEDS = (20261101, 20261102)
FORMAL_STEPS = 1000
EXPECTED_BASE_SHA256 = "615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e"
KNOWN_P_CHECKPOINT = Path(
    "outputs/gt4a2_y04_validation_fix3_20261025_225902/training/P_multi/gt4a2_c3_C_final_step1000.pt"
)
KNOWN_P_CHECKPOINT_SHA256 = "77a86cdf523f776608b6b591d6c6552002b90398e85d00d3a3fe29afe14a0a24"
EXPECTED_SOURCE_SHA256 = {
    "experiments/p1/gt4a2_y04_fair_multiscene.py": "2a24e5a67487d55177e3374f9c6be5e92795b508dfcdb299cefe46e37cc31d4f",
    "experiments/p1/gt4a2_c3_closed_loop.py": "76327208acac4543b2a87ccba8b7af435e4c5ce65267b42052e51b435cbad7f5",
    "experiments/p1/gt4a2_q0_3_necessity_controls.py": "5ac9e648dbae36e17ebf64cd8891c226052f1a71171acf326f7dc6b591ff2390",
    "experiments/p1/gt4a2_exp1_train_eval.py": "c24441cb405465c1a16a86127b51aade8f9af69b5d368f43d830d0e759ee3d7f",
    "experiments/p1/gt2_fixed_clip_repair_train.py": "bf7cd2021b1d0bbf66d75502dba6afed87f4c8ff3ef58ff7b18f56ac28044192",
    "velodepth/models/decoder_velo.py": "35ca12629accd1a2a77595f505165fa17dffab07c763a8b639604285b94bdb08",
    "velodepth/models/persistent_feature_repair.py": "6dfc8b0c5565ea9139b8c266c5ef337b3309ce0fde2f365428fc44390b84117c",
}
EXPECTED_ORDER_SHA256 = {
    20261025: "aa7aff6213fdbf9ddce8703e2eb707e8cb8eab495a575f96698d9d25d9b9d934",
    20261101: "afed28e25737aa6355e0aad93a621dd24dae1b7b4718219ee0fae0efc7d8f47f",
    20261102: "d94beec434fc78a986875bc99d373585b241a4b962d23dd20002abc7bc4cf684",
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_pinned_sources() -> dict[str, str]:
    actuals: dict[str, str] = {}
    failures = []
    for rel, expected in EXPECTED_SOURCE_SHA256.items():
        path = ROOT / rel
        if not path.is_file():
            failures.append(f"MISSING {rel}")
            continue
        actual = file_sha256(path)
        actuals[rel] = actual
        if actual != expected:
            failures.append(f"HASH {rel}: expected={expected} actual={actual}")
    if failures:
        raise RuntimeError("Pinned source baseline failed:\n" + "\n".join(failures))
    print(f"[PASS] pinned source baseline: {len(actuals)}/7 SHA256 matched", flush=True)
    return actuals


def sample_order(seed: int, n_slots: int = 24, steps: int = FORMAL_STEPS) -> list[int]:
    rng = random.Random(int(seed))
    order: list[int] = []
    while len(order) < steps:
        block = list(range(n_slots))
        rng.shuffle(block)
        order.extend(block)
    return order[:steps]


def sample_order_sha256(seed: int) -> str:
    order = sample_order(seed)
    return hashlib.sha256(",".join(map(str, order)).encode()).hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> int:
    for seed in (20261025, *ALLOWED_SEEDS):
        actual = sample_order_sha256(seed)
        expected = EXPECTED_ORDER_SHA256[seed]
        if actual != expected:
            raise RuntimeError(f"sample-order self-test failed for {seed}: {actual} != {expected}")
        order = sample_order(seed)
        y04_order = y04.balanced_order(24, FORMAL_STEPS, seed)
        if order != y04_order:
            raise RuntimeError(f"Y04 helper order differs for seed {seed}")
        if len(order) != FORMAL_STEPS or set(order) != set(range(24)):
            raise RuntimeError(f"sample-order geometry failed for seed {seed}")
        counts = [order.count(i) for i in range(24)]
        if min(counts) < 41 or max(counts) > 42:
            raise RuntimeError(f"sample-order balance failed for seed {seed}: {counts}")
        print(f"[PASS] seed={seed} order_sha256={actual}")
    print(f"[{RUNNER_VERSION}] SELF-TEST PASS; no torch/CUDA/training used")
    return 0


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=RUNNER_VERSION)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--seed", type=int, choices=ALLOWED_SEEDS)
    p.add_argument("--snapshot", default=os.environ.get("VDEEP_SNAPSHOT", ""))
    p.add_argument("--crowd2-pair-csv", default="outputs/gt4a2_exp1/gt4a2_exp1_crowd2_pairs_used.csv")
    p.add_argument("--crowd3-pair-csv", default="outputs/gt4a2_exp1/gt4a2_exp1_crowd3_pairs_used.csv")
    p.add_argument("--balloon2-pair-csv", default="outputs/gt4a2_exp1/gt4a2_exp1_balloon2_pairs_used.csv")
    p.add_argument("--previous-config", default="outputs/gt4a2_exp1/gt4a2_exp1_config.json")
    p.add_argument("--c3-frame-csv", default="outputs/gt4a2_c3/gt4a2_c3_full_sequence_frame_metrics.csv")
    p.add_argument("--c3-output-dir", default="outputs/gt4a2_c3")
    p.add_argument("--c-checkpoint", default="outputs/gt4a2_c3/gt4a2_c3_C_final_step1000.pt")
    p.add_argument("--d-checkpoint", default="outputs/gt4a2_c3/gt4a2_c3_D_final_step1000.pt")
    p.add_argument("--out-dir", default="")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


def run(args: argparse.Namespace) -> int:
    if args.seed not in ALLOWED_SEEDS:
        raise RuntimeError(f"--seed is required and must be one of {ALLOWED_SEEDS}")
    if args.preflight_only and args.smoke_only:
        raise RuntimeError("--preflight-only and --smoke-only are mutually exclusive")
    if args.resume and (args.preflight_only or args.smoke_only):
        raise RuntimeError("--resume is only valid for a formal training run")
    if not args.snapshot:
        raise RuntimeError("VDEEP_SNAPSHOT is unset; pass --snapshot or export it")

    snapshot = Path(args.snapshot).expanduser().resolve()
    if not (snapshot / "config.json").is_file() or not (snapshot / "model.safetensors").is_file():
        raise FileNotFoundError(f"VeloDepth snapshot is incomplete: {snapshot}")
    actual_base_sha = file_sha256(snapshot / "model.safetensors")
    if actual_base_sha != EXPECTED_BASE_SHA256:
        raise RuntimeError(
            f"Pinned VeloDepth base SHA256 mismatch: {actual_base_sha} != {EXPECTED_BASE_SHA256}"
        )
    source_hashes = verify_pinned_sources()
    known_checkpoint = ROOT / KNOWN_P_CHECKPOINT
    if not known_checkpoint.is_file():
        raise FileNotFoundError(f"Known seed-20261025 P_multi checkpoint missing: {known_checkpoint}")
    known_sha = file_sha256(known_checkpoint)
    if known_sha != KNOWN_P_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Known seed-20261025 P_multi checkpoint SHA mismatch: {known_sha} != {KNOWN_P_CHECKPOINT_SHA256}"
        )
    print("[PASS] known M05 seed 20261025 checkpoint SHA256 matched", flush=True)
    args.snapshot = str(snapshot)

    if not args.out_dir:
        suffix = "_smoke" if args.smoke_only else ""
        args.out_dir = f"outputs/cvpr_m05_pmulti_seed_{args.seed}{suffix}"
    out_dir = Path(args.out_dir).expanduser().resolve()
    marker = out_dir / "y04_summary.json"
    if marker.exists() and not args.preflight_only:
        raise RuntimeError(f"Completion marker exists; refusing overwrite: {marker}")

    # Load the exact pinned Y04 helper stack, then override only the Y04 outer seed.
    y04.load_runtime()
    y04.SEED = int(args.seed)
    print(f"[{RUNNER_VERSION}] seed={args.seed}", flush=True)
    print(f"[Y04 BASE] version={y04.VERSION} sha256={file_sha256(Path(y04.__file__))}", flush=True)
    print(f"[ORDER] expected_y04_sample_order_sha256={sample_order_sha256(args.seed)}", flush=True)

    a = SimpleNamespace(
        snapshot=str(snapshot),
        crowd2_pair_csv=args.crowd2_pair_csv,
        crowd3_pair_csv=args.crowd3_pair_csv,
        balloon2_pair_csv=args.balloon2_pair_csv,
        previous_config=args.previous_config,
        c3_frame_csv=args.c3_frame_csv,
        c3_output_dir=args.c3_output_dir,
        c_checkpoint=args.c_checkpoint,
        d_checkpoint=args.d_checkpoint,
        out_dir=str(out_dir),
        resume=bool(args.resume),
        smoke_only=bool(args.smoke_only),
        steps=FORMAL_STEPS,
    )

    qa = y04.args_for_q03(a)
    pre, manifests, _historical_refs, _c3_order, _c3_order_sha, context = y04.q03.build_preflight(
        qa, list(y04.SEQUENCES)
    )

    single_slots = [("crowd2", i) for i in range(24)]
    multi_slots_formal = (
        [("crowd2", i) for i in range(8)]
        + [("crowd3", i) for i in range(8)]
        + [("balloon2", i) for i in range(8)]
    )
    split_rows = []
    for seq, rows in manifests.items():
        train_end = y04.TRAIN_CYCLES[seq] * y04.REFRESH
        for idx, rec in enumerate(rows):
            matched = str(rec.get("gt_match_status", "")).upper() == "MATCHED"
            split_rows.append({
                "sequence": seq,
                "frame_index_zero_based": idx,
                "rgb_file": Path(rec["rgb_path"]).name,
                "gt_file": Path(rec["depth_path"]).name if matched else "",
                "gt_matched": str(matched).lower(),
                "multi_scene_split": "train_fit" if idx < train_end else "temporal_tail_eval",
                "used_for_fit_multi_scene": str(idx < train_end).lower(),
            })

    pair_csvs = {seq: Path(getattr(a, f"{seq}_pair_csv")).resolve() for seq in y04.SEQUENCES}
    source_modules = (y04.q03, y04.c3, y04.exp1, y04.gt2)
    source_sha = {str(ROOT / rel): digest for rel, digest in source_hashes.items()}
    source_sha.update({str(Path(m.__file__).resolve()): file_sha256(Path(m.__file__).resolve())
                       for m in source_modules})
    source_sha[str(ROOT / "velodepth" / "models" / "decoder_velo.py")] = file_sha256(
        ROOT / "velodepth" / "models" / "decoder_velo.py"
    )
    source_sha[str(ROOT / "velodepth" / "models" / "persistent_feature_repair.py")] = file_sha256(
        ROOT / "velodepth" / "models" / "persistent_feature_repair.py"
    )

    runner_sha = file_sha256(Path(__file__).resolve())
    base_sha = file_sha256(snapshot / "model.safetensors")
    init_hash = context["initial_head_hash"]
    identity = {
        "version": y04.VERSION,
        "script_sha256": file_sha256(Path(y04.__file__).resolve()),
        "m05_seed_runner_version": RUNNER_VERSION,
        "m05_seed_runner_sha256": runner_sha,
        "y04_outer_seed": int(args.seed),
        "optimizer_order_seed": int(args.seed),
        "training_scope": "P_multi_only",
        "pretrained_weights_sha256": base_sha,
        "common_initial_head_sha256": init_hash,
        "pair_csv_sha256": {seq: file_sha256(path) for seq, path in pair_csvs.items()},
        "source_sha256": source_sha,
        "single_scene_slots_reference": [list(x) for x in single_slots],
        "multi_scene_slots": [list(x) for x in multi_slots_formal],
        "temporal_tail_eval_start": y04.EVAL_START,
        "formal_steps": FORMAL_STEPS,
        "smoke_only": bool(args.smoke_only),
        "persistent_training_mode": "pre_readout",
        "checkpoint_selection": "final step only; no validation-based selection",
    }
    y04.ensure_run_identity(out_dir / "y04_run_identity.json", identity, resume=args.resume)

    protocol = {
        "status": "PREFLIGHT_PASS",
        "experiment": "Y04 P_multi additional seed for CVPR M05",
        "version": y04.VERSION,
        "runner_version": RUNNER_VERSION,
        "runner_sha256": runner_sha,
        "y04_script_sha256": identity["script_sha256"],
        "y04_outer_seed": int(args.seed),
        "expected_y04_sample_order_sha256": sample_order_sha256(args.seed),
        "nested_c3_seed_metadata": 20261024,
        "nested_c3_seed_is_authoritative": False,
        "snapshot": str(snapshot),
        "pretrained_weights_sha256": base_sha,
        "common_initial_head_sha256": init_hash,
        "sequences": list(y04.SEQUENCES),
        "refresh_interval": y04.REFRESH,
        "formal_steps": FORMAL_STEPS,
        "multi_scene_slots": [list(x) for x in multi_slots_formal],
        "multi_scene_fit": "Crowd2 0..255 + Crowd3 0..255 + Balloon2 0..255 (8 cycles per sequence)",
        "training_arm": "P_multi",
        "training_mode": "pre_readout",
        "checkpoint_selection": "final step only; no validation-based selection",
        "historical_q03_preflight": pre,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "y04_preflight.json", protocol)
    write_csv(out_dir / "y04_split_manifest.csv", split_rows)
    print("[PASS] M05 P_multi preflight: 24 slots; 8 cycles/scene; common historical C3 initialization", flush=True)

    if args.preflight_only:
        print("[PREFLIGHT ONLY] no cache construction, CUDA training, inference, or checkpoint writes", flush=True)
        return 0

    torch = y04.torch
    np = y04.np
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for M05 P_multi training")
    torch.set_num_threads(4)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device("cuda")
    readers, cache_meta = y04.build_caches(a, context, manifests, out_dir, device, args.smoke_only)
    write_json(out_dir / "y04_training_cycle_manifest.json", cache_meta)

    if args.smoke_only:
        slots = [("crowd2", 0), ("crowd3", 0), ("balloon2", 0)]
        order = [0, 1, 2]
        steps = 3
    else:
        slots = multi_slots_formal
        order = sample_order(int(args.seed))
        if order != y04.balanced_order(24, FORMAL_STEPS, int(args.seed)):
            raise RuntimeError("Local seed order differs from the pinned Y04 balanced_order helper")
        steps = FORMAL_STEPS
    reader = y04.SceneCycleReader(readers, slots)
    unique_gt = sum(int(readers[seq].load(cycle)["n_supervised"]) for seq, cycle in slots)
    gt_per_slot = [int(readers[seq].load(cycle)["n_supervised"]) for seq, cycle in slots]
    exposure_gt = sum(gt_per_slot[index] for index in order)
    write_json(out_dir / "y04_budget_manifest.json", {
        "y04_outer_seed": int(args.seed),
        "training_scope": "P_multi_only",
        "multi_scene_unique_gt_frames_across_slots": unique_gt,
        "multi_scene_optimizer_steps": len(order),
        "multi_scene_slots": len(slots),
        "sampled_gt_frame_exposures": exposure_gt,
        "sample_order_sha256": hashlib.sha256(",".join(map(str, order)).encode()).hexdigest(),
        "gt_exposure_must_be_reported": True,
    })

    init_state = context["initial_head_state"]
    result = y04.train_persistent(
        a, "P_multi", reader.slots, reader, order, init_state,
        out_dir, device, bool(args.smoke_only), steps
    )

    if args.smoke_only:
        write_json(marker, {
            "status": "SMOKE_PASS",
            "runner_version": RUNNER_VERSION,
            "y04_outer_seed": int(args.seed),
            "training_performed": True,
            "formal_checkpoints_saved": False,
            "trained_arms": ["P_multi"],
            "optimizer_updates": steps,
            "training_result": result,
            "inference_performed": False,
        })
        print(f"[SMOKE PASS] seed={args.seed}; no formal checkpoint saved", flush=True)
        return 0

    snapshot_hash = base_sha
    y04.write_checkpoint_provenance(
        out_dir / "training" / "P_multi", "C", "persistent",
        list(y04.SEQUENCES), slots, order, init_hash, snapshot_hash,
        unique_gt, "Y04_P_multi", pair_csvs,
    )

    checkpoint = out_dir / "training" / "P_multi" / "gt4a2_c3_C_final_step1000.pt"
    summary_path = out_dir / "training" / "P_multi" / "gt4a2_c3_C_training_summary.json"
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt.update({
        "y04_outer_seed": int(args.seed),
        "y04_optimizer_order_seed": int(args.seed),
        "m05_seed_runner_version": RUNNER_VERSION,
        "m05_seed_runner_sha256": runner_sha,
        "y04_sample_order_sha256": sample_order_sha256(int(args.seed)),
    })
    torch.save(ckpt, checkpoint)
    del ckpt

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    final_checkpoint_sha = file_sha256(checkpoint)
    summary.update({
        "checkpoint_sha256": final_checkpoint_sha,
        "y04_outer_seed": int(args.seed),
        "y04_optimizer_order_seed": int(args.seed),
        "y04_sample_order_sha256": sample_order_sha256(int(args.seed)),
        "m05_seed_runner_version": RUNNER_VERSION,
        "m05_seed_runner_sha256": runner_sha,
    })
    write_json(summary_path, summary)

    training_compute = {
        "status": "TRAINING_COMPLETE",
        "training_scope": "P_multi_only",
        "y04_outer_seed": int(args.seed),
        "steps_per_arm": FORMAL_STEPS,
        "training_results": {"P_multi": summary},
        "multi_scene_unique_gt_frames": unique_gt,
        "sampled_gt_frame_exposures": exposure_gt,
        "sample_order_sha256": sample_order_sha256(int(args.seed)),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": final_checkpoint_sha,
    }
    write_json(out_dir / "y04_training_compute.json", training_compute)
    write_json(marker, {
        "status": "TRAINING_ONLY_COMPLETE",
        "experiment": "Y04 P_multi additional seed for CVPR M05",
        "runner_version": RUNNER_VERSION,
        "y04_outer_seed": int(args.seed),
        "nested_c3_seed_metadata": 20261024,
        "nested_c3_seed_is_authoritative": False,
        "trained_arms": ["P_multi"],
        "training_steps": FORMAL_STEPS,
        "evaluation_performed": False,
        "inference_performed": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": final_checkpoint_sha,
        "y04_sample_order_sha256": sample_order_sha256(int(args.seed)),
        "formal_m05_compatible": True,
    })
    print("========== M05 P_multi seed training complete ==========")
    print(f"SEED: {args.seed}")
    print(f"ORDER_SHA256: {sample_order_sha256(int(args.seed))}")
    print(f"CHECKPOINT: {checkpoint}")
    print(f"CHECKPOINT_SHA256: {final_checkpoint_sha}")
    print("TRAINING: 1000/1000 optimizer updates")
    print("EVALUATION: not run; M05 formal inference remains a separate frozen stage")
    return 0


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[M05_PMULTI_SEED_FAIL] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
