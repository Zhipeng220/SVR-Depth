import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from collections import Counter

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_crosssource_v2_E4_baseline as base

RUNS = {
    "E4-C": (
        "outputs/crosssource_v2_E4_20261104_C_s20261103_1000step",
        "C_tartanair_kitti.json",
        {"tartanair": 0.5, "kitti_sceneflow": 0.5},
    ),
    "E5-A": (
        "outputs/crosssource_v2_E5A_20261104_A_tartanair_s20261103_1000step",
        "A_tartanair_only.json",
        {"tartanair": 1.0},
    ),
    "E5-B": (
        "outputs/crosssource_v2_E5B_20261104_B_kitti_s20261103_1000step",
        "B_kitti_only.json",
        {"kitti_sceneflow": 1.0},
    ),
}

CONFIG_DIR = ROOT / "configs/crosssource_v2/generated_20261103"

def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    require(torch.cuda.is_available(), "CUDA unavailable")

    snapshot = base.SNAPSHOT
    snapshot_hash = sha256(snapshot / "model.safetensors")

    require(
        snapshot_hash ==
        "615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e",
        "VeloDepth snapshot hash mismatch"
    )

    print("========== E6 IDENTITY AUDIT ==========")

    loaded = {}
    common_init_hash = None

    for name, (run_rel, config_name, weights) in RUNS.items():
        run = ROOT / run_rel
        config = CONFIG_DIR / config_name

        require(run.is_dir(), f"Missing run: {run}")
        require(config.is_file(), f"Missing config: {config}")

        identity = load_json(run / "run_identity.json")

        expected = {
            "protocol": "SVR-Depth-CrossSource-v2",
            "method": "P_multi",
            "seed": 20261103,
            "initialization_seed": 20261103,
            "steps": 1000,
            "window_frames": 20,
            "tbptt_frames": 4,
            "dataset_weights": weights,
            "snapshot_sha256": snapshot_hash,
            "config_sha256": sha256(config),
        }

        for k, v in expected.items():
            require(
                identity.get(k) == v,
                f"{name} identity mismatch: {k}"
            )

        loaded[name] = {}

        for variant in ("best", "last"):
            path = run / f"{variant}_checkpoint.pt"

            ckpt = torch.load(
                path,
                map_location="cpu",
                weights_only=False
            )

            require(
                ckpt["identity"] == identity,
                f"{name} {variant} checkpoint identity mismatch"
            )

            init_hash = ckpt["init_head_sha256"]

            if common_init_hash is None:
                common_init_hash = init_hash

            require(
                init_hash == common_init_hash,
                f"{name} {variant} initialization mismatch"
            )

            loaded[name][variant] = ckpt

            print(
                f"[PASS] {name} {variant}: "
                f"step={ckpt['step']} "
                f"init={init_hash[:12]}"
            )

    print("\n========== BUILD COMMON VALIDATION ==========")

    common_config = CONFIG_DIR / "C_tartanair_kitti.json"

    pool = base.tr.CrossSourcePool.from_config(
        common_config,
        split="val",
        verify_files=True
    )

    refs = base.tr.build_val_refs(
        pool,
        window_frames=20,
        max_per_dataset=8,
        seed=20261104
    )

    counts = Counter(r.dataset_id for r in refs)

    expected_datasets = {"kitti_sceneflow", "tartanair"}

    require(
        set(counts) == expected_datasets
        and all(counts[d] > 0 for d in expected_datasets),
        f"Validation must contain both datasets: {counts}"
    )

    print("Actual validation distribution:", dict(counts))
    print("Total validation windows:", len(refs))


    print("Validation windows:", len(refs))
    print("Dataset distribution:", dict(counts))

    out = ROOT / args.out_dir

    if not out.is_absolute():
        out = ROOT / args.out_dir

    require(not out.exists(), f"Output already exists: {out}")
    out.mkdir(parents=True)

    save_json(
        out / "validation_refs.json",
        [base.tr.ref_to_dict(r) for r in refs]
    )

    device = torch.device("cuda")
    results = {}
    metadata = {}

    def evaluate_checkpoint(label, ckpt):
        print(f"\n========== EVALUATE {label} ==========")

        model = base.make_model(
            "pre_readout",
            device,
            ckpt["head_state"]
        )

        result = base.evaluate(model, pool, refs, device)

        del model
        torch.cuda.empty_cache()

        require(
            set(result["per_dataset"]) ==
            {"tartanair", "kitti_sceneflow"},
            f"{label} missing dataset results"
        )

        require(
            result["windows"] == dict(counts),
            f"{label} window count mismatch"
        )

        results[label] = result

        print("Macro:", result["macro_loss"])
        print("Per dataset:", result["per_dataset"])

        return result

    # ------------------------------------------------
    # 1. E4 reproduction safeguard
    # ------------------------------------------------

    e4_run = ROOT / RUNS["E4-C"][0]

    e4_best = loaded["E4-C"]["best"]
    e4_last = loaded["E4-C"]["last"]

    evaluate_checkpoint("E4-C_Best_700", e4_best)
    evaluate_checkpoint("E4-C_Last_1000", e4_last)

    with open(
        e4_run / "training_curve.csv",
        encoding="utf-8-sig"
    ) as f:
        rows = list(csv.DictReader(f))

    last_row = next(
        r for r in rows if int(r["step"]) == 1000
    )

    expected_last = {
        "macro_loss": float(last_row["val_macro_loss"]),
        "per_dataset": json.loads(last_row["val_per_dataset"])
    }

    ok_best = base.check_reproduction(
        "E4 Best",
        results["E4-C_Best_700"],
        e4_best["val"]
    )

    ok_last = base.check_reproduction(
        "E4 Last",
        results["E4-C_Last_1000"],
        expected_last
    )

    reproduction = {
        "best_pass": ok_best,
        "last_pass": ok_last,
        "expected_best": e4_best["val"],
        "actual_best": results["E4-C_Best_700"],
        "expected_last": expected_last,
        "actual_last": results["E4-C_Last_1000"]
    }

    save_json(out / "reproduction_check.json", reproduction)

    require(
        ok_best and ok_last,
        "E4 reproduction failed. Stop E6 comparison."
    )

    print("\n[E4 REPRODUCTION PASS]")

    # ------------------------------------------------
    # 2. Evaluate E5 single-source models
    # ------------------------------------------------

    for name in ("E5-A", "E5-B"):
        for variant in ("best", "last"):
            ckpt = loaded[name][variant]

            label = (
                f"{name}_"
                f"{variant.capitalize()}_"
                f"{ckpt['step']}"
            )

            evaluate_checkpoint(label, ckpt)

    # ------------------------------------------------
    # 3. Initial P_multi baseline
    # ------------------------------------------------

    print("\n========== P_MULTI INITIAL ==========")

    model = base.make_model(
        "pre_readout",
        device,
        seed=20261103
    )

    initial_head = model.pixel_decoder.persistent_feature_repair

    require(
        base.tr.state_sha256(initial_head.state_dict())
        == common_init_hash,
        "Initial P_multi head hash mismatch"
    )

    results["P_multi_Initial"] = base.evaluate(
        model, pool, refs, device
    )

    del model
    torch.cuda.empty_cache()

    # ------------------------------------------------
    # 4. Original VeloDepth baseline
    # ------------------------------------------------

    print("\n========== ORIGINAL VELODEPTH ==========")

    model = base.make_model("off", device)

    results["VeloDepth_Original"] = base.evaluate(
        model, pool, refs, device
    )

    del model
    torch.cuda.empty_cache()

    # ------------------------------------------------
    # 5. Save final results
    # ------------------------------------------------

    save_json(out / "evaluation_results.json", {
        "protocol": "E6 Common CrossSource Validation",
        "validation_windows": len(refs),
        "validation_distribution": dict(counts),
        "snapshot_sha256": snapshot_hash,
        "init_head_sha256": common_init_hash,
        "reproduction": reproduction,
        "results": results
    })

    with open(
        out / "evaluation_summary.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "model",
            "dataset",
            "validation_loss",
            "macro_loss",
            "supervised_frames",
            "windows"
        ])

        for name, result in results.items():
            for ds, value in result["per_dataset"].items():
                writer.writerow([
                    name,
                    ds,
                    value,
                    result["macro_loss"],
                    result["supervised_frames"][ds],
                    result["windows"][ds]
                ])

    print("\n========== E6 FINAL SUMMARY ==========")

    for name, result in results.items():
        print(
            f"{name:25s} "
            f"macro={result['macro_loss']:.8f}"
        )

        for ds, value in result["per_dataset"].items():
            print(f"    {ds}: {value:.8f}")

    print("\n[E6 PASS]")
    print("Output:", out)

if __name__ == "__main__":
    main()
