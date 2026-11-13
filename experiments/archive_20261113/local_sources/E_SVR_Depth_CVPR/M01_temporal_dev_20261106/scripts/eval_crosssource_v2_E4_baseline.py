import csv
import json
import math
import sys
from pathlib import Path

import torch
import cv2

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "experiments/crosssource_v2/legacy_helpers"

sys.path[:0] = [
    str(ROOT),
    str(ROOT / "scripts"),
    str(HELPERS),
]

import train_crosssource_v2 as tr
import gt2_fixed_clip_repair_train as gt2
import gt4a2_exp1_train_eval as exp1

RUN = ROOT / "outputs/crosssource_v2_E4_20261104_C_s20261103_1000step"
CONFIG = ROOT / "configs/crosssource_v2/generated_20261103/C_tartanair_kitti.json"

SNAPSHOT = Path(
    "/root/autodl-tmp/SVR-Depth/hf_cache/hub/"
    "models--lpiccinelli--velodepth/snapshots/"
    "2b2ee9fe44d0b261e736653042cecd7693f99c8c"
)

OUT = ROOT / "outputs/crosssource_v2_E4_eval_20261104_v1"

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def evaluate(model, pool, refs, device):
    model.eval()

    head = getattr(
        model.pixel_decoder,
        "persistent_feature_repair",
        None
    )
    if head is not None:
        head.eval()

    values = {}
    frame_counts = {}
    window_counts = {}

    with torch.no_grad():
        for index, ref in enumerate(refs, 1):
            window = tr.prepare_window(
                model, pool, ref, device, gt2, exp1, cv2
            )

            frames = window["frames"]
            statics = window["statics"]

            radius, confidence, features = tr.move_state(
                window["keyframe_state"], device
            )
            rays = window["keyframe_state"]["rays_map"].to(device)

            prev = frames[0]

            for offset in range(1, len(frames)):
                curr = frames[offset]

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=True
                ):
                    radius, confidence, features = exp1.one_fast_step(
                        model,
                        prev,
                        curr,
                        statics[offset - 1],
                        radius,
                        confidence,
                        features
                    )

                    if curr["gt"] is not None:
                        pred = gt2.make_z_depth(
                            radius,
                            rays,
                            window["padded_shape"],
                            window["paddings"],
                            model.interpolation_mode
                        )

                        loss, _ = gt2.masked_log_depth_loss(
                            pred, curr["gt"]
                        )

                        require(
                            torch.isfinite(loss).all().item(),
                            f"Non-finite evaluation loss: {ref}"
                        )

                        ds = ref.dataset_id
                        values.setdefault(ds, []).append(
                            float(loss.item())
                        )
                        frame_counts[ds] = frame_counts.get(ds, 0) + 1

                prev = curr

            ds = ref.dataset_id
            window_counts[ds] = window_counts.get(ds, 0) + 1

            print(
                f"[EVAL] {index}/{len(refs)} "
                f"dataset={ds} scene={ref.scene_id}",
                flush=True
            )

    require(bool(values), "No supervised validation frames")

    per_dataset = {
        k: sum(v) / len(v)
        for k, v in sorted(values.items())
    }

    return {
        "per_dataset": per_dataset,
        "macro_loss": sum(per_dataset.values()) / len(per_dataset),
        "supervised_frames": frame_counts,
        "windows": window_counts
    }

def make_model(mode, device, head_state=None, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    model = gt2.load_model(SNAPSHOT, mode, device)

    if head_state is not None:
        head = getattr(
            model.pixel_decoder,
            "persistent_feature_repair",
            None
        )
        require(head is not None, "P_multi head is missing")
        head.load_state_dict(head_state, strict=True)

    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model

def check_reproduction(label, actual, expected, tolerance=5e-4):
    errors = []

    if not math.isclose(
        actual["macro_loss"],
        float(expected["macro_loss"]),
        rel_tol=0,
        abs_tol=tolerance
    ):
        errors.append(
            f"macro: actual={actual['macro_loss']}, "
            f"expected={expected['macro_loss']}"
        )

    for ds, expected_value in expected["per_dataset"].items():
        actual_value = actual["per_dataset"].get(ds)

        if actual_value is None or not math.isclose(
            actual_value,
            float(expected_value),
            rel_tol=0,
            abs_tol=tolerance
        ):
            errors.append(
                f"{ds}: actual={actual_value}, "
                f"expected={expected_value}"
            )

    if errors:
        print(f"[FAIL] {label} reproduction mismatch:")
        for e in errors:
            print(" ", e)
        return False

    print(f"[PASS] {label} reproduces original validation")
    return True

def main():
    require(torch.cuda.is_available(), "CUDA unavailable")
    require(RUN.is_dir(), "E4 output directory missing")
    require(CONFIG.is_file(), "E4 config missing")
    require((SNAPSHOT / "config.json").is_file(), "Snapshot config missing")
    require((SNAPSHOT / "model.safetensors").is_file(), "Snapshot weights missing")

    require(not OUT.exists(), f"Output already exists: {OUT}")

    identity = load_json(RUN / "run_identity.json")

    for key, expected in {
        "protocol": "SVR-Depth-CrossSource-v2",
        "method": "P_multi",
        "seed": 20261103,
        "initialization_seed": 20261103,
        "steps": 1000,
        "window_frames": 20,
        "tbptt_frames": 4
    }.items():
        require(
            identity.get(key) == expected,
            f"E4 identity mismatch: {key}"
        )

    best = torch.load(
        RUN / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=False
    )
    last = torch.load(
        RUN / "last_checkpoint.pt",
        map_location="cpu",
        weights_only=False
    )

    for ckpt in (best, last):
        for key in (
            "protocol", "method", "seed",
            "initialization_seed", "steps",
            "window_frames", "tbptt_frames",
            "dataset_weights"
        ):
            require(
                ckpt["identity"].get(key) == identity.get(key),
                f"Checkpoint identity mismatch: {key}"
            )

    require(best["step"] == 700, "Unexpected best checkpoint step")
    require(last["step"] == 1000, "Unexpected last checkpoint step")

    with open(RUN / "training_curve.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    last_row = next(
        r for r in rows if int(r["step"]) == 1000
    )

    expected_best = best["val"]
    expected_last = {
        "macro_loss": float(last_row["val_macro_loss"]),
        "per_dataset": json.loads(last_row["val_per_dataset"])
    }

    device = torch.device("cuda")

    pool = tr.CrossSourcePool.from_config(
        CONFIG, split="val", verify_files=True
    )

    refs = tr.build_val_refs(
        pool,
        window_frames=20,
        max_per_dataset=8,
        seed=20261104
    )

    OUT.mkdir(parents=True, exist_ok=False)

    save_json(
        OUT / "validation_refs.json",
        [tr.ref_to_dict(r) for r in refs]
    )

    print("========== E4 BASELINE EVALUATION ==========")
    print("Validation windows:", len(refs))
    print("Output:", OUT)

    results = {}

    # B: E4 best
    print("\n========== B: E4 BEST STEP 700 ==========")
    model = make_model(
        "pre_readout", device, best["head_state"]
    )
    results["E4_best_700"] = evaluate(
        model, pool, refs, device
    )
    del model
    torch.cuda.empty_cache()

    # C: E4 last
    print("\n========== C: E4 LAST STEP 1000 ==========")
    model = make_model(
        "pre_readout", device, last["head_state"]
    )
    results["E4_last_1000"] = evaluate(
        model, pool, refs, device
    )
    del model
    torch.cuda.empty_cache()

    # Critical safeguard: exact E4 validation reproduction
    ok_best = check_reproduction(
        "E4 Best", results["E4_best_700"], expected_best
    )
    ok_last = check_reproduction(
        "E4 Last", results["E4_last_1000"], expected_last
    )

    save_json(OUT / "reproduction_check.json", {
        "best_pass": ok_best,
        "last_pass": ok_last,
        "expected_best": expected_best,
        "actual_best": results["E4_best_700"],
        "expected_last": expected_last,
        "actual_last": results["E4_last_1000"]
    })

    require(
        ok_best and ok_last,
        "E4 validation reproduction failed. Stop before baseline comparison."
    )

    # A0: Same P_multi architecture, original initialization
    print("\n========== A0: UNTRAINED P_MULTI ==========")
    model = make_model(
        "pre_readout",
        device,
        seed=identity["initialization_seed"]
    )
    results["P_multi_initial"] = evaluate(
        model, pool, refs, device
    )
    del model
    torch.cuda.empty_cache()

    # A: Original VeloDepth, repair head disabled
    print("\n========== A: ORIGINAL VELODEPTH ==========")
    model = make_model("off", device)
    results["VeloDepth_original"] = evaluate(
        model, pool, refs, device
    )
    del model
    torch.cuda.empty_cache()

    save_json(OUT / "evaluation_results.json", {
        "protocol": "E4 same-validation-window comparison",
        "validation_windows": len(refs),
        "results": results
    })

    baseline = results["VeloDepth_original"]["macro_loss"]

    print("\n========== FINAL COMPARISON ==========")

    for name, result in results.items():
        gain = (baseline - result["macro_loss"]) / baseline * 100

        print(
            f"{name:24s} "
            f"macro={result['macro_loss']:.8f} "
            f"gain_vs_original={gain:+.3f}%"
        )

        for ds, value in result["per_dataset"].items():
            print(f"    {ds}: {value:.8f}")

    csv_path = OUT / "evaluation_summary.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "dataset", "validation_loss",
            "macro_loss", "supervised_frames", "windows"
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

    print("\n[EVAL PASS]")
    print("Results:", OUT)

if __name__ == "__main__":
    main()
