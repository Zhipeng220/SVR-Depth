import os
import sys
import json
import csv
import hashlib
from pathlib import Path
from collections import Counter

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_crosssource_v2_E4_baseline as base

EXPECTED_SNAPSHOT = (
    "615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e"
)

RUNS = {
    "E4-C": ROOT / (
        "outputs/crosssource_v2_E4_20261104_C_s20261103_1000step"
    ),
    "P_multi_TA06": ROOT / (
        "outputs/crosssource_v2_E8B_C_lossTA06_1000step_resume500_20261104_225546"
    ),
}

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if hasattr(x, "item"):
        return x.item()
    return x

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. 请切回 GPU 实例。")

    device = torch.device("cuda:0")
    torch.manual_seed(20261104)
    torch.cuda.manual_seed_all(20261104)

    snapshot_hash = sha(base.SNAPSHOT / "model.safetensors")
    assert snapshot_hash == EXPECTED_SNAPSHOT

    print("========== CHECKPOINT IDENTITY ==========")

    checkpoints = {}
    identities = {}

    for name, run in RUNS.items():
        assert run.is_dir(), f"Missing run: {run}"

        identity = json.loads(
            (run / "run_identity.json").read_text()
        )

        assert identity["protocol"] == "SVR-Depth-CrossSource-v2"
        assert identity["method"] == "P_multi"
        assert identity["seed"] == 20261103
        assert identity["steps"] == 1000
        assert identity["window_frames"] == 20
        assert identity["tbptt_frames"] == 4
        assert identity["snapshot_sha256"] == snapshot_hash

        identities[name] = identity

        for variant in ("best", "last"):
            path = run / f"{variant}_checkpoint.pt"
            assert path.is_file(), path

            ck = torch.load(
                path, map_location="cpu", weights_only=False
            )

            assert ck["identity"] == identity
            assert isinstance(ck["head_state"], dict)

            params = sum(
                v.numel() for v in ck["head_state"].values()
            )
            assert params == 170496, (name, variant, params)

            label = f"{name}_{variant}"

            checkpoints[label] = ck

            print(
                f"[PASS] {label}: step={ck['step']} "
                f"head={params} "
                f"best_val={ck['best_val_loss']:.8f}"
            )

    print("\n========== MANIFEST PROVENANCE ==========")

    for key in ("E4-C", "P_multi_TA06"):
        m = identities[key].get("manifest_sha256", {})
        val = {k: v for k, v in m.items() if "/val/" in k}
        print(key, "validation manifests:", len(val))
        print("Validation hashes:", val)

    print("\n========== BUILD COMMON VALIDATION ==========")

    config = ROOT / (
        "configs/crosssource_v2/generated_20261103/"
        "C_tartanair_kitti.json"
    )

    pool = base.tr.CrossSourcePool.from_config(
        config,
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

    print("Validation windows:", dict(counts))
    print("Total:", len(refs))

    assert counts["kitti_sceneflow"] == 8
    assert counts["tartanair"] == 5
    assert len(refs) == 13

    ref_dicts = [
        base.tr.ref_to_dict(r) for r in refs
    ]

    ref_hash = hashlib.sha256(
        json.dumps(ref_dicts, sort_keys=True).encode()
    ).hexdigest()

    # 检查是否与此前 E6 的 13 个窗口完全一致
    e6_refs_path = ROOT / (
        "outputs/crosssource_v2_E6_common_eval_20261104_v1/"
        "validation_refs.json"
    )

    if e6_refs_path.is_file():
        old_refs = json.loads(e6_refs_path.read_text())
        assert old_refs == ref_dicts, (
            "E6 validation refs differ; stop comparison."
        )
        print("[PASS] Exact E6 validation reference match")

    out = ROOT / os.environ["OUT"]
    assert not out.exists()
    out.mkdir(parents=True)

    (out / "validation_refs.json").write_text(
        json.dumps(ref_dicts, indent=2),
        encoding="utf-8"
    )

    print("\n========== COMMON EVALUATION ==========")

    results = {}

    for label, ck in checkpoints.items():
        print(f"\nEvaluating: {label}")

        model = base.make_model(
            "pre_readout",
            device,
            ck["head_state"]
        )

        result = base.evaluate(
            model, pool, refs, device
        )

        assert result["windows"] == dict(counts)

        results[label] = clean(result)

        print("Macro:", result["macro_loss"])
        print("Per dataset:", result["per_dataset"])

        del model
        torch.cuda.empty_cache()

    print("\n========== FINAL COMPARISON ==========")

    for variant in ("best", "last"):
        a = results[f"E4-C_{variant}"]
        b = results[f"P_multi_TA06_{variant}"]

        print(f"\n{variant.upper()}")

        print(
            "E4 macro:",
            f"{a['macro_loss']:.8f}"
        )
        print(
            "TA06 macro:",
            f"{b['macro_loss']:.8f}"
        )
        print(
            "TA06 - E4:",
            f"{b['macro_loss'] - a['macro_loss']:+.8f}"
        )

        for ds in ("kitti_sceneflow", "tartanair"):
            delta = (
                b["per_dataset"][ds]
                - a["per_dataset"][ds]
            )
            print(ds, f"{delta:+.8f}")

    payload = {
        "protocol": "Common 13-window validation",
        "validation_windows": dict(counts),
        "validation_refs_sha256": ref_hash,
        "snapshot_sha256": snapshot_hash,
        "identities": identities,
        "results": results
    }

    (out / "evaluation_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n[PASS] COMMON VALIDATION COMPLETE")
    print("OUTPUT:", out)

if __name__ == "__main__":
    main()
