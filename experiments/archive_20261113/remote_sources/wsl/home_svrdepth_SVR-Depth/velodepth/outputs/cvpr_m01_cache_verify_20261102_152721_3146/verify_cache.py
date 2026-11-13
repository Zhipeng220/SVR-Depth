import csv
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

root = Path.cwd().resolve()
out = Path(sys.argv[1]).resolve()
outputs = root / "outputs"

runs = {
    "20261025": {
        "P": outputs / "gt4a2_y04_validation_fix3_20261025_225902",
        "O_multi": outputs / "gt4a2_y04_validation_fix3_20261025_225902",
    },
    "20261101": {
        "P": outputs / "cvpr_m05_pmulti_seed_20261101",
        "O_multi": outputs / "cvpr_omulti_seed_20261101_20261101_222630_11024",
    },
    "20261102": {
        "P": outputs / "cvpr_m05_pmulti_seed_20261102",
        "O_multi": outputs / "cvpr_omulti_seed_20261102_20261102_114447_1479",
    },
}

sequences = ("crowd2", "crowd3", "balloon2")
hash_cache = {}
rows = []
errors = []

def file_sha256(path):
    with path.open("rb") as f:
        before = os.fstat(f.fileno())
        key = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )

        if key in hash_cache:
            return hash_cache[key], before.st_size

        h = hashlib.sha256()
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)

        after = os.fstat(f.fileno())
        after_key = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )

        if key != after_key:
            raise RuntimeError(f"哈希期间文件发生变化：{path}")

        digest = h.hexdigest()
        hash_cache[key] = digest
        return digest, before.st_size

def load_manifest(run, seq):
    directory = run / "training/scene_caches" / seq / "cycle_cache"
    path = directory / "cache_manifest.json"

    with path.open("r", encoding="utf-8-sig") as f:
        manifest = json.load(f)

    return directory, path, manifest

for seed, methods in runs.items():
    print(f"\n========== SEED {seed} ==========", flush=True)

    for seq in sequences:
        try:
            p_dir, p_manifest_path, p_manifest = load_manifest(methods["P"], seq)
            o_dir, o_manifest_path, o_manifest = load_manifest(methods["O_multi"], seq)

            metadata_equal = all(
                p_manifest.get(key) == o_manifest.get(key)
                for key in (
                    "format",
                    "pair_sha256",
                    "pretrained_weights_sha256",
                    "original_source_hashes",
                )
            )

            shared = p_dir.resolve() == o_dir.resolve()
            print(
                f"[SEQUENCE] {seq}; 共用同一路径={shared}; "
                f"清单元数据一致={metadata_equal}",
                flush=True,
            )

            for cycle in range(8):
                entry_p = p_manifest.get("cycles", {}).get(str(cycle))
                entry_o = o_manifest.get("cycles", {}).get(str(cycle))

                if not entry_p or not entry_o:
                    raise ValueError(f"{seq} cycle={cycle} 缺少清单记录")

                p_file = p_dir / f"cycle_{cycle:02d}.pt"
                o_file = o_dir / f"cycle_{cycle:02d}.pt"

                p_sha, p_size = file_sha256(p_file)
                o_sha, o_size = file_sha256(o_file)

                p_matches_manifest = (
                    p_sha == entry_p.get("sha256")
                    and p_size == entry_p.get("bytes")
                )
                o_matches_manifest = (
                    o_sha == entry_o.get("sha256")
                    and o_size == entry_o.get("bytes")
                )
                files_equal = p_sha == o_sha
                supervision_count_equal = (
                    entry_p.get("supervised") == entry_o.get("supervised")
                )

                passed = (
                    metadata_equal
                    and p_matches_manifest
                    and o_matches_manifest
                    and files_equal
                    and supervision_count_equal
                )

                rows.append({
                    "seed": seed,
                    "sequence": seq,
                    "cycle": cycle,
                    "same_cache_path": shared,
                    "metadata_equal": metadata_equal,
                    "p_file_matches_manifest": p_matches_manifest,
                    "o_file_matches_manifest": o_matches_manifest,
                    "p_o_file_bytes_equal": files_equal,
                    "supervision_count_equal": supervision_count_equal,
                    "p_sha256": p_sha,
                    "o_sha256": o_sha,
                    "p_file": str(p_file),
                    "o_file": str(o_file),
                    "passed": passed,
                })

                print(
                    f"  cycle={cycle}: "
                    f"{'PASS' if passed else 'MISMATCH'}",
                    flush=True,
                )

                if not passed:
                    errors.append(f"{seed}/{seq}/cycle_{cycle:02d}: 不一致")

        except Exception as exc:
            message = f"{seed}/{seq}: {type(exc).__name__}: {exc}"
            errors.append(message)
            print("[ERROR]", message, flush=True)

expected_rows = len(runs) * len(sequences) * 8
all_passed = (
    len(rows) == expected_rows
    and all(row["passed"] for row in rows)
    and not errors
)

summary = {
    "status": "CACHE_FILES_VERIFIED" if all_passed else "INCOMPLETE_OR_MISMATCH",
    "expected_cycle_pairs": expected_rows,
    "checked_cycle_pairs": len(rows),
    "passed_cycle_pairs": sum(row["passed"] for row in rows),
    "unique_files_hashed": len(hash_cache),
    "errors": errors,
    "historical_20261025": (
        "P and O_multi refer to the same historical cache path; "
        "this is a shared-cache check, not two independent copies."
    ),
    "scope": (
        "Verifies current cache bytes against recorded manifest hashes. "
        "It does not independently prove the original raw GT file bytes, "
        "access history, or final-test independence."
    ),
    "test_images_read": False,
    "model_inference_performed": False,
    "formal_test_frozen": False,
}

summary_path = out / "summary.json"
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

csv_path = out / "cache_pairs.csv"
fields = [
    "seed", "sequence", "cycle", "same_cache_path", "metadata_equal",
    "p_file_matches_manifest", "o_file_matches_manifest",
    "p_o_file_bytes_equal", "supervision_count_equal",
    "p_sha256", "o_sha256", "p_file", "o_file", "passed",
]

with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

archive = out / "m01_cache_verification.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(summary_path, "summary.json")
    z.write(csv_path, "cache_pairs.csv")

print("\n========== 缓存验证结束 ==========", flush=True)
print("状态:", summary["status"], flush=True)
print(
    f"通过周期: {summary['passed_cycle_pairs']}/"
    f"{summary['expected_cycle_pairs']}",
    flush=True,
)
print("审计包:", archive, flush=True)

for error in errors:
    print("[ERROR]", error, flush=True)

if not all_passed:
    raise SystemExit(1)
