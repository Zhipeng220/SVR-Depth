import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

root = Path.cwd().resolve()
manifest_root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()

names = (
    "rgbd_bonn_balloon_tracking",
    "rgbd_bonn_kidnapping_box",
    "rgbd_bonn_placing_obstructing_box",
    "rgbd_bonn_removing_obstructing_box",
    "rgbd_bonn_synchronous2",
)

dev_sequences = (
    "moving_nonobstructing_box",
    "moving_obstructing_box",
    "placing_nonobstructing_box",
    "removing_nonobstructing_box",
    "person_tracking2",
)

expected_y11_protocol_sha = (
    "11f30c44b7b38ccf05174225b870eac04b9ced1fc36241a5dd386eab5c79d0e4"
)

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

errors = []
references = []

# 2. 旧训练/开发录像的参考清单
for seq in ("crowd2", "crowd3", "balloon2"):
    path = (
        root / "outputs/gt4a2_exp1"
        / f"gt4a2_exp1_{seq}_pairs_used.csv"
    )
    references.append((seq, path))

# 3. Y11 开发录像的参考清单
y11_protocol = root / "outputs/y11_confirmation_data/dataset_protocol.json"

if not y11_protocol.is_file():
    errors.append(f"缺少 Y11 数据协议：{y11_protocol}")
else:
    actual_sha = sha256_file(y11_protocol)

    if actual_sha != expected_y11_protocol_sha:
        errors.append(
            f"Y11 数据协议 SHA256 不匹配：{actual_sha}"
        )
    else:
        protocol = json.loads(y11_protocol.read_text(encoding="utf-8-sig"))

        for seq in dev_sequences:
            raw = protocol.get("manifests", {}).get(seq, {}).get("path", "")
            if not raw:
                errors.append(f"Y11 协议没有记录 {seq} 的清单路径")
                continue

            path = Path(raw)
            if not path.is_absolute():
                path = root / path

            references.append((seq, path.resolve()))

print("========== 参考 RGB 哈希 ==========", flush=True)

reference_hashes = {}
reference_counts = {}
hashed_paths = {}

for seq, manifest in references:
    if not manifest.is_file():
        errors.append(f"参考清单不存在：{manifest}")
        continue

    try:
        rows = read_rows(manifest)

        if not rows or "rgb_path" not in rows[0]:
            raise ValueError("清单为空或缺少 rgb_path")

        count = 0

        for row in rows:
            path = Path(row["rgb_path"])

            if not path.is_absolute() or not path.is_file():
                raise ValueError(f"参考 RGB 文件不存在或路径不绝对：{path}")

            key = str(path.resolve())

            if key not in hashed_paths:
                hashed_paths[key] = sha256_file(path)

            digest = hashed_paths[key]
            reference_hashes.setdefault(digest, set()).add(seq)
            count += 1

            if count % 500 == 0:
                print(f"[REFERENCE] {seq}: {count}/{len(rows)}", flush=True)

        reference_counts[seq] = count
        print(f"[REFERENCE DONE] {seq}: {count}", flush=True)

    except Exception as exc:
        errors.append(f"{seq}: {type(exc).__name__}: {exc}")
        print(f"[REFERENCE ERROR] {seq}: {exc}", flush=True)

print("\n========== 候选 RGB 哈希 ==========", flush=True)

candidate_results = []
overlaps = []

for name in names:
    manifest = manifest_root / name / "pairs.csv"
    audit_path = manifest_root / name / "audit.json"

    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))

        if sha256_file(manifest) != audit.get("manifest_sha256"):
            raise ValueError("候选清单 SHA256 与审计记录不一致")

        rows = read_rows(manifest)

        if [int(r["frame_index_zero_based"]) for r in rows] != list(range(len(rows))):
            raise ValueError("候选 RGB 索引不连续")

        overlap_count = 0

        for index, row in enumerate(rows):
            path = Path(row["rgb_path"])

            if not path.is_absolute() or not path.is_file():
                raise ValueError(f"候选 RGB 文件不存在或路径不绝对：{path}")

            digest = sha256_file(path)
            refs = reference_hashes.get(digest, set())

            if refs:
                overlap_count += 1
                overlaps.append({
                    "candidate_sequence": name,
                    "frame_index_zero_based": index,
                    "rgb_path": str(path),
                    "matching_reference_sequences": ",".join(sorted(refs)),
                    "rgb_sha256": digest,
                })

            if (index + 1) % 500 == 0:
                print(f"[CANDIDATE] {name}: {index + 1}/{len(rows)}", flush=True)

        candidate_results.append({
            "sequence": name,
            "rgb_count": len(rows),
            "byte_identical_to_known_reference": overlap_count,
        })

        print(
            f"[CANDIDATE DONE] {name}: "
            f"RGB={len(rows)}，与已检查参考字节相同={overlap_count}",
            flush=True,
        )

    except Exception as exc:
        errors.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"[CANDIDATE ERROR] {name}: {exc}", flush=True)

expected_references = set(("crowd2", "crowd3", "balloon2") + dev_sequences)
references_complete = expected_references == set(reference_counts)
candidates_complete = len(candidate_results) == len(names)

if errors or not references_complete or not candidates_complete:
    status = "INCOMPLETE"
elif overlaps:
    status = "OVERLAP_FOUND"
else:
    status = "NO_BYTE_OVERLAP_WITH_CHECKED_REFERENCES"

summary = {
    "status": status,
    "reference_counts": reference_counts,
    "references_complete": references_complete,
    "candidate_results": candidate_results,
    "overlap_count": len(overlaps),
    "errors": errors,
    "scope": (
        "Exact RGB byte hashes against the listed reference manifests only. "
        "Does not establish access history, physical-scene independence, "
        "pretraining overlap, semantic duplicates, or GT correctness."
    ),
    "model_inference_performed": False,
    "formal_test_frozen": False,
}

summary_path = out / "summary.json"
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

overlap_path = out / "overlaps.csv"
with overlap_path.open("w", encoding="utf-8", newline="") as f:
    fields = [
        "candidate_sequence",
        "frame_index_zero_based",
        "rgb_path",
        "matching_reference_sequences",
        "rgb_sha256",
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(overlaps)

archive = out / "rgb_overlap_audit.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(summary_path, "summary.json")
    z.write(overlap_path, "overlaps.csv")

print("\n========== 审计结束 ==========", flush=True)
print("状态:", status, flush=True)
print("完全相同的候选 RGB 帧:", len(overlaps), flush=True)
print("参考清单齐全:", references_complete, flush=True)
print("审计包:", archive, flush=True)

for error in errors:
    print("[ERROR]", error, flush=True)

if status != "NO_BYTE_OVERLAP_WITH_CHECKED_REFERENCES":
    raise SystemExit(1)
