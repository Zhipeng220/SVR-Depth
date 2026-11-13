from pathlib import Path
import csv
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/gt4a1_diagnostics"
ARCHIVE = ROOT / "outputs/GT4_A1_review.zip"
GT4 = ROOT / "outputs/gt4_validation"
GT2 = ROOT / "outputs/gt2_validation"

OUT.mkdir(parents=True, exist_ok=True)

included = {}
skipped = []

def add(path, archive_name=None, max_bytes=None):
    path = Path(path)
    if not path.is_file():
        return False

    size = path.stat().st_size
    if max_bytes is not None and size > max_bytes:
        skipped.append({
            "path": str(path),
            "reason": f"size exceeds {max_bytes} bytes",
            "size": size,
        })
        return False

    name = archive_name or str(path.relative_to(ROOT))
    included[name] = path
    return True

# 1. GT4-A1 汇总表、运行日志和总报告。
essential = [
    "gt4a1_manifest.json",
    "gt4a1_selected_clips_all.csv",
    "gt4a1_frame_metrics.csv",
    "gt4a1_depth_bin_metrics.csv",
    "gt4a1_spatial_grid_metrics.csv",
    "gt4a1_depth_bin_summary.csv",
    "gt4a1_spatial_grid_summary.csv",
    "gt4a1_run.log",
]

for name in essential:
    add(OUT / name)

required = [
    "gt4a1_manifest.json",
    "gt4a1_selected_clips_all.csv",
    "gt4a1_frame_metrics.csv",
    "gt4a1_depth_bin_metrics.csv",
    "gt4a1_spatial_grid_metrics.csv",
]

missing = [name for name in required if not (OUT / name).is_file()]

# 2. 每条序列的报告、汇总和片段选择清单。
for seq in ("crowd3", "balloon2"):
    seq_out = OUT / seq

    for name in (
        "GT4_A1_report.md",
        "gt4a1_summary.json",
        "gt4a1_selected_clips.csv",
    ):
        add(seq_out / name)

    # 只收录诊断拼图和对应元数据，不收录全部原始图像。
    for montage in sorted(seq_out.glob("frames/**/diagnostic_montage.png")):
        add(montage)

        metadata = montage.parent / "frame_metadata.json"
        add(metadata)

        delta_info = montage.parent / "error_delta_visualization.json"
        add(delta_info)

# 3. GT4 协议核验：仅收录配置、汇总、指标和配对清单。
# 不递归打包 GT4 的其他预测图、特征或中间张量。
for seq in ("crowd3", "balloon2"):
    seq_dir = GT4 / seq

    for name in (
        "gt4_config.json",
        "gt4_summary.json",
        "gt4_frame_metrics.csv",
    ):
        add(
            seq_dir / name,
            f"provenance/gt4/{seq}/{name}",
        )

    cfg_path = seq_dir / "gt4_config.json"
    pair_path = None

    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            raw = cfg.get("pair_csv")
            if raw:
                candidate = Path(raw)
                candidates = (
                    [candidate] if candidate.is_absolute()
                    else [ROOT / candidate, seq_dir / candidate]
                )
                pair_path = next(
                    (p for p in candidates if p.is_file()), None
                )
        except Exception as e:
            print(f"[WARN] 无法解析 {cfg_path}: {e}")

    if pair_path is None:
        for name in ("gt4_pairs.csv", "gt4_pair_csv_used.csv"):
            candidate = seq_dir / name
            if candidate.is_file():
                pair_path = candidate
                break

    if pair_path is not None:
        add(
            pair_path,
            f"provenance/gt4/{seq}/evaluation_pairs.csv",
            max_bytes=20 * 1024 * 1024,
        )

# 4. GT2 训练摘要及训练配对清单，用于核验训练数据来源。
gt2_summary_path = GT2 / "gt2_summary.json"
add(gt2_summary_path, "provenance/gt2/gt2_summary.json")

if gt2_summary_path.is_file():
    try:
        summary = json.loads(
            gt2_summary_path.read_text(encoding="utf-8")
        )
        raw = summary.get("pair_csv")
        if raw:
            candidate = Path(raw)
            candidates = (
                [candidate] if candidate.is_absolute()
                else [ROOT / candidate, GT2 / candidate]
            )
            pair_path = next(
                (p for p in candidates if p.is_file()), None
            )
            if pair_path is not None:
                add(
                    pair_path,
                    "provenance/gt2/training_pairs.csv",
                    max_bytes=20 * 1024 * 1024,
                )
    except Exception as e:
        print(f"[WARN] GT2 摘要解析失败：{e}")

# 5. 复现实验所需的轻量源码和操作说明。
for name in (
    "experiments/p1/gt4a1_failure_diagnosis.py",
    "experiments/p1/run_gt4a1.sh",
    "experiments/p1/gt4a1_package_review.py",
    "GT4_A1_README.md",
):
    add(ROOT / name)

# 6. 记录实际纳入文件的大小和 SHA-256。
manifest = {
    "experiment": "GT4-A1 compact review bundle",
    "complete": not missing,
    "missing_required_files": missing,
    "included_file_count": len(included),
    "excluded_content": [
        "所有原始逐像素 NPY 数组",
        "所有独立的全分辨率预测图和误差图",
        "GT4 目录中与本次诊断无关的文件",
        "模型权重文件",
        "缓存和临时文件",
    ],
    "files": [],
    "skipped": skipped,
}

for name, path in sorted(included.items()):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    manifest["files"].append({
        "archive_path": name,
        "size_bytes": path.stat().st_size,
        "sha256": h.hexdigest(),
    })

manifest_path = OUT / "gt4a1_package_manifest.json"
manifest_path.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
add(manifest_path)

# 7. 只按明确白名单打包。
with zipfile.ZipFile(
    ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
) as z:
    for name, path in sorted(included.items()):
        z.write(path, name)

with zipfile.ZipFile(ARCHIVE) as z:
    bad = z.testzip()
    names = z.namelist()

print("\n========== GT4-A1 精简打包 ==========")
print("必需文件状态：", "COMPLETE" if not missing else "INCOMPLETE")
if missing:
    print("缺失必需文件：", ", ".join(missing))

print("纳入文件数：", len(names))
print("ZIP 完整性：", "PASS" if bad is None else f"FAIL: {bad}")
print("压缩包：", ARCHIVE)
print("压缩包大小：", f"{ARCHIVE.stat().st_size / 1024 / 1024:.2f} MB")

print("\n文件分类统计：")
for prefix in ("outputs/", "provenance/", "experiments/", "GT4_A1_README.md"):
    count = sum(name.startswith(prefix) for name in names)
    if count:
        print(f"  {prefix}: {count}")

if bad is not None:
    raise SystemExit(1)
