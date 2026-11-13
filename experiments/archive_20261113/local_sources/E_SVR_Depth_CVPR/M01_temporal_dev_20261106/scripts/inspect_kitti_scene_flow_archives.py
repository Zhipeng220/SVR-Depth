#!/usr/bin/env python3
"""Inspect KITTI Scene Flow archive members without extracting or modifying them."""
from __future__ import annotations
import argparse, json, zipfile
from pathlib import Path

EXPECTED = {
    "main": ("training/image_2/", "training/disp_occ_0/", "training/disp_occ_1/"),
    "multiview": ("training/image_2/",),
    "calib": ("training/calib_cam_to_cam/",),
    "devkit": (),
}


def inspect(label: str, path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        files = [n for n in names if not n.endswith("/")]
        prefixes = EXPECTED[label]
        normalized = {n: "/" + n.lstrip("./") for n in files}
        found = {prefix: sum(("/" + prefix) in path for path in normalized.values()) for prefix in prefixes}
        examples = {prefix: [n for n, path in normalized.items() if ("/" + prefix) in path][:3] for prefix in prefixes}
        return {"archive": str(path), "file_count": len(files), "expected_prefix_counts": found,
                "examples": examples, "all_prefixes_present": all(found.values()) if found else True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("main", "multiview", "calib", "devkit"):
        p.add_argument(f"--{name}-zip", required=(name != "devkit"))
    p.add_argument("--out", default=None)
    a = p.parse_args()
    result = {}
    for name in ("main", "multiview", "calib", "devkit"):
        value = getattr(a, f"{name}_zip")
        if value:
            result[name] = inspect(name, Path(value).expanduser().resolve())
    text = json.dumps(result, indent=2)
    print(text)
    if a.out:
        Path(a.out).expanduser().resolve().write_text(text, encoding="utf-8")

if __name__ == "__main__": main()
