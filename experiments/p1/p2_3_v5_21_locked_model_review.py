#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2.3 v5.21 locked-model cross-sequence review.

This is a readable two-file launcher, not a Base64-packed copy of the experiment.
It reuses the already-installed v5.20 collector/evaluator source and creates
plain-text v5.21 runtime copies under experiments/p1/P2_3_v5_21_runtime/.
The v5.20 package and locked 53-anchor CSV must already exist on AutoDL.

Default workflow:
  python experiments/p1/p2_3_v5_21_locked_model_review.py all

Modes:
  self-test  Check dependencies and run synthetic/unit self-tests only.
  collect    Collect synchronous labels with frozen VeloDepth weights (GPU).
  evaluate   Evaluate existing labels (CPU only).
  all        Run self-tests, collect if labels are absent, then evaluate.

No VeloDepth weights are trained or modified.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

VERSION_FROM = "v5.20"
VERSION_TO = "v5.21"
DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/datasets/rgbd_bonn_synchronous")
DEFAULT_SEQUENCE = "synchronous"
DEFAULT_FRAMES = 86


def resolve_repo_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / "experiments" / "p1").is_dir():
            raise FileNotFoundError(f"Not a VeloDepth project root: {root}")
        return root

    env_root = os.environ.get("SVRDEPTH_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if (root / "experiments" / "p1").is_dir():
            return root

    script = Path(__file__).resolve()
    for candidate in script.parents:
        if (
            (candidate / "experiments" / "p1" / "P2_3_v5_20_package").is_dir()
            and (candidate / "velodepth").is_dir()
        ):
            return candidate

    # Normal deployment location: <repo>/experiments/p1/<this script>
    if len(script.parents) >= 3:
        candidate = script.parents[2]
        if (candidate / "experiments" / "p1").is_dir():
            return candidate

    raise FileNotFoundError(
        "Cannot locate project root. Run this from "
        "/root/autodl-tmp/SVR-Depth/velodepth or set SVRDEPTH_ROOT."
    )


def paths_for(root: Path) -> dict[str, Path]:
    p1 = root / "experiments" / "p1"
    package = p1 / "P2_3_v5_20_package"
    runtime = p1 / "P2_3_v5_21_runtime"
    output = root / "outputs" / "p2_3_v5_21_external" / DEFAULT_SEQUENCE
    return {
        "root": root,
        "p1": p1,
        "package": package,
        "runtime": runtime,
        "train": package / "inputs" / "p2_3_v5_18_aligned_53_anchor_dataset.csv",
        "source_collector": package / "p2_3_v5_20_external_sequence_collector.py",
        "source_evaluator": package / "analyze_p2_3_v5_20_external_validation.py",
        "runtime_collector": runtime / "p2_3_v5_21_external_sequence_collector.py",
        "runtime_evaluator": runtime / "analyze_p2_3_v5_21_external_validation.py",
        "data": DEFAULT_DATA_ROOT,
        "output": output,
        "labels": output / "p2_3_v5_15_same_state_anchor_labels.csv",
        "evaluation": output / "evaluation",
        "review_zip": p1 / "P2_3_v5_21_external_validation_review.zip",
    }


def require_file(path: Path, description: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Missing {description}: {path}\n"
            "This two-file workflow reuses the existing v5.20 package; "
            "do not upload another full package unless this file is genuinely absent."
        )


def make_runtime_copies(p: dict[str, Path]) -> None:
    require_file(p["source_collector"], "existing v5.20 collector")
    require_file(p["source_evaluator"], "existing v5.20 evaluator")
    p["runtime"].mkdir(parents=True, exist_ok=True)

    collector_text = p["source_collector"].read_text(encoding="utf-8")
    collector_text = collector_text.replace("v5.20", "v5.21")
    collector_text = collector_text.replace("v5_20", "v5_21")
    p["runtime_collector"].write_text(collector_text, encoding="utf-8")

    evaluator_text = p["source_evaluator"].read_text(encoding="utf-8")
    evaluator_text = evaluator_text.replace("P2_3_v5_20", "P2_3_v5_21")
    evaluator_text = evaluator_text.replace("p2_3_v5_20", "p2_3_v5_21")
    evaluator_text = evaluator_text.replace("v5.20", "v5.21")
    evaluator_text = evaluator_text.replace("v5_20", "v5_21")
    evaluator_text = evaluator_text.replace(
        "independent-sequence validation",
        "locked-model cross-sequence replication",
    )
    evaluator_text = evaluator_text.replace(
        "genuinely unseen sequence label CSVs",
        "held-out sequence label CSVs",
    )
    evaluator_text = evaluator_text.replace(
        '"sequence_name_independent_check": "PASS",',
        '"sequence_name_independent_check": '
        '"PASS_NOT_IN_LOCKED_TRAINING_SET; HISTORICAL_EXPOSURE_RECORDED",',
    )
    evaluator_text = evaluator_text.replace(
        '"primary_model": "state_age_ols vs age_resid_state2_mean",',
        '"primary_model": "state_age_ols vs age_resid_state2_mean",\n'
        '            "historical_sequence_exposure": '
        '"synchronous was used in earlier P2.3 v5.10/v5.11 experiments; '
        'this is a locked-model replication, not a pristine project-wide unseen-sequence test.",',
    )
    evaluator_text = evaluator_text.replace(
        '"## 1. 实验定位", "",',
        '"## 1. 实验定位", "", '
        '"项目历史记录显示 synchronous 已用于 P2.3 v5.10/v5.11，因此本轮是锁定模型跨序列复核，不是全项目范围内的全新独立测试。", "",',
    )
    evaluator_text = evaluator_text.replace(
        "Only truly new sequence names/content count as external validation; renaming known footage is leakage.",
        "The synchronous sequence was used in earlier P2.3 v5.10/v5.11 experiments; this is a locked-model replication, not a pristine project-wide unseen-sequence test.",
    )
    evaluator_text = evaluator_text.replace(
        "Only truly new sequence names/content count as external validation; renaming known footage is leakage.",
        "The synchronous sequence was used in earlier P2.3 v5.10/v5.11 experiments; this is a locked-model replication, not a pristine project-wide unseen-sequence test.",
    )
    evaluator_text = evaluator_text.replace(
        "One new sequence is a pilot, not broad generalization evidence; prefer two or more unseen sequences.",
        "Only one held-out sequence is evaluated here; this is not broad generalization evidence.",
    )
    evaluator_text = evaluator_text.replace(
        "- 只把真正未参与特征发现的新序列作为外部测试；仅更改序列名不算独立验证。",
        "- synchronous 曾用于 v5.10/v5.11；本轮按锁定模型复核解释，不视为全项目全新独立测试。",
    )
    evaluator_text = evaluator_text.replace(
        "- 一条新序列只能作为 pilot；推荐至少两条未见序列。",
        "- 本轮只有一条复核序列，不能支撑广泛泛化结论。",
    )
    p["runtime_evaluator"].write_text(evaluator_text, encoding="utf-8")

    # Compile in memory to avoid generating bytecode or modifying project files.
    for source in (p["runtime_collector"], p["runtime_evaluator"]):
        compile(source.read_text(encoding="utf-8"), str(source), "exec")


def run_command(command: list[str], cwd: Path) -> None:
    print("+", " ".join(str(x) for x in command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def check_dependencies(p: dict[str, Path]) -> None:
    require_file(p["train"], "locked 53-anchor training CSV")
    require_file(p["source_collector"], "v5.20 collector")
    require_file(p["source_evaluator"], "v5.20 evaluator")

    helper_names = [
        "p2_3_online_replay_v5_4_budget_sweep.py",
        "p1_multihorizon_counterfactual.py",
        "stream_state.py",
        "stream_state_snapshot.py",
        "velodepth_stream.py",
    ]
    missing = [name for name in helper_names if not (p["p1"] / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing existing project helper(s): " + ", ".join(missing)
        )


def run_self_test(p: dict[str, Path]) -> None:
    check_dependencies(p)
    make_runtime_copies(p)
    run_command(
        [sys.executable, str(p["runtime_collector"]), "--self-test"],
        p["root"],
    )
    run_command(
        [sys.executable, str(p["runtime_evaluator"]), "--self-test"],
        p["root"],
    )
    print("PASS: v5.21 wrapper checks and synthetic self-tests. No research experiment was run.")


def validate_labels(label_path: Path, expected_sequence: str) -> int:
    import pandas as pd

    require_file(label_path, "external H=16 label CSV")
    df = pd.read_csv(label_path)
    required = {
        "sequence", "trajectory_policy", "anchor_frame", "horizon_H", "status",
        "G_state_absrel_restart_aware_discounted_per_valid_frame",
        "feature_scheduler_distance_kf", "feature_state_features_teacher_2_mean",
        "future_metric_pairs", "future_frames_requested",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"External label CSV is missing required fields: {missing}")

    good = df[
        df["sequence"].astype(str).str.strip().str.lower().eq(expected_sequence.lower())
        & df["trajectory_policy"].astype(str).str.strip().str.lower().eq("auto")
        & pd.to_numeric(df["horizon_H"], errors="coerce").eq(16)
        & df["status"].astype(str).str.strip().str.lower().eq("ok")
    ].copy()
    if good["anchor_frame"].duplicated().any():
        raise ValueError("Duplicate auto/H=16 anchor labels were found.")
    if len(good) < 12:
        raise ValueError(
            f"Only {len(good)} valid auto/H=16 anchors; minimum is 12. "
            "Do not relax the criterion to force a pass."
        )
    print(
        f"LABEL CHECK: sequence={expected_sequence}; "
        f"valid_auto_H16_anchors={len(good)}; "
        f"anchors={sorted(pd.to_numeric(good.anchor_frame).astype(int).tolist())}"
    )

    metadata_path = label_path.parent / "p2_3_v5_15_metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["experiment_version"] = "P2.3 v5.21"
            metadata["historical_sequence_exposure"] = (
                "synchronous was used in earlier P2.3 v5.10/v5.11 experiments; "
                "this is a locked-model replication, not a pristine project-wide unseen-sequence test."
            )
            metadata["model_weights_frozen"] = True
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not annotate collection metadata: {exc}") from exc
    return len(good)


def collect(p: dict[str, Path], data_root: Path, sequence: str, frames: int, force: bool) -> None:
    check_dependencies(p)
    make_runtime_copies(p)

    if sequence.lower() != "synchronous":
        raise ValueError(
            "This registered v5.21 run is fixed to sequence=synchronous. "
            "Do not rename footage to make it appear new."
        )
    if not (data_root / "rgb.txt").is_file() or not (data_root / "depth.txt").is_file():
        raise FileNotFoundError(
            f"Expected non-empty rgb.txt and depth.txt under: {data_root}"
        )
    if (not (data_root / "rgb.txt").stat().st_size
            or not (data_root / "depth.txt").stat().st_size):
        raise FileNotFoundError(f"Empty RGB-D index file under: {data_root}")

    if p["labels"].is_file() and not force:
        count = validate_labels(p["labels"], sequence)
        print(
            f"Existing valid labels found ({count} anchors); not rerunning GPU collection. "
            "Use mode 'evaluate' to reuse them."
        )
        return

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable in this Python environment. Activate the GPU-backed "
            "svrdepth environment before running collection."
        )

    p["output"].mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(p["runtime_collector"]),
        "--data-root", str(data_root),
        "--sequence", sequence,
        "--frames", str(frames),
        "--horizon", "16",
        "--gamma", "0.9",
        "--candidate-stride", "4",
        "--candidate-offset", "1",
        "--min-refresh-gap", "4",
        "--fixed-interval", "8",
        "--budget-actions", "8",
        "--policies", "auto",
        "--out-dir", str(p["output"]),
    ]
    run_command(command, p["root"])
    validate_labels(p["labels"], sequence)


def evaluate(p: dict[str, Path], sequence: str) -> None:
    check_dependencies(p)
    make_runtime_copies(p)
    validate_labels(p["labels"], sequence)
    p["evaluation"].mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, str(p["runtime_evaluator"]),
        "--train-dataset", str(p["train"]),
        "--external-label-csv", str(p["labels"]),
        "--out-dir", str(p["evaluation"]),
        "--alpha", "10",
        "--budget-k", "8",
        "--min-valid-anchors", "12",
        "--min-mean-pair-coverage", "0.90",
    ]
    run_command(command, p["root"])

    runtime_zip = p["runtime"] / "P2_3_v5_21_external_validation_review.zip"
    require_file(runtime_zip, "generated v5.21 review archive")
    shutil.copy2(runtime_zip, p["review_zip"])
    print(f"Review ZIP: {p['review_zip']}")
    print(f"Report: {p['evaluation'] / 'P2_3_v5_21_实验报告.md'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("self-test", "collect", "evaluate", "all"), nargs="?", default="all")
    parser.add_argument("--repo-root", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--force-collect", action="store_true",
                        help="Explicitly rerun GPU collection even if a valid label CSV already exists.")
    args = parser.parse_args()

    try:
        root = resolve_repo_root(args.repo_root)
        p = paths_for(root)

        if args.mode == "self-test":
            run_self_test(p)
        elif args.mode == "collect":
            collect(p, Path(args.data_root).expanduser().resolve(), args.sequence, args.frames, args.force_collect)
        elif args.mode == "evaluate":
            evaluate(p, args.sequence)
        else:
            run_self_test(p)
            if p["labels"].is_file() and not args.force_collect:
                validate_labels(p["labels"], args.sequence)
                print("Existing labels are valid; skipping GPU collection.")
            else:
                collect(p, Path(args.data_root).expanduser().resolve(), args.sequence, args.frames, args.force_collect)
            evaluate(p, args.sequence)
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
