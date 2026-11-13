"""P1 diagnostic: repeatability of official VeloDepth vs frame-step adapter.

Save this file under experiments/p1/p1_stream_diagnose.py.
Run from the VeloDepth repository root with a GPU.
This script does not change model parameters or original repository files.
"""
import argparse
import copy
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "p1"))

from velodepth.models.velodepth import VeloDepth
from velodepth_stream import VeloDepthStream


def load_rgb(folder, n):
    paths = sorted(Path(folder).glob("*.png"))[:n]
    if len(paths) != n:
        raise RuntimeError(f"Need {n} PNG files; found {len(paths)}")
    frames = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"Failed to read {p}")
        frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()


def clone_payload(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: clone_payload(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clone_payload(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(clone_payload(v) for v in obj)
    return copy.deepcopy(obj)


def report(label, ref, alt, n, tolerance):
    print(f"\n=== {label} ===")
    for key in ("radius", "rays", "confidence", "points"):
        a, b = ref[key], alt[key]
        if a.shape != b.shape:
            print(f"{key:10s} SHAPE MISMATCH: {tuple(a.shape)} != {tuple(b.shape)}")
            continue
        if a.size(0) != n:
            print(f"{key:10s} unexpected first dim: {a.size(0)}, expected {n}")
            continue
        delta = (a.float() - b.float()).abs().reshape(n, -1)
        maximum = delta.max(dim=1).values.cpu().tolist()
        average = delta.mean(dim=1).cpu().tolist()
        bad = [i for i, d in enumerate(maximum) if d > tolerance]
        print(f"{key:10s}: first_outside_tol={bad[0] if bad else 'NONE'}, "
              f"overall_max={max(maximum):.9g}, overall_mean={sum(average)/n:.9g}")
        for i in range(n):
            if i < 4 or i == n - 1 or (bad and i == bad[0]):
                print(f"  t={i:02d} max={maximum[i]:.9g} mean={average[i]:.9g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", default="/root/autodl-tmp/datasets/rgbd_bonn_crowd2/rgb")
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--resolution_level", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    args = ap.parse_args()
    if args.frames < 1:
        ap.error("--frames must be >= 1")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; use AutoDL GPU mode")
    torch.set_grad_enabled(False)
    model = VeloDepth.from_pretrained("lpiccinelli/velodepth")
    model.to("cuda")
    model.eval()  # Do not assign the return value to model.
    model.resolution_level = args.resolution_level
    model.interpolation_mode = "bilinear"
    print("model.training =", model.training)
    frames = load_rgb(args.rgb, args.frames)

    captured = {}
    original_method = model.encode_decode_test

    def capture(inputs, image_metas):
        # Snapshot BEFORE the original method alters inputs.
        captured["packed"] = clone_payload(inputs)
        captured["metas"] = clone_payload(image_metas)
        result_inputs, raw = original_method(inputs, image_metas)
        captured["reference"] = clone_payload(raw)
        return result_inputs, raw

    model.encode_decode_test = capture
    try:
        with torch.no_grad():
            model.infer(frames, normalize=True)
    finally:
        model.encode_decode_test = original_method
    ref = captured["reference"]
    if "packed" not in captured or "metas" not in captured:
        raise RuntimeError("Failed to capture the original model inputs")

    # Control: run the ORIGINAL implementation again on precisely the same packed input.
    with torch.no_grad():
        _, repeat_raw = original_method(
            clone_payload(captured["packed"]), clone_payload(captured["metas"])
        )
    repeat = clone_payload(repeat_raw)  # Freeze result before further model calls.
    report("Original vs original (repeatability control)",
           ref, repeat, args.frames, args.tolerance)

    # Treatment: now run the step-by-step adapter.
    stream = VeloDepthStream(model)
    with torch.no_grad():
        alternative_raw = stream.run(
            clone_payload(captured["packed"]), clone_payload(captured["metas"])
        )
    alternative = clone_payload(alternative_raw)
    report("Original vs step adapter", ref, alternative, args.frames, args.tolerance)
    print("\nstep execution_trace:", stream.trace)

    # Direct comparison: the second original run against the step adapter.
    # This distinguishes first-call effects from adapter-specific differences.
    report("Original repeat vs step adapter", repeat, alternative,
           args.frames, args.tolerance)

    print("\n=== Exact equality check: original repeat vs step adapter ===")
    for key in ("radius", "rays", "confidence", "points"):
        a, b = repeat[key], alternative[key]
        if a.shape != b.shape:
            print(f"{key:10s} SHAPE MISMATCH: {tuple(a.shape)} != {tuple(b.shape)}")
            continue
        delta = (a.float() - b.float()).abs()
        print(f"{key:10s} exact_equal={torch.equal(a, b)} "
              f"max_abs={delta.max().item():.9g} "
              f"mean_abs={delta.mean().item():.9g}")

    # Third official execution, after the adapter: check whether repeated
    # official inference is stable without a special first-call effect.
    with torch.no_grad():
        _, repeat3_raw = original_method(
            clone_payload(captured["packed"]), clone_payload(captured["metas"])
        )
    repeat3 = clone_payload(repeat3_raw)
    report("Original repeat #2 vs #3", repeat, repeat3,
           args.frames, args.tolerance)

    print("\nDiagnostic only: no pass/fail assertion. ")
    print("Compare repeated-original differences before judging adapter fidelity.")


if __name__ == "__main__":
    main()
