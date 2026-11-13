
"""P1-C0: Real VeloDepth branch-isolation test.

Tests:
1. Capture the real packed inputs used by VeloDepth.infer().
2. Initialize frame 0 as a keyframe.
3. Run frame 1 propagation independently from the same snapshot twice.
4. Verify identical outputs, restart decisions, and post-step states.
5. Verify branch tensors do not share storage.
6. Mutate branch A and verify branch B and the source snapshot stay unchanged.

Important:
This tests real-model propagation and state isolation only.
It does NOT implement or validate a real local refresh operation.
"""

import argparse
import copy
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "p1"))

from velodepth.models.velodepth import VeloDepth
from stream_state import StreamState, clone_state
from stream_state_snapshot import capture_state, restore_state
from velodepth_stream import VeloDepthStream


def clone_payload(obj):
    """Deep-copy input containers and tensors."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()

    if isinstance(obj, dict):
        return {k: clone_payload(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [clone_payload(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(clone_payload(v) for v in obj)

    return copy.deepcopy(obj)


def load_rgb(folder, count):
    paths = sorted(Path(folder).glob("*.png"))[:count]

    if len(paths) != count:
        raise RuntimeError(
            f"Expected {count} PNG images, found {len(paths)} "
            f"in {folder}"
        )

    frames = []

    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)

    return torch.from_numpy(
        np.stack(frames)
    ).permute(0, 3, 1, 2).contiguous()


def compare_nested(a, b, path="state", tolerance=1e-6):
    """Compare nested tensors, containers, dataclasses and scalars."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            print(
                f"FAIL {path}: shape {tuple(a.shape)} "
                f"!= {tuple(b.shape)}"
            )
            return False, float("inf")

        if a.numel() == 0:
            return True, 0.0

        delta = (a.detach().float() - b.detach().float()).abs()
        max_abs = float(delta.max().item())
        ok = bool(torch.isfinite(delta).all()) and max_abs <= tolerance

        if not ok:
            print(f"FAIL {path}: max_abs={max_abs:.9g}")

        return ok, max_abs

    if type(a) is not type(b):
        print(
            f"FAIL {path}: type {type(a).__name__} "
            f"!= {type(b).__name__}"
        )
        return False, float("inf")

    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            print(f"FAIL {path}: lengths differ")
            return False, float("inf")

        results = [
            compare_nested(x, y, f"{path}[{i}]", tolerance)
            for i, (x, y) in enumerate(zip(a, b))
        ]

        return (
            all(ok for ok, _ in results),
            max((d for _, d in results), default=0.0),
        )

    if isinstance(a, dict):
        if a.keys() != b.keys():
            print(f"FAIL {path}: dictionary keys differ")
            return False, float("inf")

        results = [
            compare_nested(a[k], b[k], f"{path}.{k}", tolerance)
            for k in a
        ]

        return (
            all(ok for ok, _ in results),
            max((d for _, d in results), default=0.0),
        )

    if is_dataclass(a) and not isinstance(a, type):
        results = [
            compare_nested(
                getattr(a, f.name),
                getattr(b, f.name),
                f"{path}.{f.name}",
                tolerance,
            )
            for f in fields(a)
        ]

        return (
            all(ok for ok, _ in results),
            max((d for _, d in results), default=0.0),
        )

    ok = a == b

    if not ok:
        print(f"FAIL {path}: {a!r} != {b!r}")

    return ok, 0.0 if ok else float("inf")


def collect_storage_ptrs(obj, result=None):
    """Collect tensor storage addresses recursively."""
    if result is None:
        result = set()

    if isinstance(obj, torch.Tensor):
        if obj.numel() > 0:
            result.add(obj.untyped_storage().data_ptr())

    elif is_dataclass(obj) and not isinstance(obj, type):
        for f in fields(obj):
            collect_storage_ptrs(getattr(obj, f.name), result)

    elif isinstance(obj, dict):
        for value in obj.values():
            collect_storage_ptrs(value, result)

    elif isinstance(obj, (list, tuple)):
        for value in obj:
            collect_storage_ptrs(value, result)

    return result


def compare_outputs(out_a, out_b, tolerance):
    all_ok = True

    for name in ("radius", "rays", "confidence"):
        a = getattr(out_a, name)
        b = getattr(out_b, name)

        ok, max_abs = compare_nested(
            a, b, path=f"output.{name}", tolerance=tolerance
        )

        print(
            f"{'PASS' if ok else 'FAIL'} output.{name}: "
            f"max_abs={max_abs:.9g}, "
            f"exact_equal={torch.equal(a, b)}"
        )

        all_ok = all_ok and ok

    return all_ok


def run_propagation_branch(
    model, stream, inputs, snapshot, frame_idx, total_frames, tolerance
):
    """Restore the same pre-frame snapshot before every branch run."""
    state = clone_state(snapshot.stream_state)

    restore_state(model, state, snapshot)

    output, state, restart = stream.propagate_step(
        inputs, frame_idx, state, total_frames
    )

    # Clone immediately: outputs may reference state tensors.
    output_copy = clone_payload(output)
    state_copy = clone_state(state)

    return output_copy, state_copy, restart


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rgb",
        default="/root/autodl-tmp/datasets/rgbd_bonn_crowd2/rgb",
    )
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--resolution_level", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-6)

    args = parser.parse_args()

    if args.frames < 2:
        parser.error("--frames must be at least 2")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU required. Switch AutoDL to a GPU instance."
        )

    torch.set_grad_enabled(False)

    print("Loading VeloDepth...")
    model = VeloDepth.from_pretrained("lpiccinelli/velodepth")
    model.to("cuda")
    model.eval()
    model.resolution_level = args.resolution_level
    model.interpolation_mode = "bilinear"

    print("Model device:", next(model.parameters()).device)
    print("model.training:", model.training)

    frames = load_rgb(args.rgb, args.frames)

    # Capture the exact packed inputs passed into encode_decode_test().
    captured = {}
    original_method = model.encode_decode_test

    def capture_inputs(inputs, image_metas):
        captured["inputs"] = clone_payload(inputs)
        captured["metas"] = clone_payload(image_metas)
        return original_method(inputs, image_metas)

    model.encode_decode_test = capture_inputs

    try:
        with torch.no_grad():
            model.infer(frames, normalize=True)
    finally:
        model.encode_decode_test = original_method

    if "inputs" not in captured:
        raise RuntimeError("Failed to capture VeloDepth inputs.")

    inputs = clone_payload(captured["inputs"])
    metas = clone_payload(captured["metas"])

    batch_size, total_frames = metas[0]["B"], metas[0]["T"]

    if batch_size != 1:
        raise RuntimeError(
            f"Expected batch size 1, got {batch_size}"
        )

    if total_frames < 2:
        raise RuntimeError("At least two captured frames are required.")

    # Match the preprocessing used by VeloDepthStream.run().
    model._set_image_paddings(inputs)

    inputs["flow_bwd"], inputs["grid_sample"] = (
        model._normalize_and_grid_from_flow(
            inputs["flow_bwd"],
            inputs["image"],
            inputs["image_paddings"],
            inputs["depth_paddings"],
        )
    )

    stream = VeloDepthStream(model)
    initial_state = StreamState()

    with torch.no_grad():
        # Establish the same real frame-0 keyframe state for all branches.
        keyframe_output, initial_state = stream.keyframe_step(
            inputs, 0, initial_state, total_frames
        )

        pre_frame_snapshot = capture_state(model, initial_state)
        snapshot_state_before = clone_state(
            pre_frame_snapshot.stream_state
        )

        print("\n=== Initial state ===")
        print("frame 0 action: keyframe")
        print(
            "frame 0 radius shape:",
            tuple(keyframe_output.radius.shape),
        )
        print("Snapshot captured before frame 1.")

        # Branch A: propagate from the saved frame-0 state.
        out_a, state_a, restart_a = run_propagation_branch(
            model, stream, inputs, pre_frame_snapshot,
            1, total_frames, args.tolerance
        )

        # Branch B: restore the same snapshot and repeat independently.
        out_b, state_b, restart_b = run_propagation_branch(
            model, stream, inputs, pre_frame_snapshot,
            1, total_frames, args.tolerance
        )

    print("\n=== Test 1: independent propagation reproducibility ===")

    outputs_ok = compare_outputs(out_a, out_b, args.tolerance)

    states_ok, state_max = compare_nested(
        state_a, state_b,
        path="post_frame1_state",
        tolerance=args.tolerance,
    )

    print(
        f"{'PASS' if states_ok else 'FAIL'} "
        f"post-frame state: max_abs={state_max:.9g}"
    )

    restart_ok = restart_a == restart_b

    print(
        f"{'PASS' if restart_ok else 'FAIL'} restart decision: "
        f"A={restart_a}, B={restart_b}"
    )

    print("\n=== Test 2: cross-branch storage isolation ===")

    ptr_a = collect_storage_ptrs(state_a)
    ptr_b = collect_storage_ptrs(state_b)
    ptr_snapshot = collect_storage_ptrs(snapshot_state_before)

    overlap_ab = ptr_a & ptr_b
    overlap_as = ptr_a & ptr_snapshot
    overlap_bs = ptr_b & ptr_snapshot

    storage_ok = not overlap_ab and not overlap_as and not overlap_bs

    print("Branch A tensor storages:", len(ptr_a))
    print("Branch B tensor storages:", len(ptr_b))
    print("Snapshot tensor storages:", len(ptr_snapshot))
    print("A/B shared storages:", len(overlap_ab))
    print("A/snapshot shared storages:", len(overlap_as))
    print("B/snapshot shared storages:", len(overlap_bs))
    print(
        f"{'PASS' if storage_ok else 'FAIL'} "
        "cross-branch storage isolation"
    )

    print("\n=== Test 3: mutation cannot leak across branches ===")

    state_b_before = clone_state(state_b)
    snapshot_before_mutation = clone_state(snapshot_state_before)

    if state_a.radius_teacher is None:
        raise RuntimeError("Branch A has no radius_teacher state.")

    # Deliberately mutate only Branch A.
    state_a.radius_teacher.add_(1.0)

    branch_b_ok, branch_b_max = compare_nested(
        state_b, state_b_before,
        path="branch_B_after_mutating_A",
        tolerance=0.0,
    )

    snapshot_ok, snapshot_max = compare_nested(
        pre_frame_snapshot.stream_state,
        snapshot_before_mutation,
        path="snapshot_after_mutating_A",
        tolerance=0.0,
    )

    print(
        f"{'PASS' if branch_b_ok else 'FAIL'} "
        f"branch B unchanged: max_abs={branch_b_max:.9g}"
    )
    print(
        f"{'PASS' if snapshot_ok else 'FAIL'} "
        f"source snapshot unchanged: max_abs={snapshot_max:.9g}"
    )

    # Check that internal aliases which are expected to be shared
    # within a single propagated state remain shared after cloning.
    alias_ok = True

    if (
        state_a.features_teacher is not None
        and state_a.fast_tail_features is not None
    ):
        alias_ok = (
            state_a.features_teacher is state_a.fast_tail_features
        )

    print(
        f"{'PASS' if alias_ok else 'FAIL'} "
        "within-branch feature-list alias check"
    )

    all_ok = (
        outputs_ok
        and states_ok
        and restart_ok
        and storage_ok
        and branch_b_ok
        and snapshot_ok
        and alias_ok
    )

    print("\n" + "=" * 68)
    print("REAL-MODEL BRANCH ISOLATION SUMMARY")
    print("=" * 68)
    print("Frames tested: 2")
    print("Propagation branches: 2")
    print("Output reproducibility:", "PASS" if outputs_ok else "FAIL")
    print("State reproducibility:", "PASS" if states_ok else "FAIL")
    print("Storage isolation:", "PASS" if storage_ok else "FAIL")
    print("Mutation isolation:", "PASS" if branch_b_ok and snapshot_ok else "FAIL")
    print("Internal alias check:", "PASS" if alias_ok else "FAIL")
    print("Overall:", "PASS" if all_ok else "FAIL")

    if not all_ok:
        raise AssertionError(
            "Branch isolation failed. Preserve the log and fix this "
            "before implementing real B1/B2 refresh."
        )

    print(
        "PASS: real-model propagation branches are independently "
        "reproducible and isolated."
    )
    print(
        "LIMIT: no real refresh executor or B1/B2 commit intervention "
        "has been tested yet."
    )


if __name__ == "__main__":
    main()
