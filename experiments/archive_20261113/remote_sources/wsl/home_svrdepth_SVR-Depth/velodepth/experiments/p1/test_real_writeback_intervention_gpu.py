
"""P1-C1: Real-model state-writeback intervention.

This is an offline diagnostic, not the final online action policy.

At frame 1:
  - Compute a propagation candidate.
  - Compute a full-frame keyframe refresh candidate.
  - Create two branches with the SAME current refresh output:
      commit:    write refreshed geometric state back
      no_commit: keep propagated persistent state

At frame 2:
  - Force one propagation step in each branch.
  - Compare outputs and resulting states.

No training and no weight modification.
"""

import copy
import sys
import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "p1"))

from velodepth.models.velodepth import VeloDepth
from stream_state import StreamState, clone_state
from stream_state_snapshot import capture_state, restore_state
from velodepth_stream import VeloDepthStream

from test_action_branch_isolation_gpu import (
    clone_payload,
    load_rgb,
    compare_outputs,
    run_propagation_branch,
)


def max_abs_diff(a, b):
    """Return the largest numeric difference in nested state values."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return float("inf")
        if a.numel() == 0:
            return 0.0
        delta = (a.detach().float() - b.detach().float()).abs()
        if not bool(torch.isfinite(delta).all()):
            return float("inf")
        return float(delta.max().item())

    if type(a) is not type(b):
        return float("inf")

    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return float("inf")
        return max(
            (max_abs_diff(x, y) for x, y in zip(a, b)),
            default=0.0,
        )

    if isinstance(a, dict):
        if a.keys() != b.keys():
            return float("inf")
        return max(
            (max_abs_diff(a[k], b[k]) for k in a),
            default=0.0,
        )

    if hasattr(a, "__dataclass_fields__"):
        return max(
            (
                max_abs_diff(
                    getattr(a, name),
                    getattr(b, name),
                )
                for name in a.__dataclass_fields__
            ),
            default=0.0,
        )

    return 0.0 if a == b else float("inf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rgb",
        default="/root/autodl-tmp/datasets/rgbd_bonn_crowd2/rgb",
    )
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--resolution_level", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    if args.frames < 3:
        parser.error("--frames must be at least 3")

    if not torch.cuda.is_available():
        raise RuntimeError("A GPU instance is required.")

    torch.set_grad_enabled(False)

    print("Loading VeloDepth...")
    model = VeloDepth.from_pretrained("lpiccinelli/velodepth")
    model.to("cuda")
    model.eval()
    model.resolution_level = args.resolution_level
    model.interpolation_mode = "bilinear"

    frames = load_rgb(args.rgb, args.frames)

    # Capture the exact packed inputs used by the original inference path.
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
        raise RuntimeError("Failed to capture packed model inputs.")

    inputs = clone_payload(captured["inputs"])
    metas = clone_payload(captured["metas"])
    batch_size, total_frames = metas[0]["B"], metas[0]["T"]

    if batch_size != 1 or total_frames < 3:
        raise RuntimeError(
            f"Expected B=1 and T>=3, got B={batch_size}, T={total_frames}"
        )

    model._set_image_paddings(inputs)
    inputs["flow_bwd"], inputs["grid_sample"] = (
        model._normalize_and_grid_from_flow(
            inputs["flow_bwd"],
            inputs["image"],
            inputs["image_paddings"],
            inputs["depth_paddings"],
        )
    )

    # Match the original inference path when camera rays are provided.
    if "rays" in inputs:
        model.pixel_decoder.test_gt_camera = True

    stream = VeloDepthStream(model)
    state0 = StreamState()

    with torch.no_grad():
        # Initialize from frame 0.
        _, state0 = stream.keyframe_step(
            inputs, 0, state0, total_frames
        )
        snapshot0 = capture_state(model, state0)

        print("\n=== Step 1: frame-1 propagation candidate ===")

        prop_out, prop_state, restart_prop = (
            run_propagation_branch(
                model,
                stream,
                inputs,
                snapshot0,
                1,
                total_frames,
                args.tolerance,
            )
        )

        print("Propagation requested restart:", restart_prop)

        print("\n=== Step 2: frame-1 full-refresh candidate ===")

        # Restore the identical frame-0 state before full refresh.
        refresh_state = clone_state(snapshot0.stream_state)
        restore_state(model, refresh_state, snapshot0)

        refresh_out, refresh_state = stream.keyframe_step(
            inputs, 1, refresh_state, total_frames
        )

        # This is a diagnostic oracle candidate, not an online local refresh.
        print("Full-refresh candidate computed.")
        print("Refresh radius shape:", tuple(refresh_out.radius.shape))

        # Branch A: commit refreshed geometric state.
        # Preserve the propagation scheduler state so this experiment isolates
        # geometric writeback rather than resetting keyframe scheduling.
        commit_state = clone_state(prop_state)
        refresh_geometry = clone_state(refresh_state)

        commit_state.features_teacher = refresh_geometry.features_teacher
        commit_state.radius_teacher = refresh_geometry.radius_teacher
        commit_state.confidence_teacher = refresh_geometry.confidence_teacher
        commit_state.rays_base = refresh_geometry.rays_base

        # Branch B: do not commit refreshed geometric state.
        no_commit_state = clone_state(prop_state)

        # The two branches intentionally emit the same current-frame refresh.
        current_commit = clone_payload(refresh_out)
        current_no_commit = clone_payload(refresh_out)

        print("\n=== Step 3: current-frame output control ===")
        current_ok = compare_outputs(
            current_commit,
            current_no_commit,
            args.tolerance,
        )

        print("\n=== Step 4: verify persistent states differ ===")
        diff_features = max_abs_diff(
            commit_state.features_teacher,
            no_commit_state.features_teacher,
        )
        diff_radius = max_abs_diff(
            commit_state.radius_teacher,
            no_commit_state.radius_teacher,
        )
        diff_confidence = max_abs_diff(
            commit_state.confidence_teacher,
            no_commit_state.confidence_teacher,
        )
        diff_rays = max_abs_diff(
            commit_state.rays_base,
            no_commit_state.rays_base,
        )

        print("features_teacher max_abs_diff:", diff_features)
        print("radius_teacher max_abs_diff:", diff_radius)
        print("confidence_teacher max_abs_diff:", diff_confidence)
        print("rays_base max_abs_diff:", diff_rays)

        state_diff_observed = any(
            d > 0
            for d in (
                diff_features,
                diff_radius,
                diff_confidence,
                diff_rays,
            )
        )
        print(
            "Persistent geometric state differs:",
            state_diff_observed,
        )

        # Capture independent snapshots for the same future frame.
        snapshot_commit = capture_state(model, commit_state)
        snapshot_no_commit = capture_state(model, no_commit_state)

        print("\n=== Step 5: frame-2 forced propagation ===")

        out_commit, state_commit_after, restart_commit = (
            run_propagation_branch(
                model,
                stream,
                inputs,
                snapshot_commit,
                2,
                total_frames,
                args.tolerance,
            )
        )

        out_no_commit, state_no_commit_after, restart_no_commit = (
            run_propagation_branch(
                model,
                stream,
                inputs,
                snapshot_no_commit,
                2,
                total_frames,
                args.tolerance,
            )
        )

    print("\n=== Step 6: measure downstream effect ===")

    print("Commit-branch restart signal:", restart_commit)
    print("No-commit restart signal:", restart_no_commit)

    future_same = compare_outputs(
        out_commit,
        out_no_commit,
        args.tolerance,
    )

    future_output_diff = max(
        max_abs_diff(out_commit.radius, out_no_commit.radius),
        max_abs_diff(out_commit.rays, out_no_commit.rays),
        max_abs_diff(out_commit.confidence, out_no_commit.confidence),
    )

    future_state_diff = max_abs_diff(
        state_commit_after,
        state_no_commit_after,
    )

    print("Frame-2 output max_abs_diff:", future_output_diff)
    print("Frame-2 state max_abs_diff:", future_state_diff)
    print("Frame-2 outputs equal within tolerance:", future_same)

    print("\n" + "=" * 64)
    print("WRITEBACK INTERVENTION SUMMARY")
    print("=" * 64)
    print("Current refresh outputs identical:", current_ok)
    print("Persistent states differ:", state_diff_observed)
    print("Future output difference:", future_output_diff)
    print("Future state difference:", future_state_diff)
    print("Future output effect observed:", not future_same)
    print(
        "NOTE: frame-2 restart signals are logged but no keyframe "
        "is executed in this forced-propagation comparison."
    )
    print(
        "LIMIT: this experiment does not yet measure GT depth error, "
        "long-horizon benefit, or local-refresh performance."
    )

    if not current_ok:
        raise AssertionError(
            "Current-frame output control failed."
        )

    if not state_diff_observed:
        raise AssertionError(
            "Refresh candidate did not change the selected persistent state."
        )

    print("PASS: same-output state-writeback intervention constructed.")


if __name__ == "__main__":
    main()
