"""CPU-only tests for stream_state_snapshot.py.

Place this file next to stream_state.py and stream_state_snapshot.py in
experiments/p1, then run:
    python experiments/p1/test_stream_state_snapshot.py
"""
from types import SimpleNamespace

import torch

from stream_state import StreamState
from stream_state_snapshot import capture_state, clone_snapshot, restore_state


def make_dummy_model():
    return SimpleNamespace(
        pixel_decoder=SimpleNamespace(
            test_gt_camera=False,
            camera_gt=True,
            common_shape=(31, 41),
            angular_module=SimpleNamespace(shapes=(434, 574)),
            radial_module=SimpleNamespace(
                shapes=(31, 41), original_shapes=(434, 574)
            ),
        )
    )


def main():
    model = make_dummy_model()
    shared = torch.tensor([1.0, 2.0])
    state = StreamState(
        features_teacher=[shared],
        fast_tail_features=[shared],
        radius_teacher=shared,
        current_t=3,
        distance_kf=2,
        first_keyframe=False,
    )

    snapshot = capture_state(model, state)

    # Mutate both stream state and decoder runtime attributes.
    state.radius_teacher.add_(10)
    state.current_t = 99
    model.pixel_decoder.camera_gt = False
    model.pixel_decoder.common_shape = (1, 1)
    model.pixel_decoder.angular_module.shapes = (1, 1)
    model.pixel_decoder.radial_module.original_shapes = (1, 1)

    restore_state(model, state, snapshot)
    assert torch.equal(state.radius_teacher, torch.tensor([1.0, 2.0]))
    assert state.current_t == 3
    assert state.distance_kf == 2
    assert state.first_keyframe is False
    assert model.pixel_decoder.camera_gt is True
    assert model.pixel_decoder.common_shape == (31, 41)
    assert model.pixel_decoder.angular_module.shapes == (434, 574)
    assert model.pixel_decoder.radial_module.original_shapes == (434, 574)
    assert state.features_teacher[0] is state.fast_tail_features[0]
    print("PASS: stream state and decoder runtime attributes restored")

    # If an attribute did not exist at snapshot time, restore removes it.
    del model.pixel_decoder.test_gt_camera
    snapshot_without_flag = capture_state(model, state)
    model.pixel_decoder.test_gt_camera = True
    restore_state(model, state, snapshot_without_flag)
    assert not hasattr(model.pixel_decoder, "test_gt_camera")
    print("PASS: attributes created after snapshot are removed")

    # A cloned snapshot must be independent of the source snapshot.
    snapshot_copy = clone_snapshot(snapshot)
    snapshot_copy.stream_state.radius_teacher.add_(5)
    assert torch.equal(snapshot.stream_state.radius_teacher, torch.tensor([1.0, 2.0]))
    print("PASS: cloned snapshots are isolated")


if __name__ == "__main__":
    main()
