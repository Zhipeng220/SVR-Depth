"""Snapshot/restore helpers for SVR-Depth P1.

This module snapshots the Python-side streaming state and the small set of
mutable VeloDepth decoder runtime attributes that are changed during inference.
It deliberately does NOT copy model parameters or the entire model.

Expected import context: experiments/p1 is on sys.path, so ``stream_state`` can
be imported directly, as in the other P1 scripts.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any

import torch

from stream_state import StreamState, clone_state


# These attributes are assigned by Decoder.set_shapes(), AngularModule.set_shapes(),
# RadialModule.set_shapes(), VeloDepthStream.run(), and the original inference path.
# Capture existence as well as value because some code versions create
# ``test_gt_camera`` dynamically while the uploaded decoder reads ``camera_gt``.
_RUNTIME_ATTR_PATHS: tuple[tuple[str, ...], ...] = (
    ("pixel_decoder", "test_gt_camera"),
    ("pixel_decoder", "camera_gt"),
    ("pixel_decoder", "common_shape"),
    ("pixel_decoder", "angular_module", "shapes"),
    ("pixel_decoder", "radial_module", "shapes"),
    ("pixel_decoder", "radial_module", "original_shapes"),
)


@dataclass
class RuntimeAttribute:
    existed: bool
    value: Any = None


@dataclass
class StreamSnapshot:
    """A point-in-time copy of StreamState and selected decoder runtime attrs."""

    stream_state: StreamState
    runtime_attrs: dict[tuple[str, ...], RuntimeAttribute]


def _clone_value(value: Any) -> Any:
    """Clone snapshot metadata without sharing mutable Python containers/tensors."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, list):
        return [_clone_value(x) for x in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(x) for x in value)
    if isinstance(value, dict):
        return {k: _clone_value(v) for k, v in value.items()}
    return deepcopy(value)


def _get_parent_and_name(root: Any, path: tuple[str, ...]) -> tuple[Any, str]:
    if not path:
        raise ValueError("Attribute path cannot be empty")
    parent = root
    for part in path[:-1]:
        if not hasattr(parent, part):
            raise AttributeError(
                f"Cannot resolve attribute path {'.'.join(path)!r}: "
                f"missing intermediate attribute {part!r}"
            )
        parent = getattr(parent, part)
    return parent, path[-1]


def capture_state(model: Any, stream_state: StreamState) -> StreamSnapshot:
    """Capture the stream state and selected mutable decoder attributes.

    The model must already be initialized. Missing leaf attributes are recorded
    as absent so restore can remove attributes created after the snapshot.
    """
    runtime_attrs: dict[tuple[str, ...], RuntimeAttribute] = {}
    for path in _RUNTIME_ATTR_PATHS:
        parent, name = _get_parent_and_name(model, path)
        exists = hasattr(parent, name)
        value = _clone_value(getattr(parent, name)) if exists else None
        runtime_attrs[path] = RuntimeAttribute(existed=exists, value=value)

    return StreamSnapshot(
        stream_state=clone_state(stream_state),
        runtime_attrs=runtime_attrs,
    )


def clone_snapshot(snapshot: StreamSnapshot) -> StreamSnapshot:
    """Return an independent copy of an existing snapshot."""
    return StreamSnapshot(
        stream_state=clone_state(snapshot.stream_state),
        runtime_attrs={
            path: RuntimeAttribute(
                existed=attr.existed,
                value=_clone_value(attr.value) if attr.existed else None,
            )
            for path, attr in snapshot.runtime_attrs.items()
        },
    )


def restore_state(
    model: Any,
    stream_state: StreamState,
    snapshot: StreamSnapshot,
) -> StreamState:
    """Restore model runtime attributes and overwrite ``stream_state`` in place.

    Tensor/container data is cloned from the snapshot, so the restored state
    does not alias the snapshot. The supplied StreamState object is preserved
    to avoid invalidating references held by the caller.
    """
    for path, attr in snapshot.runtime_attrs.items():
        parent, name = _get_parent_and_name(model, path)
        if attr.existed:
            setattr(parent, name, _clone_value(attr.value))
        elif hasattr(parent, name):
            delattr(parent, name)

    restored = clone_state(snapshot.stream_state)
    for field in fields(StreamState):
        setattr(stream_state, field.name, getattr(restored, field.name))
    return stream_state
