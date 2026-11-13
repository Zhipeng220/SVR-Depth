"""Frame-at-a-time, fixed-interval propagation scheduling."""
from __future__ import annotations
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class StreamOutput:
    index: int
    age: int
    is_keyframe: bool
    prediction: Any
    state: Any

def run_fixed_interval(frames: Iterable[Any], *, interval: int,
                       keyframe: Callable[[Any], tuple[Any, Any]],
                       propagate: Callable[[Any, Any, Any], tuple[Any, Any]]) -> Iterator[StreamOutput]:
    """Run every frame in order; callbacks supply actual model computation.

    `keyframe(frame)` and `propagate(previous_frame, frame, state)` each return
    `(prediction, committed_state)`. This scheduler does not load a model,
    compute flow, or implement adaptive keyframes.
    """
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError('interval must be a positive integer')
    state = previous = None
    for index, frame in enumerate(frames):
        age = index % interval
        if age == 0:
            prediction, state = keyframe(frame)
        else:
            prediction, state = propagate(previous, frame, state)
        yield StreamOutput(index=index, age=age, is_keyframe=(age == 0), prediction=prediction, state=state)
        previous = frame
