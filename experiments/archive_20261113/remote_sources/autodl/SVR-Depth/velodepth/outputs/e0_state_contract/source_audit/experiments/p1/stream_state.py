
from dataclasses import dataclass, fields
import torch


@dataclass
class StreamState:
    # 跨帧几何状态
    features_teacher: list | None = None
    radius_teacher: torch.Tensor | None = None
    confidence_teacher: torch.Tensor | None = None

    # 关键帧交接状态
    fast_tail_features: list | None = None
    fast_tail_radius: torch.Tensor | None = None
    fast_tail_confidence: torch.Tensor | None = None

    # 当前关键帧射线
    rays_base: torch.Tensor | None = None

    # 调度状态
    current_t: int = 0
    distance_kf: int = 1
    flow_magnitude: float = 0.0
    total_luma_change: float = 0.0
    first_keyframe: bool = True


def clone_value(value, memo):
    if value is None:
        return None

    key = id(value)
    if key in memo:
        return memo[key]

    if isinstance(value, torch.Tensor):
        result = value.detach().clone()
    elif isinstance(value, list):
        result = [clone_value(x, memo) for x in value]
    elif isinstance(value, tuple):
        result = tuple(clone_value(x, memo) for x in value)
    elif isinstance(value, dict):
        result = {
            k: clone_value(v, memo)
            for k, v in value.items()
        }
    elif isinstance(value, (int, float, bool, str)):
        return value
    else:
        raise TypeError(
            f"Unsupported state type: {type(value)}"
        )

    memo[key] = result
    return result


def clone_state(state: StreamState) -> StreamState:
    memo = {}
    values = {
        f.name: clone_value(getattr(state, f.name), memo)
        for f in fields(state)
    }
    return StreamState(**values)
