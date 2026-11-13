
from dataclasses import dataclass
from typing import Callable

import torch

from stream_state import StreamState, clone_state


@dataclass
class StepResult:
    """单个动作的输出和下一帧状态。"""
    depth: torch.Tensor
    state: StreamState


@dataclass
class BranchResult:
    """三个动作的反事实分支。"""
    b0: StepResult
    b1: StepResult
    b2: StepResult


class StreamAdapter:
    """
    SVR-Depth P1：状态分支控制器

    B0：传播输出 + 传播状态
    B1：刷新输出 + 刷新状态（写回）
    B2：刷新输出 + 传播状态（不写回）

    当前是模型无关的动作逻辑层。
    后续将接入 VeloDepth 的真实推理函数。
    """

    def __init__(
        self,
        propagate_fn: Callable,
        refresh_fn: Callable,
    ):
        self.propagate_fn = propagate_fn
        self.refresh_fn = refresh_fn

    @torch.no_grad()
    def branch(
        self,
        frame: torch.Tensor,
        incoming_state: StreamState,
    ) -> BranchResult:

        # 1. 使用独立状态计算传播结果
        prop = self.propagate_fn(
            frame,
            clone_state(incoming_state),
        )

        # 2. 从传播状态生成刷新候选
        # 克隆状态，避免刷新修改传播分支
        refreshed = self.refresh_fn(
            frame,
            clone_state(prop.state),
        )

        # 3. B0：传播、不刷新
        b0 = StepResult(
            depth=prop.depth.detach().clone(),
            state=clone_state(prop.state),
        )

        # 4. B1：刷新并写回状态
        b1 = StepResult(
            depth=refreshed.depth.detach().clone(),
            state=clone_state(refreshed.state),
        )

        # 5. B2：刷新但不写回状态
        b2 = StepResult(
            depth=refreshed.depth.detach().clone(),
            state=clone_state(prop.state),
        )

        # 6. 验证当前帧 B1/B2 输出相同
        assert torch.equal(b1.depth, b2.depth), (
            "B1/B2 当前输出不一致"
        )

        return BranchResult(
            b0=b0,
            b1=b1,
            b2=b2,
        )
