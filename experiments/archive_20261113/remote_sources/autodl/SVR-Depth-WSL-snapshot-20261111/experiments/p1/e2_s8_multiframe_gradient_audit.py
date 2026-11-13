import copy
import json
import os
import sys
import traceback
from pathlib import Path

import torch
from velodepth.models.velodepth import VeloDepth


def main():
    torch.manual_seed(20261023)
    torch.set_num_threads(4)

    snapshot = Path(os.environ["VDEEP_SNAPSHOT"])
    config = json.loads(
        (snapshot / "config.json").read_text(encoding="utf-8")
    )
    config = copy.deepcopy(config)
    config["model"]["persistent_feature_repair"] = {
        "enabled": True,
        "mode": "post_readout",
        "bottleneck": 32,
    }

    print("STAGE 1: construct model", flush=True)
    model = VeloDepth(config)
    model.cpu()
    model.eval()

    radial = model.pixel_decoder.radial_module
    head = model.pixel_decoder.persistent_feature_repair
    assert head is not None

    print("修复头参数量：", sum(p.numel() for p in head.parameters()))

    # Controlled recurrence: retain the actual forward_fast implementation,
    # while replacing propagation and readout with differentiable test stubs.
    # This isolates cross-frame gradient wiring from image/flow preprocessing.
    print("STAGE 2: install controlled propagation/readout boundaries", flush=True)

    def propagate_from_previous_state(teacher, residual, flow, flow_features):
        return [x.clone() for x in teacher]

    weights = torch.linspace(0.5, 1.5, 16).reshape(1, 16)

    def differentiable_readout(features):
        # Deterministic scalar readout with a nonzero gradient to the state.
        pooled = features[0][:, :16].mean(dim=(2, 3))
        depth = (pooled * weights).sum(dim=1, keepdim=True)
        confidence = 0.7 * depth
        return depth, confidence

    radial.propagate_features = propagate_from_previous_state
    radial.readout_fast = differentiable_readout

    # Initial persistent state for frame 1.
    state = [
        torch.randn(1, 512, 32, 32),
        torch.randn(1, 512, 16, 16),
        torch.randn(1, 512, 8, 8),
        torch.randn(1, 512, 4, 4),
    ]

    print("STAGE 3: execute three recurrent frames in D mode", flush=True)
    outputs = []

    for frame_idx in range(3):
        depth, confidence, next_state = radial.forward_fast(
            state,
            [],
            None,
            None,
            feature_repair=head,
            feature_repair_mode="post_readout",
        )
        outputs.append(depth)
        state = next_state
        print(
            f"frame={frame_idx + 1}, "
            f"depth_shape={tuple(depth.shape)}, "
            f"state_levels={len(state)}",
            flush=True,
        )

    # Crucial: loss uses ONLY frame 3's current readout.
    # In D mode, frame 3's readout occurs before frame 3's repair.
    # Therefore its gradient to the repair head must travel through
    # repaired persistent states from earlier frames.
    print("STAGE 4: backpropagate frame-3-only loss", flush=True)
    loss = outputs[2].sum()
    print("frame-3 loss:", loss.item(), flush=True)

    head.zero_grad(set_to_none=True)
    loss.backward()

    grads = [
        p.grad
        for p in head.parameters()
        if p.grad is not None
    ]
    if not grads:
        raise AssertionError("修复头没有收到梯度")

    finite = all(torch.isfinite(g).all().item() for g in grads)
    nonzero = sum(bool(torch.count_nonzero(g).item()) for g in grads)
    grad_l1 = sum(g.abs().sum().item() for g in grads)

    print("具有梯度的参数张量数：", len(grads), flush=True)
    print("含非零梯度的参数张量数：", nonzero, flush=True)
    print("梯度绝对值总和：", grad_l1, flush=True)
    print("梯度全部有限：", finite, flush=True)

    assert finite, "发现 NaN 或 Inf 梯度"
    assert nonzero > 0, "修复头梯度全部为零"
    assert grad_l1 > 0.0, "修复头梯度总量为零"

    print("\nE2-S8-S2 CONTROLLED RECURRENCE: PASS", flush=True)
    print(
        "SCOPE: three-frame controlled state recurrence; "
        "not yet real-video validation.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
