"""GPU integration test for P1 stream adapter; run from project root.

Captures EXACT tensors at the point model.infer() calls encode_decode_test(),
so pre-processing / optical flow are identical across both comparisons.
This isolates the stateful execution logic. It DOES NOT measure online latency.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "p1"))

from velodepth.models.velodepth import VeloDepth, _postprocess, get_paddings
from velodepth_stream import VeloDepthStream

def load_rgb(folder, count):
    paths = sorted(Path(folder).glob("*.png"))[:count]
    if len(paths) != count:
        raise RuntimeError(f"Expected {count} PNG images, found {len(paths)}")
    frames = []
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed reading {p}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()


def diff_stats(a, b):
    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch: {a.shape} != {b.shape}")
    diff = (a - b).abs()
    return float(diff.max()), float(diff.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", default="/root/autodl-tmp/datasets/rgbd_bonn_crowd2/rgb")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--resolution_level", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This integration test requires GPU; don't run in no-card mode")

    torch.set_grad_enabled(False)
    device = torch.device("cuda")

    model = VeloDepth.from_pretrained("lpiccinelli/velodepth")

    # 移动模型到 GPU
    model.to(device)

    # 切换评估模式，不要把返回值赋给 model
    model.eval()

    model.resolution_level = args.resolution_level
    model.interpolation_mode = "bilinear"

    assert model is not None
    assert not model.training
    print("PASS: VeloDepth 模型初始化成功")

    model.interpolation_mode = "bilinear"
    rgbs = load_rgb(args.rgb, args.frames)
    captured = {}

    original = model.encode_decode_test
    def capture(inputs, image_metas):
        captured["inputs"] = {
            k: v.detach().clone() if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        captured["metas"] = [dict(x) for x in image_metas]
        _, raw_outputs = original(inputs, image_metas)
        captured["original_internal"] = {
            k: v.detach().clone() if isinstance(v, torch.Tensor) else v
            for k, v in raw_outputs.items()
        }
        return _, raw_outputs

    model.encode_decode_test = capture
    try:
        official = model.infer(rgbs, normalize=True)
    finally:
        model.encode_decode_test = original

    assert "inputs" in captured, "Original encode_decode_test() was not called"
    stream = VeloDepthStream(model)
    alternative = stream.run(captured["inputs"], captured["metas"])
    orig = captured["original_internal"]
    summary = {}
    for key in ["radius", "rays", "confidence", "points"]:
        mx, avg = diff_stats(orig[key], alternative[key])
        summary[key] = {"max": mx, "mean": avg}
        print(f"{key:10s} max_abs={mx:.9g} mean_abs={avg:.9g}")

    # Recreate final depth with EXACT source postprocessing.
    H, W = rgbs.shape[-2:]
    paddings, padded_hw = get_paddings((H, W), model.shape_constraints["ratio_bounds"])
    predicted_points = _postprocess(
        alternative["points"], padded_hw, paddings=paddings,
        interpolation_mode=model.interpolation_mode
    )
    stream_depth = predicted_points[:, -1:]
    mxd, avd = diff_stats(official["depth"], stream_depth)
    summary["final_depth"] = {"max": mxd, "mean": avd}
    print(f"final_depth max_abs={mxd:.9g} mean_abs={avd:.9g}")
    print("execution_trace:", stream.trace)

    folder = ROOT / "outputs" / "p1_trace"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "stream_equivalence.json").write_text(
        json.dumps({"stats": summary, "trace": stream.trace}, indent=2),
        encoding="utf-8"
    )
    if max(x["max"] for x in summary.values()) > args.tolerance:
        raise AssertionError("FAIL: exceeds tolerance; inspect flow, keyframes and tensor differences")
    print("PASS: original / stream outputs within tolerance")


if __name__ == "__main__":
    main()
