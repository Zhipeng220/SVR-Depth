"""
GPU integration test: multi-frame VeloDepth stream-state restore/replay.

测试目标：
1. 捕获 VeloDepth.infer() 预处理后的真实输入，但不执行完整视频网络推理。
2. 执行第 0 帧关键帧初始化。
3. 对第 1 到第 T-1 帧逐帧执行：
   - 保存当前 StreamState 和模型运行时快照；
   - 正常执行当前帧传播；
   - 保存第一次执行结果；
   - 恢复当前帧执行前的状态；
   - 检查恢复后的 StreamState；
   - 重放当前帧；
   - 比较两次输出、传播后的状态和 restart 标志。
4. 任意一帧不一致则失败退出。

在仓库根目录运行：
    python experiments/p1/test_stream_state_restore_gpu.py \
        --frames 16 \
        --resolution_level 0

注意：
- 不训练模型，不修改模型权重。
- 需要 GPU。
- 依赖现有的 stream_state.py、
  stream_state_snapshot.py 和 velodepth_stream.py。
"""

import argparse
import copy
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


# ============================================================
# 1. 路径与项目模块
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "p1"))

from velodepth.models.velodepth import VeloDepth

from stream_state import StreamState, clone_state
from stream_state_snapshot import capture_state, restore_state
from velodepth_stream import VeloDepthStream


# ============================================================
# 2. 输入捕获中断标志
# ============================================================

class InputsCaptured(Exception):
    """
    仅用于捕获 infer() 预处理后的输入。

    当 model.infer() 调用 encode_decode_test() 时，
    保存输入后抛出该异常，阻止完整视频网络推理。

    外层只捕获此专用异常，不吞掉其他运行错误。
    """
    pass


# ============================================================
# 3. 通用深复制
# ============================================================

def clone_payload(obj):
    """
    深复制输入结构。

    Tensor 使用 detach().clone()，避免后续模型操作
    修改已捕获的输入张量。
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()

    if isinstance(obj, dict):
        return {
            key: clone_payload(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            clone_payload(value)
            for value in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            clone_payload(value)
            for value in obj
        )

    return copy.deepcopy(obj)


# ============================================================
# 4. 加载 RGB 帧
# ============================================================

def load_rgb(folder, count):
    """
    按文件名排序，读取前 count 张 PNG 图片。

    返回：
        Tensor，形状为 [T, C, H, W]，RGB，uint8。
    """
    folder = Path(folder)

    if not folder.is_dir():
        raise FileNotFoundError(
            f"RGB folder does not exist: {folder}"
        )

    paths = sorted(folder.glob("*.png"))[:count]

    if len(paths) != count:
        raise RuntimeError(
            f"Expected {count} PNG images, "
            f"found {len(paths)} in {folder}"
        )

    frames = []

    for path in paths:
        bgr = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if bgr is None:
            raise RuntimeError(
                f"Failed to read image: {path}"
            )

        rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB,
        )

        frames.append(rgb)

    array = np.stack(frames, axis=0)

    return torch.from_numpy(array).permute(
        0, 3, 1, 2
    ).contiguous()


# ============================================================
# 5. 递归比较状态
# ============================================================

def compare_nested(
    a,
    b,
    path="state",
    tolerance=1e-6,
):
    """
    比较嵌套状态。

    支持：
        Tensor、dict、list、tuple、dataclass、
        以及普通 Python 标量。

    返回：
        (是否通过, 最大绝对误差)
    """

    # ---------------- Tensor ----------------

    if isinstance(a, torch.Tensor) and isinstance(
        b, torch.Tensor
    ):
        if a.shape != b.shape:
            print(
                f"FAIL {path}: shape "
                f"{tuple(a.shape)} != {tuple(b.shape)}"
            )
            return False, float("inf")

        if a.dtype != b.dtype:
            print(
                f"FAIL {path}: dtype "
                f"{a.dtype} != {b.dtype}"
            )
            return False, float("inf")

        if a.device != b.device:
            print(
                f"FAIL {path}: device "
                f"{a.device} != {b.device}"
            )
            return False, float("inf")

        if a.numel() == 0:
            return True, 0.0

        if a.is_floating_point() or a.is_complex():
            delta = (
                a.detach().to(torch.float64)
                - b.detach().to(torch.float64)
            ).abs()

            if not torch.isfinite(delta).all():
                print(
                    f"FAIL {path}: non-finite difference"
                )
                return False, float("inf")

            maximum = float(delta.max().item())
            ok = maximum <= tolerance

        else:
            equal = torch.equal(a, b)
            maximum = 0.0 if equal else float("inf")
            ok = equal

        if not ok:
            print(
                f"FAIL {path}: max_abs={maximum:.9g}"
            )

        return ok, maximum

    # ---------------- 类型 ----------------

    if type(a) is not type(b):
        print(
            f"FAIL {path}: type "
            f"{type(a).__name__} != "
            f"{type(b).__name__}"
        )
        return False, float("inf")

    # ---------------- list / tuple ----------------

    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            print(
                f"FAIL {path}: length "
                f"{len(a)} != {len(b)}"
            )
            return False, float("inf")

        results = []

        for index, (x, y) in enumerate(zip(a, b)):
            result = compare_nested(
                x,
                y,
                path=f"{path}[{index}]",
                tolerance=tolerance,
            )
            results.append(result)

        all_ok = all(
            ok for ok, _ in results
        )

        max_abs = max(
            (value for _, value in results),
            default=0.0,
        )

        return all_ok, max_abs

    # ---------------- dict ----------------

    if isinstance(a, dict):
        if a.keys() != b.keys():
            print(
                f"FAIL {path}: dictionary keys differ"
            )
            print("A keys:", list(a.keys()))
            print("B keys:", list(b.keys()))
            return False, float("inf")

        results = []

        for key in a:
            result = compare_nested(
                a[key],
                b[key],
                path=f"{path}.{key}",
                tolerance=tolerance,
            )
            results.append(result)

        all_ok = all(
            ok for ok, _ in results
        )

        max_abs = max(
            (value for _, value in results),
            default=0.0,
        )

        return all_ok, max_abs

    # ---------------- dataclass ----------------

    if is_dataclass(a) and not isinstance(a, type):
        results = []

        for field in fields(a):
            name = field.name

            result = compare_nested(
                getattr(a, name),
                getattr(b, name),
                path=f"{path}.{name}",
                tolerance=tolerance,
            )

            results.append(result)

        all_ok = all(
            ok for ok, _ in results
        )

        max_abs = max(
            (value for _, value in results),
            default=0.0,
        )

        return all_ok, max_abs

    # ---------------- 普通标量 ----------------

    try:
        ok = a == b
        if isinstance(ok, (bool, np.bool_)):
            ok = bool(ok)
        else:
            ok = bool(ok)
    except Exception:
        ok = False

    if not ok:
        print(
            f"FAIL {path}: {a!r} != {b!r}"
        )

    return ok, 0.0 if ok else float("inf")


# ============================================================
# 6. 比较传播输出
# ============================================================

def compare_output(
    first,
    second,
    tolerance,
    prefix="",
):
    """
    比较现有流式接口返回的关键输出：
        radius
        rays
        confidence

    返回是否全部通过。
    """
    all_ok = True

    for name in (
        "radius",
        "rays",
        "confidence",
    ):
        a = getattr(first, name)
        b = getattr(second, name)

        label = (
            f"{prefix}.{name}"
            if prefix
            else f"output.{name}"
        )

        if not isinstance(a, torch.Tensor):
            print(
                f"FAIL {label}: first output "
                f"is not a Tensor"
            )
            all_ok = False
            continue

        if not isinstance(b, torch.Tensor):
            print(
                f"FAIL {label}: replay output "
                f"is not a Tensor"
            )
            all_ok = False
            continue

        if a.shape != b.shape:
            print(
                f"FAIL {label}: shape "
                f"{tuple(a.shape)} != "
                f"{tuple(b.shape)}"
            )
            all_ok = False
            continue

        if a.dtype != b.dtype:
            print(
                f"FAIL {label}: dtype "
                f"{a.dtype} != {b.dtype}"
            )
            all_ok = False
            continue

        if a.numel() == 0:
            maximum = 0.0
            mean = 0.0
        else:
            delta = (
                a.detach().float()
                - b.detach().float()
            ).abs()

            if not torch.isfinite(delta).all():
                print(
                    f"FAIL {label}: non-finite difference"
                )
                all_ok = False
                continue

            maximum = float(delta.max().item())
            mean = float(delta.mean().item())

        exact_equal = torch.equal(a, b)
        ok = maximum <= tolerance

        print(
            f"{'PASS' if ok else 'FAIL'} {label}: "
            f"max_abs={maximum:.9g}, "
            f"mean_abs={mean:.9g}, "
            f"exact_equal={exact_equal}"
        )

        all_ok = all_ok and ok

    return all_ok


# ============================================================
# 7. 捕获 infer() 的真实预处理输入
# ============================================================

def capture_infer_inputs(model, frames):
    """
    调用 model.infer()，运行原版预处理，
    在 encode_decode_test() 入口捕获输入。

    通过 InputsCaptured 异常中止完整模型推理，
    避免在正式状态测试前先执行完整视频推理。
    """
    captured = {}

    original_method = model.encode_decode_test

    def capture_inputs(inputs, image_metas):
        captured["inputs"] = clone_payload(inputs)
        captured["metas"] = clone_payload(image_metas)

        # 关键修改：捕获后立即中止原版推理。
        raise InputsCaptured()

    model.encode_decode_test = capture_inputs

    try:
        with torch.no_grad():
            model.infer(
                frames,
                normalize=True,
            )

    except InputsCaptured:
        print(
            "PASS: captured infer() inputs "
            "without executing encode_decode_test()."
        )

    finally:
        # 无论成功与否，都恢复原方法。
        model.encode_decode_test = original_method

    if "inputs" not in captured:
        raise RuntimeError(
            "Failed to capture packed inputs from model.infer()."
        )

    return (
        clone_payload(captured["inputs"]),
        clone_payload(captured["metas"]),
    )


# ============================================================
# 8. 多帧真实模型状态恢复测试
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Multi-frame VeloDepth real-model "
            "state restore/replay test"
        )
    )

    parser.add_argument(
        "--rgb",
        default=(
            "/root/autodl-tmp/datasets/"
            "rgbd_bonn_crowd2/rgb"
        ),
        help="Folder containing RGB PNG frames",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=16,
        help="Number of consecutive RGB frames to test",
    )

    parser.add_argument(
        "--resolution_level",
        type=int,
        default=0,
        help="VeloDepth resolution level, in [0, 9]",
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Maximum permitted absolute difference",
    )

    args = parser.parse_args()

    # ---------------- 参数检查 ----------------

    if args.frames < 2:
        parser.error(
            "--frames must be at least 2"
        )

    if not 0 <= args.resolution_level < 10:
        parser.error(
            "--resolution_level must be in [0, 9]"
        )

    if args.tolerance < 0:
        parser.error(
            "--tolerance must be non-negative"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU required. Switch AutoDL to a GPU instance."
        )

    torch.set_grad_enabled(False)

    # ---------------- 加载模型 ----------------

    print("Loading VeloDepth model...")

    model = VeloDepth.from_pretrained(
        "lpiccinelli/velodepth"
    )

    model.to("cuda")
    model.eval()

    model.resolution_level = args.resolution_level
    model.interpolation_mode = "bilinear"

    print("Model device:", model.device)
    print("model.training:", model.training)

    # ---------------- 加载帧 ----------------

    print(
        f"Loading {args.frames} frames "
        f"from: {args.rgb}"
    )

    frames = load_rgb(
        args.rgb,
        args.frames,
    )

    print(
        "Loaded RGB tensor:",
        tuple(frames.shape),
        frames.dtype,
    )

    # ---------------- 捕获输入 ----------------

    inputs, metas = capture_infer_inputs(
        model,
        frames,
    )

    B, T = (
        metas[0]["B"],
        metas[0]["T"],
    )

    print("Captured B:", B)
    print("Captured T:", T)

    if B != 1:
        raise RuntimeError(
            f"This P1 test supports B=1 only, got B={B}."
        )

    if T != args.frames:
        raise RuntimeError(
            f"Requested {args.frames} frames, "
            f"but infer() produced T={T}."
        )

    if T < 2:
        raise RuntimeError(
            f"Captured T={T}; need at least 2 frames."
        )

    # ---------------- 与流式接口保持一致的预处理 ----------------

    # 匹配 VeloDepthStream.run() 中的预处理步骤。
    model._set_image_paddings(inputs)

    inputs["flow_bwd"], inputs["grid_sample"] = (
        model._normalize_and_grid_from_flow(
            inputs["flow_bwd"],
            inputs["image"],
            inputs["image_paddings"],
            inputs["depth_paddings"],
        )
    )

    # 输入在后续传播中可能被修改。
    # 保留不可变基准，确保每次重放使用相同的输入。
    inputs_reference = clone_payload(inputs)

    # ---------------- 初始化流式状态 ----------------

    stream = VeloDepthStream(model)
    state = StreamState()

    tested_frames = 0
    overall_ok = True

    print()
    print("=" * 68)
    print("Real-model multi-frame restore/replay test")
    print("=" * 68)

    # ---------------- 第 0 帧：关键帧初始化 ----------------

    with torch.no_grad():
        _, state = stream.keyframe_step(
            inputs,
            0,
            state,
            T,
        )

    print("frame 0 action: keyframe")
    print("PASS: keyframe initialization completed.")

    # ========================================================
    # 对第 1 到 T-1 帧逐帧恢复与重放
    # ========================================================

    for frame_idx in range(1, T):

        print()
        print("-" * 68)
        print(
            f"Testing frame {frame_idx}/{T - 1}"
        )

        # 每一帧开始前，使用不可变基准恢复输入。
        # 防止第一次传播对输入字典的修改污染重放。
        inputs = clone_payload(inputs_reference)

        # ----------------------------------------------------
        # A. 保存执行当前帧之前的状态
        # ----------------------------------------------------

        state_before = clone_state(state)

        snapshot = capture_state(
            model,
            state,
        )

        # ----------------------------------------------------
        # B. 第一次执行当前帧
        # ----------------------------------------------------

        with torch.no_grad():
            first_out, state, restart_first = (
                stream.propagate_step(
                    inputs,
                    frame_idx,
                    state,
                    T,
                )
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        first_state_after_step = clone_state(state)

        # ----------------------------------------------------
        # C. 恢复执行前的状态
        # ----------------------------------------------------

        restore_state(
            model,
            state,
            snapshot,
        )

        restored_state = clone_state(state)

        restore_ok, restore_max = compare_nested(
            state_before,
            restored_state,
            path=f"frame_{frame_idx}.restored_state",
            tolerance=args.tolerance,
        )

        print(
            f"{'PASS' if restore_ok else 'FAIL'} "
            f"frame {frame_idx} restored StreamState: "
            f"max_abs={restore_max:.9g}"
        )

        if not restore_ok:
            raise AssertionError(
                f"StreamState restoration failed "
                f"at frame {frame_idx}."
            )

        # ----------------------------------------------------
        # D. 使用相同输入重放当前帧
        # ----------------------------------------------------

        inputs = clone_payload(inputs_reference)

        with torch.no_grad():
            replay_out, state, restart_replay = (
                stream.propagate_step(
                    inputs,
                    frame_idx,
                    state,
                    T,
                )
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        replay_state_after_step = clone_state(state)

        # ----------------------------------------------------
        # E. 比较两次执行的输出
        # ----------------------------------------------------

        outputs_ok = compare_output(
            first_out,
            replay_out,
            args.tolerance,
            prefix=f"frame_{frame_idx}",
        )

        # ----------------------------------------------------
        # F. 比较两次执行后的 StreamState
        # ----------------------------------------------------

        states_ok, state_max = compare_nested(
            first_state_after_step,
            replay_state_after_step,
            path=f"frame_{frame_idx}.post_step_state",
            tolerance=args.tolerance,
        )

        print(
            f"{'PASS' if states_ok else 'FAIL'} "
            f"frame {frame_idx} post-step StreamState: "
            f"max_abs={state_max:.9g}"
        )

        # ----------------------------------------------------
        # G. 比较是否触发 restart
        # ----------------------------------------------------

        restart_ok = (
            restart_first == restart_replay
        )

        print(
            f"{'PASS' if restart_ok else 'FAIL'} "
            f"frame {frame_idx} restart consistency: "
            f"first={restart_first}, "
            f"replay={restart_replay}"
        )

        # ----------------------------------------------------
        # H. 当前帧验收
        # ----------------------------------------------------

        frame_ok = (
            restore_ok
            and outputs_ok
            and states_ok
            and restart_ok
        )

        if not frame_ok:
            overall_ok = False

            raise AssertionError(
                f"Restore/replay failed at frame {frame_idx}. "
                "Stop here; do not proceed to B1/B2."
            )

        tested_frames += 1

        print(
            f"PASS: frame {frame_idx} "
            "restore/replay completed."
        )

        # 注意：
        # 此时 state 是重放后得到的状态。
        # 它已通过与第一次执行后状态的比较，
        # 可以继续用于下一帧的测试。

    # ========================================================
    # 最终汇总
    # ========================================================

    print()
    print("=" * 68)
    print("Multi-frame restore/replay summary")
    print("=" * 68)

    print("Requested frames:", args.frames)
    print("Captured frames:", T)
    print("Keyframe frames:", 1)
    print("Propagation frames tested:", tested_frames)
    print("Expected propagation frames:", T - 1)
    print("Overall status:", "PASS" if overall_ok else "FAIL")

    if (
        overall_ok
        and tested_frames == T - 1
    ):
        print(
            f"PASS: all {tested_frames} propagation frames "
            "passed restore/replay checks."
        )
    else:
        raise AssertionError(
            "Multi-frame restore/replay test did not "
            "complete successfully."
        )


if __name__ == "__main__":
    main()