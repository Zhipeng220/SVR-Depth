
import json
import runpy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "scripts" / "demo.py"
OUTPUT = ROOT / "outputs" / "p1_trace"
OUTPUT.mkdir(parents=True, exist_ok=True)

# 对应当前已经核对过的 VeloDepth 源码行号
WATCH = {
    500: "keyframe_ready",
    522: "propagation_read",
    554: "propagation_state_updated",
    564: "keyframe_decision",
}


STATE_NAMES = [
    "features_teacher",
    "radius_teacher",
    "confidence_teacher",
    "fast_tail_features",
    "fast_tail_radius",
    "fast_tail_confidence",
    "rays_base",
]


def describe(value):
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "data_ptr": value.data_ptr(),
        }

    if isinstance(value, (list, tuple)):
        return [describe(x) for x in value]

    if value is None:
        return None

    return str(type(value))


def monitor(frame, event, arg):
    if event != "line":
        return monitor

    line = frame.f_lineno
    if line not in WATCH:
        return monitor

    loc = frame.f_locals

    record = {
        "event": WATCH[line],
        "source_line": line,
        "frame": loc.get("current_t"),
        "distance_kf": loc.get("distance_kf"),
        "first_keyframe": loc.get("_first"),
        "flow_heur": loc.get("flow_heur"),
        "luma_heur": loc.get("luma_heur"),
        "states": {
            name: describe(loc.get(name))
            for name in STATE_NAMES
        },
    }

    log.write(json.dumps(record) + "\n")
    log.flush()
    return monitor


def tracer(frame, event, arg):
    if (
        event == "call"
        and frame.f_code.co_name == "encode_decode_test"
        and Path(frame.f_code.co_filename).name == "velodepth.py"
    ):
        return monitor

    return None


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.argv = [str(DEMO)] + sys.argv[1:]

    log_path = OUTPUT / "state_trace.jsonl"

    with log_path.open("w", encoding="utf-8") as log:
        sys.settrace(tracer)
        try:
            runpy.run_path(str(DEMO), run_name="__main__")
        finally:
            sys.settrace(None)

    print("State trace saved:", log_path)
