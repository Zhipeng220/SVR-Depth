from pathlib import Path
from datetime import datetime
import sys, json, hashlib, time

import cv2
import h5py
import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/SVR-Depth-CVPR")
HELPERS = ROOT / "experiments/crosssource_v2/legacy_helpers"

H5 = Path(
    "/root/autodl-tmp/datasets/VeloDepth-Evaluation/Bonn.hdf5"
)

SEQ = "/scratch/lpiccinelli/datasets/bonn/outputs/static_train"

BASE_SHA = (
    "615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e"
)

N = 16

assert torch.cuda.is_available(), "CUDA unavailable"
torch.cuda.set_device(0)
device = torch.device("cuda:0")

# 与原 CrossSource-v2 评测入口保持相同导入顺序
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "scripts"),
    str(HELPERS),
]

import gt2_fixed_clip_repair_train as gt2
import gt4a2_exp1_train_eval as exp1
import velodepth.models.velodepth as vd


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


print("========== M01 P_MULTI K16 SMOKE ==========")
print("GPU:", torch.cuda.get_device_name(0))

assert Path(vd.__file__).resolve().is_relative_to(
    ROOT / "velodepth"
), vd.__file__

# 1. 定位与训练时一致的预训练底座
candidates = [
    ROOT / "outputs/rk3588_handoff_e8c_ubbrvwzx/snapshot",

    Path(
        "/root/autodl-tmp/SVR-Depth/hf_cache/hub/"
        "models--lpiccinelli--velodepth/snapshots/"
        "2b2ee9fe44d0b261e736653042cecd7693f99c8c"
    ),
]

snapshot = None

for p in candidates:
    if (p / "model.safetensors").is_file():
        actual = sha(p / "model.safetensors")
        print("Snapshot:", p)
        print("SHA256:", actual)

        if actual == BASE_SHA:
            snapshot = p
            break

if snapshot is None:
    raise RuntimeError("Pinned VeloDepth snapshot not found")

print("[PASS] BASE MODEL")

# 2. 检查 E4-C 和 E7 权重身份
runs = [
    (
        "E4-C",
        "outputs/crosssource_v2_E4_20261104_C_s20261103_1000step",
        "configs/crosssource_v2/generated_20261103/C_tartanair_kitti.json",
        {"tartanair": 0.5, "kitti_sceneflow": 0.5},
        0.268657,
    ),
    (
        "E7",
        "outputs/crosssource_v2_E7_20261104_D_kitti_heavy_1000step",
        "configs/crosssource_v2/generated_20261103/D_kitti_heavy_75_25.json",
        {"tartanair": 0.25, "kitti_sceneflow": 0.75},
        0.277792,
    ),
]

ckpts = {}
init_hash = None

for label, run_rel, cfg_rel, weights, expected_loss in runs:

    run = ROOT / run_rel
    cfg = ROOT / cfg_rel

    identity = json.loads(
        (run / "run_identity.json").read_text()
    )

    config = json.loads(cfg.read_text())

    ckpt_path = run / "best_checkpoint.pt"

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    assert identity["protocol"] == "SVR-Depth-CrossSource-v2"
    assert identity["method"] == "P_multi"

    assert identity["steps"] == 1000
    assert identity["window_frames"] == 20
    assert identity["tbptt_frames"] == 4

    assert identity["dataset_weights"] == weights
    assert config["training"]["dataset_weights"] == weights

    assert identity["snapshot_sha256"] == BASE_SHA
    assert identity["config_sha256"] == sha(cfg)

    assert ckpt["identity"] == identity

    head_state = ckpt["head_state"]

    params = sum(
        v.numel() for v in head_state.values()
    )

    assert params == 170496

    if init_hash is None:
        init_hash = ckpt["init_head_sha256"]

    assert ckpt["init_head_sha256"] == init_hash

    print(
        f"[CHECKPOINT PASS] {label}: "
        f"step={ckpt['step']} "
        f"loss={ckpt['best_val_loss']:.6f} "
        f"reference={expected_loss:.6f} "
        f"params={params}"
    )

    ckpts[label] = (ckpt_path, ckpt)

# 3. 读取 Bonn train split，仅使用 RGB
print("\n========== BONN RGB INPUT ==========")

with h5py.File(H5, "r") as f:

    sequences = json.loads(
        bytes(f["sequences.json"][()]).decode("utf-8-sig")
    )

    assert SEQ in sequences

    train_text = bytes(
        f["train.txt"][()]
    ).decode("utf-8-sig")

    assert SEQ in train_text

    image_keys = sequences[SEQ]["image"]

    assert len(image_keys) >= N

    rgbs = []

    for key in image_keys[:N]:

        raw = np.asarray(
            f[key][()],
            dtype=np.uint8,
        )

        bgr = cv2.imdecode(
            np.frombuffer(raw.tobytes(), np.uint8),
            cv2.IMREAD_COLOR,
        )

        if bgr is None:
            raise RuntimeError(f"Decode failed: {key}")

        rgbs.append(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        )

H, W = rgbs[0].shape[:2]

assert all(x.shape == (H, W, 3) for x in rgbs)

print("Sequence:", SEQ)
print("Frames:", N)
print("Resolution:", W, "x", H)

# 4. 加载同一底座与 pre_readout P_multi
print("\n========== LOAD P_MULTI MODEL ==========")

model = gt2.load_model(
    snapshot,
    "pre_readout",
    device,
)

head = model.pixel_decoder.persistent_feature_repair

assert head is not None

assert (
    model.pixel_decoder.feature_repair_mode
    == "pre_readout"
)

for p in model.parameters():
    p.requires_grad_(False)

model.eval()
head.eval()

# 与 CrossSource 训练 helper 相同的 resolution level
paddings, (ph, pw) = gt2.get_paddings(
    (H, W),
    model.shape_constraints["ratio_bounds"],
)

pmin = float(model.shape_constraints["pixels_min"])
pmax = float(model.shape_constraints["pixels_max"])

_, (nh, nw) = gt2.get_resize_factor(
    (ph, pw),
    (pmin, pmin + (pmax - pmin) / 10.0),
)

mean = torch.tensor(
    gt2.IMAGENET_DATASET_MEAN,
    device=device,
).view(1, 3, 1, 1)

std = torch.tensor(
    gt2.IMAGENET_DATASET_STD,
    device=device,
).view(1, 3, 1, 1)

shape = {
    "original_shape": (H, W),
    "paddings": paddings,
    "padded_shape": (ph, pw),
    "network_shape": (nh, nw),
    "mean": mean,
    "std": std,
    "validity_mask": torch.ones(
        (1, 1, nh, nw),
        device=device,
        dtype=torch.bool,
    ),
    "image_paddings": torch.zeros(
        (1, 4), device=device, dtype=torch.long
    ),
    "depth_paddings": torch.zeros(
        (1, 4), device=device, dtype=torch.long
    ),
}

frames = []

for i, rgb in enumerate(rgbs):

    img, frame_u8 = gt2.preprocess_rgb(
        rgb,
        device,
        paddings,
        (nh, nw),
        mean,
        std,
    )

    frames.append({
        "index": i,
        "img": img,
        "frame_u8": frame_u8,
        "rgb_key": str(image_keys[i]),
    })

# 5. 固定 K16 连续推理
OUT = ROOT / "outputs" / (
    f"m01_pmulti_bonn_k16_smoke_{datetime.now():%Y%m%d_%H%M%S}"
)

OUT.mkdir(parents=True, exist_ok=False)

dis = cv2.DISOpticalFlow_create(
    cv2.DISOPTICAL_FLOW_PRESET_FAST
)

summary = {
    "status": "SMOKE_ONLY_NOT_OFFICIAL_M01_SCORE",
    "protocol": "P_multi_pre_readout_K16_gain1_f1_raw",
    "dataset": "Bonn",
    "split": "train.txt",
    "sequence": SEQ,
    "frames": N,
    "keyframes": 1,
    "propagation_steps": 15,
    "gt_loaded": False,
    "official_scoring": False,
    "base_snapshot_sha256": BASE_SHA,
    "input_rgb_keys": [str(x) for x in image_keys[:N]],
    "models": {},
}

for label, (cp_path, ckpt) in ckpts.items():

    print(f"\n========== RUN {label} ==========")

    head.load_state_dict(
        ckpt["head_state"],
        strict=True,
    )

    head.eval()
    model.eval()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():

        # Frame 0: keyframe
        rays, radius, confidence, features = (
            exp1.keyframe_state(
                model,
                frames[0],
                shape,
            )
        )

        preds = [
            gt2.make_z_depth(
                radius,
                rays,
                shape["padded_shape"],
                paddings,
                model.interpolation_mode,
            ).float().cpu().numpy()
        ]

        # Frames 1-15: continuous P_multi propagation
        for t in range(1, N):

            static = exp1.transition_static(
                model,
                frames[t - 1],
                frames[t],
                shape,
                dis,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            ):

                radius, confidence, features = (
                    exp1.one_fast_step(
                        model,
                        frames[t - 1],
                        frames[t],
                        static,
                        radius,
                        confidence,
                        features,
                    )
                )

                pred = gt2.make_z_depth(
                    radius,
                    rays,
                    shape["padded_shape"],
                    paddings,
                    model.interpolation_mode,
                )

            preds.append(
                pred.float().cpu().numpy()
            )

            if t in (1, 4, 8, 15):
                print(f"[{label}] propagation {t}/15")

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    pred_np = np.stack(preds).astype(np.float32)

    finite = float(np.isfinite(pred_np).mean())

    positive = pred_np[
        np.isfinite(pred_np) & (pred_np > 0)
    ]

    assert pred_np.shape == (N, H, W)
    assert finite >= 0.9999
    assert positive.size > 0

    pred_path = OUT / f"{label}_depth.npy"

    np.save(pred_path, pred_np)

    result = {
        "checkpoint": str(cp_path),
        "checkpoint_sha256": sha(cp_path),
        "step": int(ckpt["step"]),
        "best_val_loss": float(ckpt["best_val_loss"]),
        "dataset_weights": ckpt["identity"]["dataset_weights"],
        "seed": int(ckpt["identity"]["seed"]),
        "init_head_sha256": ckpt["init_head_sha256"],
        "prediction_file": str(pred_path),
        "prediction_shape": list(pred_np.shape),
        "finite_ratio": finite,
        "positive_ratio": float((pred_np > 0).mean()),
        "median_depth_m": float(np.median(positive)),
        "inference_seconds": float(elapsed),
        "peak_cuda_allocated_gib": float(
            torch.cuda.max_memory_allocated(device) / 1024**3
        ),
    }

    summary["models"][label] = result

    print(
        f"[{label} PASS] "
        f"shape={pred_np.shape} "
        f"finite={finite:.6f} "
        f"median_depth={np.median(positive):.4f}m "
        f"time={elapsed:.2f}s"
    )

(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False)
)

print("\n========== SMOKE COMPLETE ==========")
print("OUTPUT:", OUT)
print("SUMMARY:", OUT / "summary.json")
print("NO GT SCORES / NO TRAINING / NO O_METHODS")
