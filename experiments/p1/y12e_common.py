"""Y12-D 1.0.3: frozen helpers and protocol definitions (stdlib only)."""
from pathlib import Path
import csv, hashlib, json, math

VERSION = "Y12-E-1.0.0"
PINNED_DECODER_SHA256 = "35ca12629accd1a2a77595f505165fa17dffab07c763a8b639604285b94bdb08"
PINNED_EXP1_SHA256 = "c24441cb405465c1a16a86127b51aade8f9af69b5d368f43d830d0e759ee3d7f"
SEQS = ("crowd2", "crowd3", "balloon2")
EXPECTED_CASES = 27
HORIZON = 4
K = 32
AGE_SET = (8, 16, 24)
PRACTICAL_SCREEN_FLOOR_ABSREL = 1e-4
SEVERE_HARM_THRESHOLD_ABSREL = 0.01
BRANCHES = ("E0_CSTAR", "E1_FEATURE_ONLY", "E2_GEOMETRY_ONLY", "E3_FULL_TEACHER")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jread(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def jwrite(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(p)


def cread(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def cwrite(path, rows):
    require(bool(rows), "Refusing to write empty CSV: " + str(path))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(p)


def canonical(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def truth(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def find_y05r_dir(explicit, root):
    root = Path(root)
    if explicit and explicit != "auto":
        p = Path(explicit).expanduser().resolve()
        require(p.is_dir(), "Y05R_DIR does not exist: " + str(p))
        return p
    candidates = []
    for p in (root / "outputs").glob("gt4a2_y05r_*/validation"):
        if not (p / "protocol.json").is_file() or not (p / "summary.json").is_file():
            continue
        try:
            protocol = jread(p / "protocol.json")
            summary = jread(p / "summary.json")
            caches_present = (p / "training/C_star/final.pt").is_file() and all(
                (p / "evaluation/own_parents/C_star" / seq / "parents.pt").is_file()
                and (p / "evaluation/own_parents/C_star" / seq / "complete.json").is_file()
                for seq in SEQS)
            if (protocol.get("experiment") == "Y05-R"
                    and protocol.get("version") == "Y05-R-1.0.1"
                    and not protocol.get("partial_smoke")
                    and summary.get("status") == "Y05R_COMPLETE_NOT_METHOD_PASS"
                    and summary.get("signature") == sha(p / "protocol.json")
                    and caches_present):
                candidates.append(p.resolve())
        except Exception:
            continue
    require(bool(candidates), "No complete Y05-R 1.0.1 validation found. Set Y05R_DIR to its validation directory; do not use the review ZIP as a substitute for checkpoints/parent caches.")
    return max(candidates, key=lambda p: (p / "summary.json").stat().st_mtime_ns)


def validate_y05r_summary(y05r_dir):
    y05r_dir = Path(y05r_dir)
    protocol_path = y05r_dir / "protocol.json"
    summary_path = y05r_dir / "summary.json"
    require(protocol_path.is_file() and summary_path.is_file(), "Y05-R protocol/summary missing")
    protocol = jread(protocol_path)
    summary = jread(summary_path)
    signature = sha(protocol_path)
    require(protocol.get("experiment") == "Y05-R" and protocol.get("version") == "Y05-R-1.0.1", "Expected the reviewed Y05-R 1.0.1 protocol")
    require(not protocol.get("partial_smoke"), "Y12-D requires a formal Y05-R run, not smoke")
    require(summary.get("status") == "Y05R_COMPLETE_NOT_METHOD_PASS", "Y05-R is not a complete formal run")
    require(summary.get("signature") == signature, "Y05-R summary/protocol signature mismatch")
    require(summary.get("method_pass") is False and summary.get("Y11_pass") is False, "Y05-R method status is unexpected; inspect source result before continuing")
    require(protocol.get("settings", {}).get("train_K") == K, "Y05-R K differs from Y12-D K32 contract")
    require(protocol.get("settings", {}).get("horizon") == HORIZON, "Y05-R H4 contract differs")
    cases = [dict(c) for c in protocol.get("eval_plan", []) if c.get("origin") == "S8"]
    require(len(cases) == EXPECTED_CASES, f"Expected 27 C* own-S8 diagnostic anchors, found {len(cases)}")
    for seq in SEQS:
        local = [c for c in cases if c.get("sequence") == seq]
        require(len(local) == 9, f"Expected 9 anchors for {seq}, found {len(local)}")
        require(sorted(int(c["age"]) for c in local) == sorted(list(AGE_SET) * 3), f"Unexpected age plan for {seq}")
    require(len({c.get("case_id") for c in cases}) == EXPECTED_CASES, "Y05-R S8 anchor case IDs are not unique")
    for c in cases:
        a = int(c["anchor"])
        require(int(c["K"]) == K and int(c["horizon"]) >= HORIZON, "Y05-R anchor K/H contract mismatch")
        require(int(c["snapshot_index"]) == a - 1, "Parent snapshot must be the frame immediately before the anchor")
        require(a % K in AGE_SET and int(c["age"]) == a % K, f"Anchor age is not 8/16/24: {c['case_id']}")
        require((a + HORIZON) // K == a // K, f"H4 crosses a K32 reset at {c['case_id']}")
    return protocol, summary, signature, sorted(cases, key=lambda c: (SEQS.index(c["sequence"]), int(c["anchor"])))


def select_cases(cases, smoke=False):
    if not smoke:
        return list(cases)
    # One deterministic, protocol-derived anchor per sequence; no error-based selection.
    chosen = []
    for seq in SEQS:
        rows = [c for c in cases if c["sequence"] == seq]
        chosen.append(rows[0])
    return chosen


def frozen_settings(smoke=False):
    return {
        "K": K,
        "schedule": "S8",
        "horizon_H4": HORIZON,
        "anchors": "27 Y05-R eval_plan cases with origin=S8; C* own natural-history parents at anchor-1",
        "branches": list(BRANCHES),
        "D0": "SKIP: native propagation at anchor; no writeback; no later repair during H4",
        "D1": "C_STAR: one C* head writeback at anchor; no later repair during H4",
        "D2": "TEACHER_STATE_COMMIT: full current-frame pixel encoder; apply pinned native RadialModule.forward_keyframe(adapted_features, fixed C* rays converted from BCHW to Bx(HW)x3); use native depth_proj/confidence_proj and native geometry transforms; commit radius, confidence and all four conditioned persistent feature levels together; retain original BCHW rays_base in the state tuple.",
        "teacher_uses_GT": False,
        "teacher_cost_is_charged_separately": True,
        "practical_screen_floor_absrel": PRACTICAL_SCREEN_FLOOR_ABSREL,
        "severe_harm_delta_absrel_threshold": SEVERE_HARM_THRESHOLD_ABSREL,
        "severe_harm_threshold_note": "Implementation-defined diagnostic threshold, carried from the project's prior per-frame harm convention; not a statistical threshold.",
        "region_proxies": {
            "normal_warp": "in-bounds and valid previous/current warp, excluding top-10-percent flow-gradient cells",
            "invalid_or_exposure_proxy": "current-valid cells whose backward warp is out-of-bounds or samples invalid previous support",
            "motion_boundary_proxy": "stable top ceil(10 percent) of flow-gradient magnitude within valid warp cells; proxy only, not semantic boundary GT",
            "quantile": 0.90,
        },
        "pixel_maps": "paired anchor-frame AbsRel-error maps: D0, D1, D2 and deltas; float16 compressed NPZ",
        "no_training": True,
        "no_future_loss": True,
        "no_gate_or_new_network": True,
        "no_hyperparameter_search": True,
        "smoke_cases": 3 if smoke else EXPECTED_CASES,
        "partial_smoke": bool(smoke),
        "decision_note": "Y12-D is a recovery-space diagnostic, not a method pass, not a statistical generalization claim, and not a compute-saving benchmark.",
    }


def finite_float(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (TypeError, ValueError):
        return None


def classify_recovery(seq_rows, macro_row, severe_harm_total):
    """Engineering screen only; 1e-4 is a practical floor, not a significance test."""
    delta = finite_float(macro_row.get("teacher_minus_cstar_H4_AbsRel"))
    if delta is None:
        return "INCONCLUSIVE_MISSING_METRICS"
    improved = sum(1 for r in seq_rows if finite_float(r.get("teacher_minus_cstar_H4_AbsRel")) is not None and float(r["teacher_minus_cstar_H4_AbsRel"]) < -PRACTICAL_SCREEN_FLOOR_ABSREL)
    if delta <= -PRACTICAL_SCREEN_FLOOR_ABSREL and improved >= 2 and int(severe_harm_total) == 0:
        return "RECOVERY_SPACE_SIGNAL_REVIEW_FOR_Y12Z"
    if delta >= -PRACTICAL_SCREEN_FLOOR_ABSREL:
        return "NO_CLEAR_TEACHER_RECOVERY_SIGNAL"
    return "MIXED_RECOVERY_OR_HARM_REQUIRES_REVIEW"


def normalized_image_from_frame(frame):
    """Return normalized BCHW image from the pinned Y05 Engine.load_frame contract."""
    require(isinstance(frame, dict), "Y05 Engine frame must be a dict")
    require("img" in frame, "Y05 Engine.load_frame contract expects frame['img']; available keys="
            + repr(sorted(str(k) for k in frame.keys())))
    return frame["img"]


def prediction_to_numpy(np, prediction):
    """Convert either a NumPy prediction or a Torch/CUDA tensor to CPU float32 NumPy.

    Y05 Engine.eval_step already returns NumPy, while Engine.read_depth returns a
    device tensor. Y12-D anchor branches must be normalized to one representation
    before metrics, hashes, and region diagnostics consume them.
    """
    value = prediction
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    require(np.issubdtype(array.dtype, np.number),
            "Prediction cannot be converted to a numeric NumPy array: " + str(array.dtype))
    return array.astype("float32", copy=False)
