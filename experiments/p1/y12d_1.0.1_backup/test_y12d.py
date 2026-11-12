"""Offline protocol tests; imports no Torch/CUDA or private dataset."""
from y12d_common import *
from pathlib import Path
import json, tempfile


def synthetic_cases():
    rows = []
    for seq, starts in (("crowd2", (768, 800, 832)), ("crowd3", (256, 512, 800)), ("balloon2", (256, 320, 416))):
        for start in starts:
            for age in AGE_SET:
                anchor = start + age
                rows.append({"case_id": f"probe_{seq}_S8_t{anchor:06d}", "sequence": seq,
                    "origin": "S8", "K": 32, "anchor": anchor, "age": age,
                    "snapshot_index": anchor - 1, "horizon": 16, "split": "tail"})
    return rows


def run():
    # Pinned Y05 Engine.load_frame exposes normalized BCHW as frame["img"].
    sentinel = object()
    assert normalized_image_from_frame({"img": sentinel}) is sentinel
    try:
        normalized_image_from_frame({"image": sentinel, "gt_np": None})
    except RuntimeError as exc:
        assert "frame['img']" in str(exc)
    else:
        raise AssertionError("Y12-D accepted unsupported frame['image'] alias")

    cases = synthetic_cases()
    fake_protocol = {"experiment": "Y05-R", "version": "Y05-R-1.0.1", "partial_smoke": False,
        "settings": {"train_K": 32, "horizon": 4}, "eval_plan": cases}
    # Exercise the complete Y05-R protocol contract using a synthetic, non-data fixture.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "protocol.json").write_text(json.dumps(fake_protocol), encoding="utf-8")
        fake_summary = {"status": "Y05R_COMPLETE_NOT_METHOD_PASS", "method_pass": False,
                        "Y11_pass": False, "signature": sha(root / "protocol.json")}
        (root / "summary.json").write_text(json.dumps(fake_summary), encoding="utf-8")
        _, _, _, validated = validate_y05r_summary(root)
        assert len(validated) == EXPECTED_CASES
    # Exercise the same plan checks without private data or CUDA.
    assert len(cases) == EXPECTED_CASES
    assert len(select_cases(cases, smoke=True)) == 3
    assert len(select_cases(cases, smoke=False)) == 27
    for seq in SEQS:
        local = [x for x in cases if x["sequence"] == seq]
        assert len(local) == 9
        assert sorted(int(x["age"]) for x in local) == sorted(list(AGE_SET) * 3)
    for c in cases:
        a = int(c["anchor"])
        assert int(c["snapshot_index"]) == a - 1
        assert a % K in AGE_SET
        assert (a + HORIZON) // K == a // K
    settings = frozen_settings(False)
    assert settings["no_training"] is True
    assert settings["teacher_uses_GT"] is False
    assert settings["branches"] == ["SKIP", "C_STAR", "TEACHER_STATE_COMMIT"]
    # Decision labels are descriptive triage only, not method pass.
    good = [{"sequence": s, "teacher_minus_cstar_H4_AbsRel": -0.0002} for s in SEQS]
    assert classify_recovery(good, {"teacher_minus_cstar_H4_AbsRel": -0.0002}, 0) == "RECOVERY_SPACE_SIGNAL_REVIEW_FOR_Y12Z"
    assert classify_recovery(good, {"teacher_minus_cstar_H4_AbsRel": 0.0002}, 0) == "NO_CLEAR_TEACHER_RECOVERY_SIGNAL"
    assert classify_recovery(good, {"teacher_minus_cstar_H4_AbsRel": -0.0002}, 1) == "MIXED_RECOVERY_OR_HARM_REQUIRES_REVIEW"
    # Region masks are disjoint by construction: normal + invalid + motion are disjoint subsets.
    import numpy as np
    valid = np.array([[1, 1, 1], [1, 1, 1]], dtype=bool)
    motion = np.array([[0, 1, 0], [0, 0, 1]], dtype=bool)
    warp = np.array([[1, 1, 0], [1, 1, 1]], dtype=bool)
    normal = warp & ~motion
    invalid = valid & ~warp
    boundary = warp & motion
    assert not np.any(normal & invalid) and not np.any(normal & boundary) and not np.any(invalid & boundary)
    assert np.array_equal(normal | invalid | boundary, valid)
    print("[PASS] Y12-D offline protocol tests; no model/GPU execution was attempted.")


if __name__ == "__main__":
    run()
