"""Offline protocol tests; imports no Torch/CUDA or private dataset."""
from y12e_common import *
from pathlib import Path
import json, tempfile, ast


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


def verify_native_keyframe_readout_contract():
    """Check exact pinned decoder and Y05 BCHW ray-state source contracts."""
    here = Path(__file__).resolve().parent
    repo_root = here.parents[2]
    decoder_path = here / "reference" / "decoder_velo_pinned.py"
    if not decoder_path.is_file():
        decoder_path = repo_root / "velodepth" / "models" / "decoder_velo.py"
    require(decoder_path.is_file() and sha(decoder_path) == PINNED_DECODER_SHA256,
            "Pinned decoder source missing or changed")
    tree = ast.parse(decoder_path.read_text(encoding="utf-8"))
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    require("RadialModule" in classes and "Decoder" in classes,
            "Pinned decoder source lacks RadialModule/Decoder")

    def method(class_name, method_name):
        for node in classes[class_name].body:
            if isinstance(node, ast.FunctionDef) and node.name == method_name:
                return node
        return None

    radial_keyframe = method("RadialModule", "forward_keyframe")
    decoder_keyframe = method("Decoder", "forward_keyframe")
    require(radial_keyframe is not None and decoder_keyframe is not None,
            "Pinned decoder lacks native keyframe methods")
    radial_calls = {node.func.attr for node in ast.walk(radial_keyframe)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    require({"embed_rays", "process", "decode"}.issubset(radial_calls),
            "RadialModule.forward_keyframe lacks embed/process/decode path")
    decoder_calls = {node.func.attr for node in ast.walk(decoder_keyframe)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    require("forward_keyframe" in decoder_calls and "exp" in decoder_calls,
            "Decoder.forward_keyframe wrapper no longer delegates/transforms native geometry")

    eval_source = (here / "y12e_eval.py").read_text(encoding="utf-8")
    require("radial.forward_keyframe(\n            adapted, rays_hr" in eval_source,
            "D2 does not call native RadialModule.forward_keyframe with flattened rays")
    require("rays_base.permute(0, 2, 3, 1).reshape(" in eval_source,
            "D2 does not convert persistent BCHW rays to Bx(HW)x3 in channel-correct order")
    require("native_persistent_readout(engine, donor_features)" not in eval_source,
            "D2 must not use fast propagation readout for keyframe donor geometry")
    require("logconf_teacher.clip(min=-8.0, max=10.0)" in eval_source,
            "D2 does not use native keyframe confidence transform")
    require('"E1_FEATURE_ONLY": state_e1' in eval_source and '"E2_GEOMETRY_ONLY": state_e2' in eval_source,
            "Y12-E component intervention states are missing")
    require('common_h0 = prediction_to_numpy(np, pred_cstar)' in eval_source,
            "Y12-E h0 output is not fixed to the common C* prediction")
    require('state_e2 = (donor_radius.detach().clone(), state_cstar[1].detach().clone()' in eval_source,
            "Geometry-only arm does not hold C* confidence/features fixed")

    exp1_path = here / "reference" / "gt4a2_exp1_train_eval_pinned.py"
    if not exp1_path.is_file():
        exp1_path = repo_root / "experiments" / "p1" / "gt4a2_exp1_train_eval.py"
    require(exp1_path.is_file() and sha(exp1_path) == PINNED_EXP1_SHA256,
            "Pinned exp1 keyframe_state source missing or changed")
    exp1_source = exp1_path.read_text(encoding="utf-8")
    require('rays_map = rays.reshape(1, *shape["network_shape"], 3).permute(0, 3, 1, 2)' in exp1_source,
            "Pinned Y05 keyframe_state no longer returns BCHW ray maps")

    import numpy as np
    ray_map = np.arange(1 * 3 * 2 * 4, dtype=np.float32).reshape(1, 3, 2, 4)
    ray_flat = np.transpose(ray_map, (0, 2, 3, 1)).reshape(1, -1, 3)
    assert np.array_equal(ray_flat[0, 0], ray_map[0, :, 0, 0])
    assert np.array_equal(ray_flat[0, 7], ray_map[0, :, 1, 3])


def run():
    verify_native_keyframe_readout_contract()
    # Pinned Y05 Engine.load_frame exposes normalized BCHW as frame["img"].
    sentinel = object()
    assert normalized_image_from_frame({"img": sentinel}) is sentinel
    try:
        normalized_image_from_frame({"image": sentinel, "gt_np": None})
    except RuntimeError as exc:
        assert "frame['img']" in str(exc)
    else:
        raise AssertionError("Y12-E accepted unsupported frame['image'] alias")

    # GPU read_depth returns a Torch tensor; verify the conversion chain
    # without importing Torch/CUDA in the offline protocol test.
    import numpy as np
    class FakeDeviceTensor:
        def __init__(self, value):
            self.value = value
            self.calls = []
        def detach(self):
            self.calls.append("detach")
            return self
        def float(self):
            self.calls.append("float")
            return self
        def cpu(self):
            self.calls.append("cpu")
            return self
        def numpy(self):
            self.calls.append("numpy")
            return self.value
    source = np.array([[1.25, 2.5]], dtype=np.float16)
    fake = FakeDeviceTensor(source)
    converted = prediction_to_numpy(np, fake)
    assert fake.calls == ["detach", "float", "cpu", "numpy"]
    assert converted.dtype == np.float32
    assert np.array_equal(converted, source.astype(np.float32))
    already_numpy = prediction_to_numpy(np, source)
    assert already_numpy.dtype == np.float32
    assert np.array_equal(already_numpy, converted)

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
    assert settings["branches"] == ["E0_CSTAR", "E1_FEATURE_ONLY", "E2_GEOMETRY_ONLY", "E3_FULL_TEACHER"]
    # Y12-E is component attribution only; it cannot grant method/Y11 pass.
    assert len(BRANCHES) == 4
    assert BRANCHES == ("E0_CSTAR", "E1_FEATURE_ONLY", "E2_GEOMETRY_ONLY", "E3_FULL_TEACHER")
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
    print("[PASS] Y12-E offline protocol tests; no model/GPU execution was attempted.")


if __name__ == "__main__":
    run()
