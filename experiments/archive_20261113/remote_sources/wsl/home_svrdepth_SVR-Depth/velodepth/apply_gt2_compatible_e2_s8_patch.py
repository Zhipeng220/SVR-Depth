#!/usr/bin/env python3
from pathlib import Path
import hashlib
import os

ROOT = Path.cwd()
DEC = ROOT / "velodepth/models/decoder_velo.py"
HEAD = ROOT / "velodepth/models/persistent_feature_repair.py"
BACKUP = ROOT / "outputs/gt2_validation/decoder_velo.py.pre_gt2_patch.bak"
MARKER = "E2_S8_CURRENT_PATCH"

def rep(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: anchor count={n}, expected 1; no source changed.")
    return text.replace(old, new, 1)

def main():
    if not DEC.is_file() or not HEAD.is_file():
        raise SystemExit("Missing decoder_velo.py or persistent_feature_repair.py.")
    if "class PersistentFeatureRepairHead" not in HEAD.read_text(encoding="utf-8"):
        raise SystemExit("Repair-head class missing; stop.")
    if not (ROOT / "outputs/gt2_validation").is_dir():
        raise SystemExit("Run mkdir -p outputs/gt2_validation first.")
    if BACKUP.exists():
        raise SystemExit(f"Backup exists: {BACKUP}; refusing to overwrite.")
    raw = DEC.read_bytes()
    d = raw.decode("utf-8")
    if MARKER in d:
        raise SystemExit("Patch marker already exists; refusing to apply twice.")

    d = rep(d, "import torch\n",
        "import torch\nfrom velodepth.models.persistent_feature_repair import PersistentFeatureRepairHead  # E2_S8_CURRENT_PATCH\n",
        "import")

    d = rep(d, """\
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.build(config)
        self.apply(self._init_weights)
""", """\
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.build(config)
        self.apply(self._init_weights)
        # E2_S8_CURRENT_PATCH: create after generic init to preserve zero residual.
        repair_cfg = config["model"].get("persistent_feature_repair", {})
        self.feature_repair_mode = "off"
        self.persistent_feature_repair = None
        if repair_cfg.get("enabled", False):
            mode = repair_cfg.get("mode", "post_readout")
            if mode not in {"pre_readout", "post_readout"}:
                raise ValueError(
                    "persistent_feature_repair.mode must be pre_readout or post_readout"
                )
            self.feature_repair_mode = mode
            self.persistent_feature_repair = PersistentFeatureRepairHead(
                channels=int(self.radial_module.hidden_dim),
                bottleneck=int(repair_cfg.get("bottleneck", 32)),
                levels=4,
            )
""", "Decoder constructor")

    d = rep(d, """\
    def forward_fast(
        self, lateral_features_teacher, features_residual, flow_bwd, flow_features
    ):
        features_propagated = []
""", """\
    def forward_fast(
        self, lateral_features_teacher, features_residual, flow_bwd, flow_features,
        feature_repair=None, feature_repair_mode="off",
    ):
        features_propagated = []
""", "RadialModule.forward_fast signature")

    d = rep(d, """\
            features_propagated.append(corrected_features)

        lateral_features = [features_propagated[0]] + [
""", """\
            features_propagated.append(corrected_features)

        # C: repair affects current readout and the next persistent state.
        if feature_repair is not None and feature_repair_mode == "pre_readout":
            features_propagated = feature_repair(features_propagated)

        lateral_features = [features_propagated[0]] + [
""", "C insertion")

    d = rep(d, """\
        logconf_student = logdepth_student

        return logdepth_student, logconf_student, features_propagated
""", """\
        logconf_student = logdepth_student

        # D: preserve current readout; repair only state carried to the next frame.
        if feature_repair is not None and feature_repair_mode == "post_readout":
            features_propagated = feature_repair(features_propagated)

        return logdepth_student, logconf_student, features_propagated
""", "D insertion")

    d = rep(d, """\
        lograd_student, logconf_student, lateral_features_student = (
            self.radial_module.forward_fast(
                features_teacher,
                features_residual,
                flow_bwd=inputs["flow_bwd_refined"],
                flow_features=inputs["flow_features"],
            )
        )
""", """\
        # E2_S8_CURRENT_PATCH: route the head into C/D state semantics.
        lograd_student, logconf_student, lateral_features_student = (
            self.radial_module.forward_fast(
                features_teacher,
                features_residual,
                flow_bwd=inputs["flow_bwd_refined"],
                flow_features=inputs["flow_features"],
                feature_repair=self.persistent_feature_repair,
                feature_repair_mode=self.feature_repair_mode,
            )
        )
""", "Decoder forwarding")

    compile(d, str(DEC), "exec")  # syntax-check before writing
    if d.count(MARKER) < 3:
        raise RuntimeError("Unexpected marker count; no source changed.")

    tmp = DEC.with_name("decoder_velo.py.e2s8_tmp")
    if tmp.exists():
        raise SystemExit(f"Temporary file exists: {tmp}; refusing to write.")
    BACKUP.write_bytes(raw)
    try:
        tmp.write_text(d, encoding="utf-8")
        os.replace(tmp, DEC)
    except Exception:
        DEC.write_bytes(BACKUP.read_bytes())
        if tmp.exists():
            tmp.unlink()
        raise

    print("[PASS] Compatible E2-S8 patch applied.")
    print("[BACKUP]", BACKUP)
    print("[SHA256 before]", hashlib.sha256(raw).hexdigest())
    print("[SHA256 after ]", hashlib.sha256(DEC.read_bytes()).hexdigest())
    print("[MARKER COUNT]", DEC.read_text(encoding="utf-8").count(MARKER))
    print("[MODIFIED]", DEC)
    print("[UNCHANGED] velodepth/models/velodepth.py")

if __name__ == "__main__":
    main()
