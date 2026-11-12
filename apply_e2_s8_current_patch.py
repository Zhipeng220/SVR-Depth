#!/usr/bin/env python3
"""Fail-closed E2-S8 patch for the refactored VeloDepth source path.

Expected current source contract:
  RadialModule.propagate_features(...)
  RadialModule.readout_fast(...)
  RadialModule.forward_fast(...) calling those two methods
  Decoder.forward_fast(...) calling RadialModule.forward_fast(...)

Run from the repository root. This script does not use git. It writes source
backups before changing files and refuses to patch if any anchor differs.
"""
from pathlib import Path
import hashlib
import sys

ROOT = Path.cwd()
DEC = ROOT / "velodepth/models/decoder_velo.py"
VEL = ROOT / "velodepth/models/velodepth.py"
HEAD = ROOT / "velodepth/models/persistent_feature_repair.py"
BACKUP = ROOT / "outputs/e2_s8_patch_backup_current"

MARKER = "E2_S8_CURRENT_PATCH"

def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one source anchor, found {count}. "
            "No source files have been changed. Do not force this patch; inspect this checkout."
        )
    return text.replace(old, new, 1)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    for p in (DEC, VEL, HEAD):
        if not p.is_file():
            raise SystemExit(f"Required file missing: {p}")

    d = DEC.read_text(encoding="utf-8")
    v = VEL.read_text(encoding="utf-8")

    if MARKER in d or MARKER in v:
        raise SystemExit("Patch marker already present; refusing to apply twice.")

    # 1) Import repair head. Insert after the exact top-level `import torch` line.
    d = replace_once(
        d,
        "import torch\n",
        "import torch\nfrom velodepth.models.persistent_feature_repair import PersistentFeatureRepairHead  # E2_S8_CURRENT_PATCH\n",
        "decoder import",
    )

    # 2) Create the optional module AFTER Decoder's self.apply(), so the
    # generic initializer cannot overwrite the head's zero-initialized final layer.
    d = replace_once(
        d,
        "    def __init__(\n"
        "        self,\n"
        "        config,\n"
        "    ):\n"
        "        super().__init__()\n"
        "        self.build(config)\n"
        "        self.apply(self._init_weights)\n",
        "    def __init__(\n"
        "        self,\n"
        "        config,\n"
        "    ):\n"
        "        super().__init__()\n"
        "        self.build(config)\n"
        "        self.apply(self._init_weights)\n"
        "        # E2_S8_CURRENT_PATCH: optional persistent-state repair.\n"
        "        repair_cfg = config[\"model\"].get(\"persistent_feature_repair\", {})\n"
        "        self.feature_repair_mode = \"off\"\n"
        "        self.persistent_feature_repair = None\n"
        "        if repair_cfg.get(\"enabled\", False):\n"
        "            mode = repair_cfg.get(\"mode\", \"post_readout\")\n"
        "            if mode not in {\"pre_readout\", \"post_readout\"}:\n"
        "                raise ValueError(\n"
        "                    \"persistent_feature_repair.mode must be pre_readout or post_readout\"\n"
        "                )\n"
        "            self.feature_repair_mode = mode\n"
        "            self.persistent_feature_repair = PersistentFeatureRepairHead(\n"
        "                channels=int(self.radial_module.hidden_dim),\n"
        "                bottleneck=int(repair_cfg.get(\"bottleneck\", 32)),\n"
        "                levels=4,\n"
        "            )\n",
        "Decoder constructor",
    )

    # 3) Implement C/D at the natural boundary between propagation and readout.
    d = replace_once(
        d,
        "    def forward_fast(\n"
        "        self, lateral_features_teacher, features_residual, flow_bwd, flow_features\n"
        "    ):\n"
        "        features_propagated = self.propagate_features(\n"
        "            lateral_features_teacher, features_residual, flow_bwd, flow_features\n"
        "        )\n"
        "        logdepth_student, logconf_student = self.readout_fast(features_propagated)\n"
        "        return logdepth_student, logconf_student, features_propagated\n",
        "    def forward_fast(\n"
        "        self, lateral_features_teacher, features_residual, flow_bwd, flow_features,\n"
        "        feature_repair=None, feature_repair_mode=\"off\",\n"
        "    ):\n"
        "        features_propagated = self.propagate_features(\n"
        "            lateral_features_teacher, features_residual, flow_bwd, flow_features\n"
        "        )\n"
        "        # C: repaired state participates in the current depth readout.\n"
        "        if feature_repair is not None and feature_repair_mode == \"pre_readout\":\n"
        "            features_propagated = feature_repair(features_propagated)\n"
        "        logdepth_student, logconf_student = self.readout_fast(features_propagated)\n"
        "        # D: preserve current readout; repair only the state returned to the next step.\n"
        "        if feature_repair is not None and feature_repair_mode == \"post_readout\":\n"
        "            features_propagated = feature_repair(features_propagated)\n"
        "        return logdepth_student, logconf_student, features_propagated\n",
        "RadialModule.forward_fast",
    )

    # 4) Route the configured head and mode through Decoder.forward_fast.
    d = replace_once(
        d,
        "        lograd_student, logconf_student, lateral_features_student = (\n"
        "            self.radial_module.forward_fast(\n"
        "                features_teacher,\n"
        "                features_residual,\n"
        "                flow_bwd=inputs[\"flow_bwd_refined\"],\n"
        "                flow_features=inputs[\"flow_features\"],\n"
        "            )\n"
        "        )\n",
        "        # E2_S8_CURRENT_PATCH: pass repair mode into the native fast path.\n"
        "        lograd_student, logconf_student, lateral_features_student = (\n"
        "            self.radial_module.forward_fast(\n"
        "                features_teacher,\n"
        "                features_residual,\n"
        "                flow_bwd=inputs[\"flow_bwd_refined\"],\n"
        "                flow_features=inputs[\"flow_features\"],\n"
        "                feature_repair=self.persistent_feature_repair,\n"
        "                feature_repair_mode=self.feature_repair_mode,\n"
        "            )\n"
        "        )\n",
        "Decoder.forward_fast call",
    )

    # The top-level optimizer calls get_params(self.pixel_decoder, ...), so the
    # registered head is a child of pixel_decoder and is included in its parameters.
    # Preserve both files before writing either one.
    if BACKUP.exists():
        raise SystemExit(f"Backup directory already exists: {BACKUP}; refusing to overwrite.")
    BACKUP.mkdir(parents=True, exist_ok=False)
    for p, content in ((DEC, DEC.read_text(encoding="utf-8")), (VEL, VEL.read_text(encoding="utf-8"))):
        backup = BACKUP / p.relative_to(ROOT)
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_text(content, encoding="utf-8")

    # Write only after every anchor was validated.
    DEC.write_text(d, encoding="utf-8")

    print("E2-S8 current-source patch applied.")
    print("Backup directory:", BACKUP)
    print("Modified:", DEC)
    print("Unmodified:", VEL)
    print("SHA256:", sha256(DEC))
    print("Note: run syntax + unit + real-clip smoke checks before training.")

if __name__ == "__main__":
    main()
