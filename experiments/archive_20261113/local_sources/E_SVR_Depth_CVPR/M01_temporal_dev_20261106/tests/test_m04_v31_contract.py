import importlib.util
from pathlib import Path
import unittest
import tempfile

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "experiments/p1/cvpr_m04_v31.py"
SPEC = importlib.util.spec_from_file_location("cvpr_m04_v31", SCRIPT)
M04 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M04)


class TestM04V31E4CLock(unittest.TestCase):
    def test_only_e4c_checkpoint_is_accepted(self):
        self.assertEqual(M04.EXPECTED_E4C_CHECKPOINT_SHA256,
            "0447fbcc25d1ea93d5a80e7f8a7a3b38ac911a8465951009da53988e712754c0")
        self.assertEqual(M04.EXPECTED_E4C_STEP, 700)
        self.assertEqual(M04.EXPECTED_E4C_SEED, 20261103)
        self.assertEqual(M04.EXPECTED_E4C_TRAIN_STEPS, 1000)
        self.assertEqual(M04.EXPECTED_E4C_PARAMS, 170496)

    def test_four_closed_loop_branches_are_fixed(self):
        self.assertEqual(M04.BRANCHES, ("A_off", "O_same_head", "F_only", "P_full"))
        self.assertEqual(M04.K, 16)
        self.assertEqual(M04.GAIN, 1.0)
        self.assertEqual(M04.FREQUENCY, 1)

    def test_unified_repository_runtime_paths(self):
        self.assertEqual(M04.ROOT, REPO / "outputs/legacy_runtime")
        self.assertEqual(M04.E4C_HELPERS, REPO / "experiments/crosssource_v2/legacy_helpers")
        self.assertTrue(M04.ROOT.is_relative_to(REPO))

    def test_e4c_schema_not_legacy_y04_schema(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("payload['head_state']", source)
        self.assertNotIn("repair_head_state_dict", source)
        self.assertNotIn("--checkpoint-manifest", source)
        self.assertIn("E4-C Best checkpoint only", source)

    def test_wrong_checkpoint_hash_is_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrong = Path(tmp) / "not_e4c.pt"
            wrong.write_bytes(b"not the locked E4-C checkpoint")
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                M04.parse_e4c_checkpoint(wrong, with_torch=False)

    def test_e4c_default_path(self):
        self.assertEqual(M04.E4C_DEFAULT_CHECKPOINT,
            REPO / "outputs/crosssource_v2_E4_20261104_C_s20261103_1000step/best_checkpoint.pt")

    def test_launcher_and_protocol_exist(self):
        self.assertTrue((REPO / "scripts/run_m04_v31.sh").is_file())
        self.assertTrue((REPO / "configs/eval/m04_v31_e4c.json").is_file())
        self.assertTrue((REPO / "docs/M04_V3.1.md").is_file())


if __name__ == "__main__":
    unittest.main()
