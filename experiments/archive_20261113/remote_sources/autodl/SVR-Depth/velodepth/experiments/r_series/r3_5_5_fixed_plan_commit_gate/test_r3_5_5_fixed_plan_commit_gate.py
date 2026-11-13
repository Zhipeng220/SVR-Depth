import unittest
import math
import json
import tempfile
from pathlib import Path
import pandas as pd
from r3_5_5_fixed_plan_commit_gate import choose_threshold, evaluate_threshold, calibrate_loso, compare_closed_loop, FEATURE, GAIN

class R355Tests(unittest.TestCase):
    def frame(self, xs, gs):
        return pd.DataFrame({FEATURE: xs, GAIN: gs})

    def test_threshold_discards_low_feature_negative_gain(self):
        train = self.frame([0.1, 0.2, 0.8, 0.9], [-0.2, -0.1, 0.1, 0.2])
        result = choose_threshold(train)
        self.assertTrue(result["active"])
        test = self.frame([0.15, 0.85], [-0.3, 0.2])
        scored = evaluate_threshold(test, result["threshold"])
        self.assertGreater(scored["delta_utility"], 0)

    def test_no_gain_from_gate_prefers_always_commit(self):
        train = self.frame([0.1, 0.2, 0.8], [0.1, 0.2, 0.3])
        result = choose_threshold(train)
        self.assertFalse(result["active"])
        self.assertIsNone(result["threshold"])

    def test_no_valid_rows_is_safe_noop(self):
        train = self.frame([float("nan")], [float("nan")])
        result = choose_threshold(train)
        self.assertFalse(result["active"])
        self.assertIsNone(result["threshold"])


    def test_loso_calibration_writes_fold_outputs_and_uses_fixed_sequence_splits(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "collect"
            out = Path(td) / "calibration"
            for seq, xs, gs in [
                ("crowd2", [0.1, 0.2, 0.8], [-0.2, -0.1, 0.2]),
                ("balloon2", [0.1, 0.3, 0.9], [-0.1, -0.2, 0.3]),
                ("crowd3", [0.15, 0.25, 0.85], [-0.1, -0.1, 0.2]),
            ]:
                sd = root / seq
                sd.mkdir(parents=True)
                pd.DataFrame({
                    "counterfactual_status": ["ok"] * 3,
                    "anchor_frame_zero_based": [5, 17, 29],
                    FEATURE: xs,
                    GAIN: gs,
                }).to_csv(sd / "r355_action_geometry_gain.csv", index=False)
            result = calibrate_loso(root, out)
            self.assertEqual(result["valid_action_rows"], 9)
            self.assertTrue((out / "r355_loso_thresholds_and_utility.csv").exists())
            self.assertTrue((out / "r355_gate_decision.json").exists())
            self.assertEqual(set(result["thresholds_by_heldout_sequence"]), {"crowd2", "balloon2", "crowd3"})


    def test_closed_loop_compare_includes_three_controls(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            baseline, never, gate = root / "baseline", root / "never", root / "gate"
            for seq, b_abs, n_abs, g_abs in [
                ("crowd2", 0.10, 0.12, 0.09),
                ("balloon2", 0.20, 0.22, 0.19),
                ("crowd3", 0.15, 0.17, 0.151),
            ]:
                for policy_root, val in [(baseline, b_abs), (never, n_abs), (gate, g_abs)]:
                    d = policy_root / seq
                    d.mkdir(parents=True)
                    pd.DataFrame({
                        "absrel": [val] * 5, "rmse": [val + 0.1] * 5,
                        "delta1": [0.9] * 5, "geom_commit": [1, 1, 1, 1, 1],
                        "diagnostic_candidate_computed": [0, 1, 0, 1, 0],
                        "policy_action_anchor": [0, 1, 0, 1, 0],
                    }).to_csv(d / "r355_online_frame_metrics.csv", index=False)
            cal = root / "decision.json"
            cal.write_text(json.dumps({
                "offline_gate_pass": True,
                "thresholds_by_heldout_sequence": {}
            }), encoding="utf-8")
            out = root / "comparison"
            result = compare_closed_loop(baseline, never, gate, cal, out)
            self.assertTrue(result["closed_loop_gate_pass"])
            self.assertTrue(result["candidate_compute_counts_match"])
            self.assertTrue((out / "r355_closed_loop_comparison.csv").exists())

    def test_threshold_never_uses_test_gain_for_selection(self):
        train = self.frame([0.1, 0.2, 0.8, 0.9], [-0.2, -0.1, 0.1, 0.2])
        result = choose_threshold(train)
        self.assertAlmostEqual(result["threshold"], 0.5)
        # Changing only test labels cannot alter a previously selected threshold.
        _ = evaluate_threshold(self.frame([0.1, 0.9], [10.0, -10.0]), result["threshold"])
        result2 = choose_threshold(train.copy())
        self.assertEqual(result["threshold"], result2["threshold"])

if __name__ == "__main__":
    unittest.main()
