import math
import unittest
from r3_gate_rules import POLICIES, DISABLED_POLICIES, decide


class GateRuleTests(unittest.TestCase):
    def test_active_policy_set_excludes_unvalidated_confidence_gates(self):
        self.assertEqual(POLICIES, ("AlwaysCommit", "NeverCommit", "AgeGate", "ResidualGate"))
        self.assertIn("ConfidenceGate", DISABLED_POLICIES)
        self.assertIn("CombinedGate", DISABLED_POLICIES)

    def test_endpoints(self):
        self.assertTrue(decide("AlwaysCommit", {}))
        self.assertFalse(decide("NeverCommit", {}))

    def test_age_threshold_and_fail_closed(self):
        self.assertTrue(decide("AgeGate", {"distance_kf": 16}, age_threshold=16))
        self.assertFalse(decide("AgeGate", {"distance_kf": 15}, age_threshold=16))
        self.assertFalse(decide("AgeGate", {"distance_kf": float("nan")}, age_threshold=16))
        self.assertFalse(decide("AgeGate", {}, age_threshold=16))

    def test_residual_gate_threshold_boundary(self):
        self.assertTrue(decide("ResidualGate", {"depth_disagreement_rel_mean": 0.10}, residual_threshold=0.10))
        self.assertFalse(decide("ResidualGate", {"depth_disagreement_rel_mean": 0.10001}, residual_threshold=0.10))
        self.assertFalse(decide("ResidualGate", {}))
        self.assertFalse(decide("ResidualGate", {"depth_disagreement_rel_mean": float("nan")}))

    def test_residual_threshold_sensitivity_has_expected_monotonicity_for_fixed_feature(self):
        feature = {"depth_disagreement_rel_mean": 0.057}
        thresholds = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
        decisions = [decide("ResidualGate", feature, residual_threshold=t) for t in thresholds]
        self.assertEqual(decisions, [False, False, False, True, True, True])

    def test_raw_confidence_diagnostics_are_not_used_by_active_rules(self):
        f1 = {"depth_disagreement_rel_mean": 0.05,
              "confidence_prop_mean_raw_diagnostic_only": 1000.0,
              "confidence_candidate_mean_raw_diagnostic_only": 0.0001}
        f2 = {"depth_disagreement_rel_mean": 0.05,
              "confidence_prop_mean_raw_diagnostic_only": 0.0001,
              "confidence_candidate_mean_raw_diagnostic_only": 1000.0}
        self.assertEqual(decide("ResidualGate", f1), decide("ResidualGate", f2))

    def test_confidence_gates_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "not calibrated/comparable"):
            decide("ConfidenceGate", {})
        with self.assertRaisesRegex(ValueError, "non-comparable"):
            decide("CombinedGate", {})

    def test_unknown_policy_raises(self):
        with self.assertRaises(ValueError):
            decide("future_gt_oracle", {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
