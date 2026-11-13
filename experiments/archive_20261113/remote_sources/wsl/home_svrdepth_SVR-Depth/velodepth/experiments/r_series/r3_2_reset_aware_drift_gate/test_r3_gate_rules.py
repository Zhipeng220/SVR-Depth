import math
import unittest
from r3_gate_rules import POLICIES, DISABLED_POLICIES, decide


class GateRuleTests(unittest.TestCase):
    def test_policy_set_includes_original_and_new_rules(self):
        self.assertEqual(POLICIES, (
            "AlwaysCommit", "NeverCommit", "AgeGate", "ResidualGate", "DriftGate", "AgeDriftGate"
        ))
        self.assertIn("ConfidenceGate", DISABLED_POLICIES)
        self.assertIn("CombinedGate", DISABLED_POLICIES)

    def test_endpoints(self):
        self.assertTrue(decide("AlwaysCommit", {}))
        self.assertFalse(decide("NeverCommit", {}))

    def test_age_gate_boundary_and_fail_closed(self):
        self.assertTrue(decide("AgeGate", {"distance_kf": 16}, age_threshold=16))
        self.assertFalse(decide("AgeGate", {"distance_kf": 15}, age_threshold=16))
        self.assertFalse(decide("AgeGate", {"distance_kf": float("nan")}, age_threshold=16))
        self.assertFalse(decide("AgeGate", {}))

    def test_original_residual_gate_semantics_are_preserved(self):
        self.assertTrue(decide("ResidualGate", {"depth_disagreement_rel_mean": 0.04}, residual_threshold=0.04))
        self.assertFalse(decide("ResidualGate", {"depth_disagreement_rel_mean": 0.04001}, residual_threshold=0.04))
        self.assertFalse(decide("ResidualGate", {}))

    def test_drift_gate_reversed_direction(self):
        self.assertFalse(decide("DriftGate", {"depth_disagreement_rel_mean": 0.03999}, residual_threshold=0.04))
        self.assertTrue(decide("DriftGate", {"depth_disagreement_rel_mean": 0.04}, residual_threshold=0.04))
        self.assertTrue(decide("DriftGate", {"depth_disagreement_rel_mean": 0.10}, residual_threshold=0.04))
        self.assertFalse(decide("DriftGate", {"depth_disagreement_rel_mean": float("nan")}, residual_threshold=0.04))

    def test_age_drift_requires_both_conditions(self):
        self.assertFalse(decide("AgeDriftGate", {"distance_kf": 15, "depth_disagreement_rel_mean": 0.20}, 16, 0.04))
        self.assertFalse(decide("AgeDriftGate", {"distance_kf": 20, "depth_disagreement_rel_mean": 0.03}, 16, 0.04))
        self.assertTrue(decide("AgeDriftGate", {"distance_kf": 16, "depth_disagreement_rel_mean": 0.04}, 16, 0.04))
        self.assertFalse(decide("AgeDriftGate", {"distance_kf": 20}, 16, 0.04))

    def test_raw_confidence_does_not_affect_rules(self):
        f1 = {"distance_kf": 20, "depth_disagreement_rel_mean": 0.05,
              "confidence_prop_mean_raw_diagnostic_only": 1000.0}
        f2 = {"distance_kf": 20, "depth_disagreement_rel_mean": 0.05,
              "confidence_prop_mean_raw_diagnostic_only": 0.0001}
        self.assertEqual(decide("DriftGate", f1), decide("DriftGate", f2))
        self.assertEqual(decide("AgeDriftGate", f1), decide("AgeDriftGate", f2))

    def test_disabled_and_unknown_policies_raise(self):
        with self.assertRaisesRegex(ValueError, "not calibrated/comparable"):
            decide("ConfidenceGate", {})
        with self.assertRaisesRegex(ValueError, "non-comparable"):
            decide("CombinedGate", {})
        with self.assertRaises(ValueError):
            decide("future_gt_oracle", {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
