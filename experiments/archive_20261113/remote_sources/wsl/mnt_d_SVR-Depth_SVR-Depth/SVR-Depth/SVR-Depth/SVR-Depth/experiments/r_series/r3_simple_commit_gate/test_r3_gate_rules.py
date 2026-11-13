import math
import unittest
from r3_gate_rules import decide, POLICIES


class GateRuleTests(unittest.TestCase):
    def test_expected_policy_set(self):
        self.assertEqual(len(POLICIES), 6)
        self.assertEqual(len(set(POLICIES)), 6)

    def test_endpoints(self):
        self.assertTrue(decide("AlwaysCommit", {}))
        self.assertFalse(decide("NeverCommit", {}))

    def test_age_threshold_and_fail_closed(self):
        self.assertTrue(decide("AgeGate", {"distance_kf": 16}, age_threshold=16))
        self.assertFalse(decide("AgeGate", {"distance_kf": 15}, age_threshold=16))
        self.assertFalse(decide("AgeGate", {"distance_kf": float("nan")}, age_threshold=16))
        self.assertFalse(decide("AgeGate", {}, age_threshold=16))

    def test_confidence_gate(self):
        self.assertTrue(decide("ConfidenceGate", {"confidence_delta_candidate_minus_prop": 0.0}))
        self.assertTrue(decide("ConfidenceGate", {"confidence_delta_candidate_minus_prop": 0.1}))
        self.assertFalse(decide("ConfidenceGate", {"confidence_delta_candidate_minus_prop": -0.1}))
        self.assertFalse(decide("ConfidenceGate", {"confidence_delta_candidate_minus_prop": float("nan")}))

    def test_residual_gate(self):
        self.assertTrue(decide("ResidualGate", {"depth_disagreement_rel_mean": 0.10}, residual_threshold=0.10))
        self.assertFalse(decide("ResidualGate", {"depth_disagreement_rel_mean": 0.10001}, residual_threshold=0.10))
        self.assertFalse(decide("ResidualGate", {}))

    def test_combined_gate_requires_both_signals(self):
        f = {"confidence_delta_candidate_minus_prop": 0.01, "depth_disagreement_rel_mean": 0.05}
        self.assertTrue(decide("CombinedGate", f, residual_threshold=0.10))
        self.assertFalse(decide("CombinedGate", {**f, "confidence_delta_candidate_minus_prop": -0.01}, residual_threshold=0.10))
        self.assertFalse(decide("CombinedGate", {**f, "depth_disagreement_rel_mean": 0.11}, residual_threshold=0.10))

    def test_unknown_policy_raises(self):
        with self.assertRaises(ValueError):
            decide("future_gt_oracle", {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
