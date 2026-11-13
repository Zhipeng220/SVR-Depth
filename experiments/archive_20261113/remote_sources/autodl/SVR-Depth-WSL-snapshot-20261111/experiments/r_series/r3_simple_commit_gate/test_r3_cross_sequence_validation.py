import unittest
import pandas as pd

from r3_cross_sequence_validation import select_threshold_for_fold, attach_action_patterns


class LeaveOneSequenceOutTests(unittest.TestCase):
    def setUp(self):
        self.thresholds = [0.03, 0.05]
        self.baselines = {
            "crowd2": pd.DataFrame([
                {"strategy": "NeverCommit", "mean_absrel": 0.20, "mean_rmse": 0.50, "mean_delta1": 0.70},
                {"strategy": "AlwaysCommit", "mean_absrel": 0.15, "mean_rmse": 0.40, "mean_delta1": 0.90},
                {"strategy": "AgeGate", "mean_absrel": 0.14, "mean_rmse": 0.42, "mean_delta1": 0.88},
                {"strategy": "ResidualGate", "mean_absrel": 0.15, "mean_rmse": 0.41, "mean_delta1": 0.89},
            ]),
            "balloon2": pd.DataFrame([
                {"strategy": "NeverCommit", "mean_absrel": 0.08, "mean_rmse": 0.28, "mean_delta1": 0.96},
                {"strategy": "AlwaysCommit", "mean_absrel": 0.07, "mean_rmse": 0.27, "mean_delta1": 0.97},
                {"strategy": "AgeGate", "mean_absrel": 0.075, "mean_rmse": 0.26, "mean_delta1": 0.965},
                {"strategy": "ResidualGate", "mean_absrel": 0.07, "mean_rmse": 0.27, "mean_delta1": 0.97},
            ]),
            "crowd3": pd.DataFrame([
                {"strategy": "NeverCommit", "mean_absrel": 0.14, "mean_rmse": 0.42, "mean_delta1": 0.95},
                {"strategy": "AlwaysCommit", "mean_absrel": 0.12, "mean_rmse": 0.37, "mean_delta1": 0.98},
                {"strategy": "AgeGate", "mean_absrel": 0.13, "mean_rmse": 0.39, "mean_delta1": 0.97},
                {"strategy": "ResidualGate", "mean_absrel": 0.12, "mean_rmse": 0.37, "mean_delta1": 0.98},
            ]),
        }
        rows = []
        vals = {
            "crowd2": {0.03: (0.20, 0.50, 0.70), 0.05: (0.16, 0.45, 0.80)},
            "balloon2": {0.03: (0.08, 0.28, 0.96), 0.05: (0.07, 0.27, 0.97)},
            "crowd3": {0.03: (0.14, 0.42, 0.95), 0.05: (0.12, 0.37, 0.98)},
        }
        for seq, d in vals.items():
            for tau, (a, r, delta) in d.items():
                rows.append({"sequence": seq, "residual_threshold": tau, "mean_absrel": a,
                             "mean_rmse": r, "mean_delta1": delta})
        self.sensitivity = pd.DataFrame(rows)

    def test_action_patterns_are_derived_from_action_csv(self):
        action_rows = []
        patterns = {"crowd2": {0.03: "DDD", 0.05: "CDD"},
                    "balloon2": {0.03: "CDD", 0.05: "CCC"},
                    "crowd3": {0.03: "CCD", 0.05: "CCC"}}
        for seq, by_tau in patterns.items():
            for tau, pattern in by_tau.items():
                for anchor, symbol in zip([9, 37, 65], pattern):
                    action_rows.append({"sequence": seq, "residual_threshold": tau,
                                        "anchor_frame_zero_based": anchor,
                                        "selected_action": "COMMIT" if symbol == "C" else "DISCARD"})
        merged = attach_action_patterns(self.sensitivity, pd.DataFrame(action_rows))
        self.assertEqual(len(merged), len(self.sensitivity))
        self.assertEqual(merged[(merged.sequence == "crowd2") & (merged.residual_threshold == 0.03)].iloc[0]["pattern"], "DDD")
        self.assertEqual(int(merged[(merged.sequence == "balloon2") & (merged.residual_threshold == 0.05)].iloc[0]["commits"]), 3)

    def test_heldout_sequence_does_not_change_threshold_choice(self):
        chosen_a, _ = select_threshold_for_fold(self.sensitivity.assign(pattern="CDD", commits=1), self.baselines, "crowd3", self.thresholds)
        changed = self.sensitivity.copy()
        changed.loc[changed["sequence"] == "crowd3", ["mean_absrel", "mean_rmse", "mean_delta1"]] = [1e-6, 1e-6, 0.999999]
        chosen_b, _ = select_threshold_for_fold(changed.assign(pattern="CDD", commits=1), self.baselines, "crowd3", self.thresholds)
        self.assertEqual(chosen_a, chosen_b)

    def test_fold_training_sequences_exclude_heldout(self):
        _, rows = select_threshold_for_fold(self.sensitivity.assign(pattern="CDD", commits=1), self.baselines, "balloon2", self.thresholds)
        self.assertTrue(all("balloon2" not in row["training_sequences"] for row in rows))
        self.assertTrue(all(row["selection_uses_heldout_metrics"] is False for row in rows))

    def test_selects_best_train_absrel_ratio(self):
        chosen, rows = select_threshold_for_fold(self.sensitivity.assign(pattern="CDD", commits=1), self.baselines, "crowd2", self.thresholds)
        self.assertEqual(chosen, 0.05)
        self.assertEqual(sum(bool(r["selected_for_fold"]) for r in rows), 1)


if __name__ == "__main__":
    unittest.main()
