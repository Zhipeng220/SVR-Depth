import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m05_summary_test", ROOT / "experiments/p1/cvpr_m05_formal.py")
m05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m05)


class M05SummaryRegression(unittest.TestCase):
    def test_h4_columns_and_parent_strata(self):
        row = {"seed": 20261103, "sequence": "crowd2", "parent": "A_native", "age": 1,
               "H4_T_minus_O_AbsRel": "0.002", "H4_F_minus_A_AbsRel": "0.002",
               "H4_Pnative_minus_T_AbsRel": "0.1", "H4_Ronly_minus_T_AbsRel": "0.01",
               "H4_Conly_minus_T_AbsRel": "0", "T_minus_O_H1_H4_window_AbsRel": "0.003"}
        other = dict(row, parent="P_maintained", H4_T_minus_O_AbsRel="0.7")
        out = m05.summarize_effects([row, other])
        self.assertEqual(sum(s["n_H4"] for s in out["strata"]), 2)
        self.assertAlmostEqual(out["per_seed_primary"][0]["A_native_primary_sequence_equal_mean_T_minus_O_H4"], 0.002)
        self.assertAlmostEqual(out["strata"][0]["mean_Pnative_minus_T_H4_AbsRel"], 0.1)

    def test_bad_header_fails_instead_of_silent_na(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            m05.summarize_effects([{"T_minus_O_H4_AbsRel": "0.5"}])


if __name__ == "__main__":
    unittest.main()
