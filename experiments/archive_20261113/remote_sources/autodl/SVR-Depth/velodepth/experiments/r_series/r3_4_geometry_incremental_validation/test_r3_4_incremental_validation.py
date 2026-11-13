import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from r3_4_incremental_validation import fit_predict_ridge, score_predictions, TARGET

class R34Tests(unittest.TestCase):
    def test_ridge_prediction_uses_train_statistics_and_returns_shape(self):
        train = pd.DataFrame({"distance_kf":[1.,2.,3.,4.], "gain_absrel_discounted_per_valid_frame":[-0.1,-0.05,0.05,0.1]})
        test = pd.DataFrame({"distance_kf":[5.,6.]})
        pred = fit_predict_ridge(train, test, ["distance_kf"])
        self.assertEqual(pred.shape, (2,))
        self.assertTrue(np.all(np.isfinite(pred)))

    def test_mean_baseline_is_train_mean(self):
        train = pd.DataFrame({"gain_absrel_discounted_per_valid_frame":[-0.1,0.1]})
        test = pd.DataFrame(index=[0,1,2])
        pred = fit_predict_ridge(train, test, [])
        self.assertTrue(np.allclose(pred, 0.0))

    def test_missing_feature_values_are_imputed(self):
        train = pd.DataFrame({"x":[1.,np.nan,3.], "gain_absrel_discounted_per_valid_frame":[-0.1,0.0,0.1]})
        test = pd.DataFrame({"x":[np.nan,4.]})
        pred = fit_predict_ridge(train, test, ["x"])
        self.assertEqual(pred.shape, (2,))
        self.assertTrue(np.all(np.isfinite(pred)))

    def test_negative_gain_recall_is_defined(self):
        d = pd.DataFrame({
            TARGET: [-0.2, -0.1, 0.1, 0.2],
            "distance_kf":[2.,4.,18.,20.],
            "is_policy_action_anchor":[0,1,0,1],
            "pred":[-0.1,0.1,0.2,-0.1]
        })
        r = score_predictions(d, "pred", "test")
        self.assertAlmostEqual(r["negative_gain_recall_discarded"], 0.5)
        self.assertEqual(r["n"], 4)

if __name__ == "__main__":
    unittest.main()
