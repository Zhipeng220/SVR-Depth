import csv,json,unittest
from pathlib import Path
import numpy as np
from analyze_curves import curve_stats
ROOT=Path(__file__).resolve().parents[2]
class Tests(unittest.TestCase):
 def test_slope_curvature_and_oracle(self):
  a=[0,.25,.5,.75,1];r=curve_stats(a,[2,1.75,1.5,1.25,1]);self.assertAlmostEqual(r['linear_slope'],-1);self.assertFalse(r['nonmonotonic']);self.assertEqual(r['oracle_best_alpha'],1);self.assertAlmostEqual(r['max_abs_second_diff'],0)
  r=curve_stats(a,[1,0,1,2,3]);self.assertTrue(r['nonmonotonic']);self.assertEqual(r['oracle_best_alpha'],.25)
  self.assertEqual(curve_stats(a,[1,1,1,1,1])['oracle_best_alpha'],0)
 def test_invalid_curve_blocks(self):
  for x,y in [([0,.5,1],[0,0,0]),([0,.25,.5,.75,1],[0,0,np.nan,0,0])]:
   with self.assertRaises(ValueError):curve_stats(x,y)
 def test_cpu_candidate_provenance_and_norm(self):
  p=ROOT/'docs/m05x/p3c0';x=json.loads((p/'candidate_cpu_geometry.json').read_text(encoding='utf8'));m=json.loads((ROOT/'docs/m05x/p3b/evidence_manifest.json').read_text(encoding='utf8'))
  self.assertEqual(x['device'],'cpu');self.assertFalse(x['state_commit_performed']);self.assertFalse(x['gt_read']);self.assertEqual(len(x['results']),3)
  for r in x['results']:
   self.assertEqual(r['source_npz_sha256'],m['anchors'][r['anchor_id']]['files']['evidence_arrays.npz']);self.assertGreater(r['v4_delta_l2_cpu'],0);self.assertGreater(r['v5_delta_l2_saved'],0);self.assertLess(abs(r['cos_v4_cpu_vs_v5_saved']),.999);self.assertLess(r['matched_norm_relative_error'],1e-6)
 def test_manifest_classification_does_not_certify(self):
  p=ROOT/'docs/m05x/p3c0';s=json.loads((p/'independent_data_summary.json').read_text(encoding='utf8'));self.assertEqual(s['independent_validation_certified'],0);self.assertEqual(s['independent_final_test_certified'],0)
  with (p/'independent_data_audit.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
  self.assertEqual(len(rows),290);self.assertTrue(all(x['independent_final_test_certified']=='False' for x in rows))
if __name__=='__main__':unittest.main()
