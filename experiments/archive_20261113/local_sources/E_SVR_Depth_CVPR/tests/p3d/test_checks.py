import hashlib,importlib.util,tempfile,unittest
from pathlib import Path
root=Path(__file__).resolve().parents[2];spec=importlib.util.spec_from_file_location('checks',root/'scripts/p3d/checks.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
class AuditChecks(unittest.TestCase):
 def test_required_file(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';p.write_text('a');c.require_files([p])
   with self.assertRaises(FileNotFoundError):c.require_files([p,Path(d)/'missing'])
 def test_unique_and_missing_key(self):
  c.unique([{'a':1},{'a':2}],['a'])
  with self.assertRaises(ValueError):c.unique([{'a':1},{'a':1}],['a'])
  with self.assertRaises(ValueError):c.unique([{}],['a'])
 def test_sha(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';p.write_bytes(b'a');c.check_sha(p,hashlib.sha256(b'a').hexdigest())
   with self.assertRaises(ValueError):c.check_sha(p,'0'*64)
 def test_timestamp(self):
  c.check_times(['1','2','3'])
  for xs in ([1,1],[2,1],[1,float('nan')]):
   with self.subTest(xs=xs),self.assertRaises(ValueError):c.check_times(xs)
 def test_leakage(self):
  c.no_leak([{'role':'predictor','name':'update_l2'},{'role':'group','name':'sequence_id'}])
  for name in ('future_rgb','gt_mask','sequence_id','oracle_alpha'):
   with self.subTest(name=name),self.assertRaises(ValueError):c.no_leak([{'role':'predictor','name':name}])
 def test_result_fields(self):
  c.required_fields([{'status':'PASS','anchor_id':'x'}],['status','anchor_id'])
  with self.assertRaises(ValueError):c.required_fields([{'status':'PASS'}],['status','anchor_id'])
if __name__=='__main__':unittest.main()
