import importlib.util
import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('m05_e4c_test',ROOT/'experiments/p1/cvpr_m05_formal.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


class E4CDevelopmentTests(unittest.TestCase):
    def test_hash_seed_and_backbone_are_enforced(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.pt';p.write_bytes(b'test');digest=hashlib.sha256(b'test').hexdigest()
            obj={'identity':{'method':'P_multi','seed':20261103,'snapshot_sha256':M.EXPECTED_BASE_SHA256},
                 'head_state':{'weight':'test'}}
            torch=types.SimpleNamespace(load=lambda *args,**kwargs:obj)
            with patch.dict(sys.modules,{'torch':torch}):
                info=M.verify_development_e4c(p,digest,20261103)
                self.assertEqual(info['state_key'],'head_state')
                self.assertIn('development_not_Y04_or_formal',info['scope'])
                with self.assertRaisesRegex(RuntimeError,'identity'):
                    M.verify_development_e4c(p,digest,20261104)
                with self.assertRaisesRegex(RuntimeError,'SHA256'):
                    M.verify_development_e4c(p,'a'*64,20261103)


if __name__=='__main__':unittest.main()
