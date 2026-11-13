"""CPU protocol unit tests. They DO NOT imply actual VeloDepth CUDA verification."""
import importlib.util
from pathlib import Path
import json
import sys
import types
import unittest

import torch

HERE = Path(__file__).resolve().parents[1]
MOD = HERE/'experiments/p1/cvpr_m04_closed_loop.py'
spec = importlib.util.spec_from_file_location('cvpr_m04_closed_loop', MOD)
m04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m04)


class ProtocolTest(unittest.TestCase):
    def test_pinned_sources_match_m05_lock(self):
        lock = json.loads((HERE/'reference/M05_protocol_lock.json').read_text())
        expected = {k:v['actual'] for k,v in lock['source_hashes'].items()}
        self.assertEqual(m04.PINNED_SOURCES, expected)

    def test_branches_and_hyperparameters(self):
        self.assertEqual(m04.BRANCHES, ('A_off','O_same_head','F_only','P_full'))
        self.assertEqual((m04.K,m04.GAIN,m04.FREQUENCY),(16,1.0,1))

    def test_same_head_state_and_readout_contract(self):
        class FakeGT2:
            @staticmethod
            def make_z_depth(radius, rays, padded, padding, mode):
                return radius.squeeze(0).squeeze(0)
        def fake_fast(model, prev, curr, static, r, c, f):
            if model.pixel_decoder.feature_repair_mode=='off':
                return r+1, c+1, [x+2 for x in f]
            return r+3, c+4, [x+5 for x in f]
        exp1=types.SimpleNamespace(one_fast_step=fake_fast)
        model=types.SimpleNamespace(pixel_decoder=types.SimpleNamespace(feature_repair_mode='off'),interpolation_mode='bilinear')
        state={'radius':torch.zeros(1,1,2,2),'confidence':torch.zeros(1,1,2,2),
               'features':[torch.zeros(1,1,2,2)],'rays_map':torch.ones(1,1,2,2)}
        shape={'padded_shape':(2,2),'paddings':[0,0,0,0]}
        # CUDA autocast automatically disables on this CPU runner; this is a
        # control-flow test only, not a claim of real model equivalence.
        outputs={}
        for b in m04.BRANCHES:
            new,pred,count=m04.apply_branch(model,exp1,FakeGT2,None,None,None,state,b,shape,torch)
            outputs[b]=(new,pred,count)
        a,o,f,p=(outputs[x] for x in m04.BRANCHES)
        self.assertEqual(m04.maxdiff(a[0],o[0]),0)
        self.assertEqual(m04.maxdiff(a[0]['radius'],f[0]['radius']),0)
        self.assertEqual(m04.maxdiff(f[0]['features'],p[0]['features']),0)
        self.assertEqual(m04.maxdiff(o[0]['features'],a[0]['features']),0)
        self.assertEqual(m04.maxdiff(f[0]['rays_map'],a[0]['rays_map']),0)
        self.assertEqual(m04.maxdiff_tensors_numpy(a[1],f[1]),0)
        self.assertEqual(m04.maxdiff_tensors_numpy(o[1],p[1]),0)
        self.assertEqual((a[2],o[2],f[2],p[2]),(1,2,2,1))

    def test_percentile(self):
        self.assertEqual(m04.percentile([1,3,5],50),3)
        self.assertEqual(m04.percentile([1,3,5],100),5)


if __name__=='__main__':unittest.main()
