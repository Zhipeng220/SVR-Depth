#!/usr/bin/env python3
"""Meaningful protocol/analysis tests; torch CPU tests also run on the server."""
import contextlib
import importlib.util
from pathlib import Path
import sys
import unittest
import ast
from types import SimpleNamespace,ModuleType
from unittest.mock import patch
import numpy as np
from y08y09_common import *


class ProtocolTests(unittest.TestCase):
    def test_equal_selection_grid_and_zero_alias(self):
        pp=plan()
        self.assertEqual(len(pp),36)
        for f in FAMILIES:
            cc=[r for r in pp if r['family']==f]
            self.assertEqual(len(cc),8)
            self.assertEqual({r['schedule'] for r in cc},{'K16','K32','K64','Auto'})
        self.assertEqual(config('P','K64',0,4),config('A','K64'))
        self.assertEqual(config('O_fast','Auto',0),config('A','Auto'))

    def test_train_tail_boundary_and_missing_GT(self):
        self.assertTrue(is_fit_frame(255))
        for i in (0,32,256,768):self.assertFalse(is_fit_frame(i))
        self.assertFalse(is_fit_frame(1,False))
        self.assertFalse(keep_prediction('crowd2',512))
        self.assertTrue(keep_prediction('crowd2',768))

    def test_update_schedule_first_then_every_four(self):
        self.assertEqual([i for i in range(32) if repair_due(i,4)],[1,5,9,13,17,21,25,29])
        self.assertEqual(sum(repair_due(i,1) for i in range(64)),63)

    def test_order_matches_original_fair_multiscene(self):
        import hashlib
        self.assertEqual(hashlib.sha256(','.join(map(str,training_order())).encode()).hexdigest(),
            'aa7aff6213fdbf9ddce8703e2eb707e8cb8eab495a575f96698d9d25d9b9d934')

    def test_global_calibration_merge_and_no_video_fit(self):
        xx=np.linspace(1,5,5000).reshape(50,100)
        yy=1.3*xx+.2
        a=Moments();a.add(xx,yy,np)
        b=Moments();c=Moments();b.add(xx[:25],yy[:25],np);c.add(xx[25:],yy[25:],np);b.merge(c)
        aa,bb=a.finish(),b.finish()
        self.assertAlmostEqual(aa['scale'],1.3,places=10)
        self.assertAlmostEqual(aa['bias_m'],.2,places=10)
        self.assertAlmostEqual(aa['scale'],bb['scale'],places=10)
        self.assertEqual(aa['fit_pixels'],5000)

    def test_invalid_prediction_cannot_shrink_mask(self):
        gt=np.ones((40,40));p=gt.copy();p[0,0]=np.nan
        with self.assertRaisesRegex(RuntimeError,'Invalid'):metrics(p,gt,np)
        with self.assertRaisesRegex(RuntimeError,'Invalid'):Moments().add(p,gt,np)

    def test_pareto_requires_all_risk_axes_and_verified_timing(self):
        x=dict(AbsRel=.05,RMSE=.2,delta1=.98,worst_sequence_AbsRel=.08,harm_fraction=.1,frame_AbsRel_p95=.1,latency_ms=90)
        y=dict(AbsRel=.06,RMSE=.21,delta1=.97,worst_sequence_AbsRel=.09,harm_fraction=.2,frame_AbsRel_p95=.11,latency_ms=100)
        self.assertTrue(dominates(x,y));self.assertFalse(dominates(x,x))
        self.assertFalse(dominates(dict(x,frame_AbsRel_p95=.12),y))
        self.assertFalse(dominates(dict(x,delta1=.96),y))
        self.assertEqual(pareto_rows([],[]),[])

    def test_common_RGB_windows_not_method_keyframe_cycles(self):
        rows=[]
        for i in range(69):
            for name,v in [('A_K32_g0_f1/raw',.1),('P_K64_g1_f1/raw',.11)]:
                rows.append(dict(variant=name,sequence='balloon2',index=256+i,matched=i!=35,scored=True,
                    AbsRel=v,RMSE=v*2,delta1=.9,age=i%64,keyframe=i%64==0))
        ss=quality_summary(rows)
        p=next(r for r in ss if r['variant'].startswith('P_'))
        self.assertEqual(p['complete_common_32RGB_windows'],2)
        self.assertEqual(p['harmed_common_windows'],2)
        self.assertEqual(p['scored_GT_frames'],68)

    def test_real_prior_training_exposure_when_available(self):
        root=Path('/workspace/scratch/deba84404a45/analysis/y04_fix3/outputs/gt4a2_y04_validation_fix3_20261025_225902')
        p=root/'y04_all_frame_metrics.csv'
        if not p.exists():self.skipTest('Prior local CSV absent; server preflight independently checks manifests')
        old=csv_read(p);counts=[]
        for seq,c in SLOTS:
            counts.append(sum(r['sequence']==seq and r['arm']=='A_native' and str(r['gt_matched']).lower()=='true'
                and c*32<int(r['frame_index_zero_based'])<c*32+32 for r in old))
        self.assertEqual(sum(counts),740)
        self.assertEqual(sum(counts[i] for i in training_order()),30834)

    def test_recovery_rejects_duplicate_or_changed_order(self):
        curve=[dict(step=1,slot=3,supervised_frames=31),dict(step=2,slot=5,supervised_frames=30)]
        validate_recovery(curve,2,[3,5,7],61)
        with self.assertRaisesRegex(RuntimeError,'Duplicate'):validate_recovery(curve+[curve[-1]],2,[3,5,7],61)
        with self.assertRaisesRegex(RuntimeError,'order'):validate_recovery(curve,2,[3,6,7],61)
        with self.assertRaisesRegex(RuntimeError,'exposure'):validate_recovery(curve,2,[3,5,7],62)


class _Tensor:
    def __init__(self,x):self.x=np.asarray(x)
    def clone(self):return _Tensor(self.x.copy())
    def norm(self,**kw):return self
    def quantile(self,q):return _Tensor(np.quantile(self.x,q))
    def item(self):return self.x.item()
    def float(self):return self
    def bool(self):return _Tensor(self.x.astype(bool))
    def mean(self):return _Tensor(self.x.mean())
    def __lt__(self,x):return _Tensor(self.x<x)
    def __and__(self,x):return _Tensor(self.x & x.x)


class AutoControlTests(unittest.TestCase):
    """Execute the actual Stream methods with scalar fake model kernels."""
    @classmethod
    def setUpClass(cls):
        source=Path(__file__).with_name('y08y09_models.py').read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Stream')
        fake_torch=SimpleNamespace(autocast=lambda *a,**kw:contextlib.nullcontext(),float16=None,
            ones_like=lambda x:_Tensor(np.ones_like(x.x)))
        fake_f=SimpleNamespace(grid_sample=lambda x,*a,**kw:_Tensor(np.zeros_like(x.x)))
        env=dict(require=require,repair_due=repair_due,torch=fake_torch,F=fake_f,
            output_depth=lambda rt,m,state,shape:state[0],GainHead=type('GainHead',(),{}))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_Stream_methods','exec'),env)
        cls.Stream=env['Stream']

    def stream(self,schedule='Auto',strength=2.):
        s=self.Stream.__new__(self.Stream);s.cfg=config('A',schedule);s.debug=False;s.shape={}
        s.model=SimpleNamespace(pixel_decoder=SimpleNamespace(persistent_feature_repair=None),
            alpha_flow=.1,beta_flow=.15,alpha_luma=.1,beta_luma=.15)
        s.rt=SimpleNamespace(exp1=SimpleNamespace(keyframe_state=lambda m,frame,shape:
            ('rays'+str(frame['index']),100+100*frame['index'],200+100*frame['index'],['features'+str(frame['index'])])))
        s.reset()
        def fast(frame,static,apply):
            r,c,f,rays=s.state;s.state=(r+1,c+1,['candidate'],rays)
            s.counts['native_fast_calls']+=1
            return _Tensor(0.),_Tensor(strength)
        s.fast=fast
        return s

    def test_rejected_candidate_same_frame_KF_overlap(self):
        s=self.stream();module=ModuleType('velodepth.models.decoder_velo');module.flow2grid_sample=lambda *a,**kw:None
        with patch.dict(sys.modules,{'velodepth.models.decoder_velo':module}):
            p,m=s.advance(dict(index=0),0)
            p,m=s.advance(dict(index=1),1,dict(validity_mask=_Tensor(True)))
        self.assertEqual(p,101)  # Candidate r, not fresh keyframe r=200.
        self.assertEqual(s.state[3],'rays1')
        self.assertEqual(s.state[2],['features1'])
        self.assertTrue(m['keyframe']);self.assertEqual(m['age'],0)
        self.assertEqual((s.counts['keyframe_calls'],s.counts['rejected_candidates']),(2,1))

    def test_fixed_schedule_missing_GT_still_propagates(self):
        s=self.stream('K16',0.)
        for i in range(17):s.advance(dict(index=i,gt=None),i,dict(validity_mask=_Tensor(True)))
        self.assertEqual(s.counts['keyframe_calls'],2)
        self.assertEqual(s.counts['native_fast_calls'],15)
        self.assertEqual(s.state[3],'rays16')

    def test_actual_overlap_fast_calls_and_confidence(self):
        s=self.stream();s.first=False;s.age=1;s.state=(10,20,['KF']*4,'rays')
        calls=[]
        def decoder(data,metas):calls.append(data['features_teacher']);return 30,40,['new']
        s.model.pixel_decoder.forward_fast=decoder
        s.model.small_encoder=lambda **kw:([None]*4,_Tensor(0.),None)
        static=dict(flow_norm=_Tensor(.1),image_residual=_Tensor(.2),grid_sample=None,validity_mask=_Tensor(True))
        # Invoke original class method, bypassing the stub used by other tests.
        self.Stream.fast(s,dict(img=None),static,False)
        self.assertEqual(len(calls),2)
        self.assertEqual(s.state,(30,20,['new'],'rays'))


class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:raise unittest.SkipTest('torch unavailable locally; execute CPU autograd tests on server')
        cls.torch=torch
        from y08y09_models import FastDepthAdapter,ChannelAffine,GainHead
        cls.Fast=FastDepthAdapter;cls.Affine=ChannelAffine;cls.Gain=GainHead

    def test_fast_identity_positive_gradient_and_count(self):
        t=self.torch;head=self.Fast();depth=t.full((12,16),2.);rgb=t.randn(1,3,3,4)
        out=head(depth,rgb)
        self.assertEqual(sum(p.numel() for p in head.parameters()),10465)
        self.assertTrue(t.equal(depth,out))
        loss=(out-2.2).square().mean();loss.backward()
        self.assertGreater(int(t.count_nonzero(head.net[-1].weight.grad)),0)

    def test_affine_does_not_mutate_state(self):
        t=self.torch;head=self.Affine();xs=[t.randn(1,512,2,3) for _ in range(4)]
        before=[x.clone() for x in xs];out=head(xs)
        self.assertEqual(sum(p.numel() for p in head.parameters()),4096)
        self.assertTrue(all(t.equal(x,y) for x,y in zip(before,xs)))
        self.assertTrue(all(t.equal(x,y) for x,y in zip(out,xs)))

    def test_gain_identity_and_frequency_bypass(self):
        t=self.torch
        class Add(t.nn.Module):
            def forward(self,x):return [v+2 for v in x]
        x=[t.ones(2)];head=self.Gain(Add(),.25)
        self.assertTrue(t.equal(head(x)[0],t.full((2,),1.5)))
        head.active=False;self.assertIs(head(x),x);self.assertEqual(head.calls,1)


if __name__=='__main__':unittest.main(verbosity=2)
