#!/usr/bin/env python3
"""Protocol/recovery tests and real CPU-autograd tests when torch is present."""
from pathlib import Path
from types import SimpleNamespace
import copy,importlib.util,json,pickle,shutil,tempfile,unittest
import y05_common as common
from y05_common import *
from y05_eval import configurations,probe_windows
from y05_train import materialize

HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:import torch

def fixture_manifest():
    return {s:[dict(gt_match_status='MATCHED') for _ in range(n)] for s,n in N_RGB.items()}

class ProtocolTests(unittest.TestCase):
    def test_prefix_boundaries_and_balanced_order(self):
        samples=prefix_samples();self.assertEqual(len(samples),144)
        self.assertTrue(all(c['anchor']+4<256 and c['anchor']%8==0 and c['anchor']%32!=0 for c in samples))
        order=sample_order(samples,1000,17);self.assertEqual(order,sample_order(samples,1000,17))
        lookup={c['case_id']:c for c in samples}
        for begin in range(0,996,6):self.assertEqual({(lookup[x]['sequence'],lookup[x]['origin']) for x in order[begin:begin+6]},{(s,o) for s in SEQS for o in ('A','S8')})

    def test_missing_GT_is_propagated_not_filtered_as_RGB(self):
        m=fixture_manifest();m['crowd3'][9]['gt_match_status']='UNMATCHED'
        formal=prefix_samples(m);self.assertTrue(any(c['sequence']=='crowd3' and c['anchor']==8 for c in formal))
        smoke=prefix_samples(m,True)
        self.assertEqual(len(smoke),6)
        self.assertEqual([c['anchor'] for c in smoke if c['sequence']=='crowd3'],[16,16])
        m['crowd3'][8]['gt_match_status']='UNMATCHED'
        self.assertFalse(any(c['sequence']=='crowd3' and c['anchor']==8 for c in prefix_samples(m)))

    def test_pulse_schedule_matches_primary_deployment(self):
        for k in (16,32):
            pulses=[i for i in range(64) if pulse_due(i,k)]
            self.assertTrue(all(i%k and i%8==0 for i in pulses))
        for case in prefix_samples():
            self.assertTrue(pulse_due(case['anchor'],32))
            self.assertFalse(any(pulse_due(case['anchor']+h,32) for h in range(1,5)))

    def test_probe_plan_and_cost_grid(self):
        cases=probe_cases();self.assertEqual(len(cases),54)
        self.assertTrue(all(c['anchor']>=TAIL[c['sequence']] and c['anchor']+16<N_RGB[c['sequence']] for c in cases))
        self.assertEqual(len(probe_cases(True)),6)
        cfg=configurations();self.assertEqual(len(cfg),17)
        self.assertEqual(len([s for s in cfg if s['K']==32]),12)

    def test_recovery_rejects_duplicate_changed_order_and_time(self):
        order=sample_order(prefix_samples(),3,4)
        rows=[dict(step=i+1,case_id=c,cuda_sec=.1) for i,c in enumerate(order)]
        checked_curve(rows,order)
        for mutant in ([rows[0],rows[0]], [{**rows[0],'case_id':'other'}], [{**rows[0],'cuda_sec':float('nan')}]):
            with self.assertRaises(RuntimeError):checked_curve(mutant,order)

    def test_compute_budget_first_crossing(self):
        durations=[.7,.8,.9];target=1.2;cumulative=0
        for step,t in enumerate(durations,1):
            cumulative+=t
            if compute_crossed(cumulative,target):break
        self.assertEqual(step,2);self.assertLess(cumulative-target,t)

    def test_archive_does_not_reuse_input_path_as_destination(self):
        original=common.__file__
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'experiments/p1';source.mkdir(parents=True)
            for name in FILES:(source/name).write_text('# source\n')
            (root/'run_y05.sh').write_text('# launcher\n')
            out=root/'outputs/gt4a2_y05_test/validation';out.mkdir(parents=True)
            sentinel=root/'GT.png';sentinel.write_bytes(b'ORIGINAL_GT_BYTES')
            jwrite(out/'protocol.json',dict(input_sha256={str(sentinel):sha(sentinel)}));jwrite(out/'summary.json',dict(status='test'))
            try:
                common.__file__=str(source/'y05_common.py');result=archive(out)
                self.assertEqual(result,out/'y05_review.zip');self.assertEqual(sentinel.read_bytes(),b'ORIGINAL_GT_BYTES')
                with zipfile.ZipFile(result) as z:self.assertIsNone(z.testzip());self.assertNotIn('GT.png',z.namelist())
            finally:common.__file__=original

    def test_committed_snapshot_republishes_after_interruption(self):
        class FakeTorch:
            @staticmethod
            def save(obj,path):Path(path).write_bytes(pickle.dumps(obj))
            @staticmethod
            def load(path,**kw):return pickle.loads(Path(path).read_bytes())
        ctx=SimpleNamespace(signature='sig',plan={'checkpoint_sha256':'initial'},rt=SimpleNamespace(torch=FakeTorch))
        snapshot=dict(step=25,state={'w':[2.]},head_sha256='head',cuda_sec=3.)
        obj=dict(selected={'final':snapshot},curve=[dict(step=25,case_id='x')])
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);materialize(ctx,p,obj,'F_future');saved=(p/'final.pt').read_bytes()
            (p/'final.pt').unlink();materialize(ctx,p,obj,'F_future');self.assertEqual((p/'final.pt').read_bytes(),saved)
            bad=copy.deepcopy(obj);bad['selected']['final']['head_sha256']='different'
            with self.assertRaises(RuntimeError):materialize(ctx,p,bad,'F_future')

    def test_probe_metrics_exclude_current_and_preserve_missing_GT(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);case=dict(case_id='one',sequence='crowd2',origin='A',K=32,anchor=8,age=8)
            folder=out/'probes/one';rows=[]
            for arm in ('no_pulse','candidate'):
                for h in range(17):
                    rows.append(dict(arm=arm,h=h,matched=h!=1,after_reset=False,AbsRel='' if h==1 else (100. if h==0 and arm=='candidate' else (1. if arm=='candidate' else 2.))))
            cwrite(folder/'frames.csv',rows);jwrite(folder/'complete.json',dict(signature='s',rows_sha256=sha(folder/'frames.csv')))
            ctx=SimpleNamespace(out=out,signature='s',plan={'probe_plan':[case]})
            w=probe_windows(ctx);x=next(r for r in w if r['arm']=='candidate' and r['horizon']==4)
            self.assertEqual(x['n_GT'],3);self.assertEqual(x['mean_AbsRel'],1.);self.assertEqual(x['cumulative_gain'],3.)

@unittest.skipUnless(HAS_TORCH,'torch absent locally; launcher runs CPU autograd and mandatory real CUDA audit on server')
class AutogradTests(unittest.TestCase):
    def fixture(self):
        from y05_engine import Engine
        class Head(torch.nn.Module):
            def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(.2))
            def forward(self,features):return [x+self.w for x in features]
        class Gate(torch.nn.Module):
            def __init__(self):
                super().__init__();self.head=Head();self.active=False;self.gain=1.;self.capture=False;self.calls=0;self.delta=None
            def forward(self,features):
                self.delta=None
                if not self.active:return features
                self.calls+=1;out=self.head(features)
                if self.capture:self.delta=sum((y-x).square().sum() for x,y in zip(features,out))/sum(x.detach().square().sum() for x in features)
                return out
        engine=Engine.__new__(Engine);engine.gate=Gate();engine.head=engine.gate.head
        engine.model=SimpleNamespace(pixel_decoder=SimpleNamespace(persistent_feature_repair=engine.gate))
        def fast(model,context,t,r,features):
            features=model.pixel_decoder.persistent_feature_repair([x*.9+.05 for x in features])
            rr=r*.9+sum(x[0,0] for x in features)*.025
            return rr,torch.ones_like(rr),features
        engine.rt=SimpleNamespace(torch=torch,gt2=SimpleNamespace(fast_step=fast));engine.read_depth=lambda s:s[0]
        settings=frozen_settings();ctx=SimpleNamespace(rt=engine.rt,plan={'settings':settings})
        parent=(torch.ones(2,2),torch.ones(2,2),[torch.ones(1,1,2,2) for _ in range(4)],torch.ones(1,3,2,2))
        case=dict(anchor=8,K=32,origin='A',horizon=4)
        frames={h:dict(index=8+h,gt=torch.ones(2,2)) for h in range(-1,5)};statics={h:{} for h in range(5)}
        return ctx,engine,parent,case,frames,statics

    def test_actual_pulse_forward_gradient_and_detach_negative(self):
        from y05_engine import pulse_forward
        ctx,e,parent,case,frames,statics=self.fixture()
        result=pulse_forward(ctx,e,case,parent,frames,statics,'FH_future_harm',checkpoint=True)
        self.assertEqual(e.gate.calls,1);self.assertTrue(result['future_losses'][-1].requires_grad)
        g=torch.autograd.grad(result['future'],e.head.w)[0];self.assertTrue(torch.isfinite(g));self.assertGreater(abs(g.item()),0.)
        negative=pulse_forward(ctx,e,case,parent,frames,statics,'FH_future_harm',checkpoint=True,detach_after_anchor=True)
        self.assertFalse(negative['future'].requires_grad)

    def test_actual_checkpoint_closure_matches_plain_and_restores_gate(self):
        from y05_engine import pulse_forward
        ctx,e,parent,case,frames,statics=self.fixture()
        a=pulse_forward(ctx,e,case,parent,frames,statics,'F_future',checkpoint=True)
        ga=torch.autograd.grad(a['future'],e.head.w)[0]
        self.assertFalse(e.gate.active)
        b=pulse_forward(ctx,e,case,parent,frames,statics,'F_future',checkpoint=False)
        gb=torch.autograd.grad(b['future'],e.head.w)[0]
        self.assertTrue(torch.allclose(ga,gb,atol=1e-7,rtol=1e-6))
        e.gate.calls=0
        p=pulse_forward(ctx,e,case,parent,frames,statics,'P_closed_loop',checkpoint=True)
        self.assertEqual(e.gate.calls,5)

    def test_GT_only_mask_and_harm_objective(self):
        from y05_engine import depth_loss,pulse_forward
        gt=torch.tensor([[1.,0.],[2.,float('nan')]])
        pred=torch.tensor([[1.,float('nan')],[3.,1.]],requires_grad=True)
        loss,n=depth_loss(torch,pred,gt);self.assertEqual(n,2);self.assertAlmostEqual(loss.item(),.25)
        with self.assertRaises(RuntimeError):depth_loss(torch,torch.full_like(gt,float('nan')),gt)
        ctx,e,parent,case,frames,statics=self.fixture()
        a=pulse_forward(ctx,e,case,parent,frames,statics,'F_future',checkpoint=False)
        b=pulse_forward(ctx,e,case,parent,frames,statics,'FH_future_harm',checkpoint=False)
        self.assertGreater(b['harm'].item(),0.);self.assertGreater(b['objective'].item(),a['objective'].item())

if __name__=='__main__':unittest.main(verbosity=2)
