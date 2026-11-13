"""Stdlib protocol/recovery checks plus Torch module tests when Torch is installed."""
from .common import *
import ast,copy,importlib.util,tempfile,unittest

class ProtocolTests(unittest.TestCase):
    def setUp(self):self.p=read(HERE/'policy.json')
    def manifests(self):
        return {s:[dict(gt_match_status='MATCHED') for _ in range(300)] for s in self.p['old_sequences']}
    def test_prefix_h1_never_crosses_tail_or_reset(self):
        cc=cases_for(self.manifests(),self.p)
        self.assertEqual(len(cc),72)
        for c in cc:self.assertLess(c['anchor']+1,256);self.assertEqual(c['anchor']//32,(c['anchor']+1)//32)
    def test_missing_next_GT_cannot_train(self):
        mm=self.manifests();mm['crowd2'][9]['gt_match_status']='MISSING_DEPTH_FILE'
        self.assertNotIn('crowd2_000008',[c['id'] for c in cases_for(mm,self.p)])
    def test_no_training_on_new_recordings(self):
        self.assertTrue(set(c['sequence'] for c in cases_for(self.manifests(),self.p)).isdisjoint(self.p['development_sequences']))
    def test_schedule_preserves_resets(self):
        self.assertFalse(active(0));self.assertFalse(active(32));self.assertTrue(active(8));self.assertFalse(active(9))
    def test_round_robin_order_equal_exposure(self):
        cc=cases_for(self.manifests(),self.p);o=order_for(cc,999,20261030,self.p['old_sequences'])
        for s in self.p['old_sequences']:self.assertEqual(sum(x.startswith(s+'_') for x in o),333)
    def test_order_deterministic(self):
        cc=cases_for(self.manifests(),self.p);self.assertEqual(order_for(cc,1000,7,self.p['old_sequences']),order_for(cc,1000,7,self.p['old_sequences']))
    def test_recovery_rejects_duplicate_steps(self):
        with self.assertRaises(RuntimeError):check_recovery([dict(step=1,case_id='a'),dict(step=1,case_id='b')],['a','b'])
    def test_recovery_rejects_changed_order(self):
        with self.assertRaises(RuntimeError):check_recovery([dict(step=1,case_id='b')],['a'])
    def test_recovery_accepts_valid_cursor(self):check_recovery([dict(step=1,case_id='a')],['a'])
    def test_gpu_UUID_resolved_without_Torch(self):
        u,p=gpu_lock_path('1','0, GPU-aaaa\n1, GPU-bbbb\n');self.assertEqual(u,'bbbb');self.assertIn('svrdepth_diagnostics_',p.name)
    def test_multi_gpu_selector_rejected(self):
        with self.assertRaises(RuntimeError):gpu_lock_path('0,1','0, GPU-a\n1, GPU-b')
    def test_unknown_gpu_selector_rejected(self):
        with self.assertRaises(RuntimeError):gpu_lock_path('9','0, GPU-a')
    def test_mask_selection_is_GT_status_only(self):
        self.assertTrue(matched({'gt_match_status':'MATCHED'}));self.assertFalse(matched({'gt_match_status':'MISSING_DEPTH_FILE'}))
    def test_macro_is_equal_recordings(self):
        rr=[dict(sequence='a',AbsRel=1.)]*100+[dict(sequence='b',AbsRel=3.)]
        self.assertEqual(aggregate_rows(rr),2.)
    def test_hash_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a';p.write_text('ok');m={str(p):sha(p)};verify_hashes(m);p.write_text('bad')
            with self.assertRaises(RuntimeError):verify_hashes(m)
    def test_empty_group_rejected(self):
        with self.assertRaises(RuntimeError):aggregate_rows([])
    def test_no_automatic_download_or_final_test(self):
        self.assertFalse(self.p['auto_download']);self.assertFalse(self.p['final_test']);self.assertFalse(self.p['full_method_pass'])
    def test_policy_targets_are_H1(self):self.assertEqual(self.p['horizon'],1);self.assertEqual(self.p['train_updates'],1000)
    def test_model_has_no_future_GT_input(self):
        tree=ast.parse((HERE/'models.py').read_text());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        f=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        self.assertEqual([a.arg for a in f.args.args],['self','features','causal','force_alpha'])
    def test_forward_does_not_detach_commit(self):
        tree=ast.parse((HERE/'models.py').read_text());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        f=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        # Detaching the denominator is intentional; returned commit/features must not detach.
        self.assertIn('commits.append(x+commit_delta)',ast.get_source_segment((HERE/'models.py').read_text(),f))

class OrchestrationTests(unittest.TestCase):
    def test_quality_timing_report_resume_and_tamper_with_CPU_fixture(self):
        # The production evaluation functions execute; only inference and GPU monitor are fixtures.
        import numpy as np
        import contextlib,io,sys,types
        from . import evaluation as ev
        policy=read(HERE/'policy.json');A,B=policy['arms'];sequences=policy['old_sequences']+policy['development_sequences']
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);rgb=out/'rgb.png';rgb.write_bytes(b'CPU fixture bytes, never decoded by production model')
            manifests={s:[dict(rgb_path=str(rgb),frame_index_zero_based=i,gt_match_status='UNMATCHED' if i%5==0 else 'MATCHED') for i in range(24)] for s in sequences}
            gt=np.ones((3,4),dtype='float32')
            torch=types.SimpleNamespace(device=lambda x:x,cuda=types.SimpleNamespace(synchronize=lambda:None))
            rt=types.SimpleNamespace(np=np,torch=torch,exp1=types.SimpleNamespace(load_frame=lambda *a,**k:{'gt_np':gt}))
            ctx=types.SimpleNamespace(out=out,signature='fixture',policy=policy|dict(tail_start={s:4 for s in policy['old_sequences']}),
                old_sequences=policy['old_sequences'],manifests=manifests,rt=rt,plan=dict(prediction_retention='fixture'))
            values={'A':1.10,'O_fast':1.08,'O_multi':1.03,'C_star':1.05,A:1.048,B:1.04}
            calls=[]
            class FakePredictor:
                def __init__(self,c,m,s):self.method=m;self.shape={}
                def step(self,i,record):
                    calls.append((self.method,i));return np.full((3,4),values[self.method],dtype='float32'),dict(keyframe=i%32==0,age=i%32)
                def reset(self):pass
                def close(self):pass
            oldpredictor=ev.Predictor;oldbenchmark=sys.modules.get('gt4a2_y08cal_y09pareto')
            ev.Predictor=FakePredictor;sys.modules['gt4a2_y08cal_y09pareto']=types.SimpleNamespace(gpu_endpoint=lambda rt:{'clean':True})
            try:
                with contextlib.redirect_stdout(io.StringIO()):summary=ev.evaluate_all(ctx)
                self.assertEqual(summary['quality_units'],48);self.assertEqual(summary['timing_trials'],144)
                self.assertFalse(summary['method_pass']);self.assertFalse(summary['screening_gate_passed'])
                rr=csvread(out/'quality/A/crowd2.csv');self.assertEqual(len(rr),24);self.assertEqual(rr[0]['gt_match_status'],'UNMATCHED')
                count=len(calls)
                with contextlib.redirect_stdout(io.StringIO()):again=ev.evaluate_all(ctx)
                self.assertEqual(len(calls),count);self.assertEqual(summary,again)
                q=out/'quality/A/crowd2.csv';q.write_bytes(q.read_bytes()+b'changed')
                with self.assertRaises(RuntimeError):ev.quality(ctx,'A','crowd2')
            finally:
                ev.Predictor=oldpredictor
                if oldbenchmark is None:sys.modules.pop('gt4a2_y08cal_y09pareto',None)
                else:sys.modules['gt4a2_y08cal_y09pareto']=oldbenchmark

@unittest.skipUnless(importlib.util.find_spec('torch'),'Torch is unavailable locally; execute module tests in svrdepth')
class TorchContractTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t=torch;torch.manual_seed(9);torch.set_num_threads(2)
        self.p=read(HERE/'policy.json')|dict(feature_channels=16,hidden=8,levels=2)
        self.features=[torch.randn(1,16,17,19),torch.randn(1,16,5,7)];self.z=torch.randn(1,4,17,19)
    def model(self,series='N13',arm='N13_B'):
        from .models import ResidualPaths
        return ResidualPaths(self.p|dict(experiment=series),arm)
    def test_zero_initialization_identity_both_routes(self):
        for series in ['D13','N13']:
            y=self.model(series,series+'_B')(self.features,self.z)
            for x,r,c in zip(self.features,y['read'],y['commit']):self.assertTrue(self.t.equal(x,r));self.assertTrue(self.t.equal(x,c))
    def test_forward_never_mutates_input_features(self):
        m=self.model();before=[x.clone() for x in self.features];m(self.features,self.z)
        self.assertTrue(all(self.t.equal(x,b) for x,b in zip(self.features,before)))
    def test_N_B_readout_has_no_future_gradient(self):
        m=self.model();y=m(self.features,self.z);sum(x.square().mean() for x in y['commit']).backward()
        self.assertTrue(all(p.grad is None for p in m.read_heads.parameters()))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in m.trans_heads.parameters() if p.grad is not None),0)
    def test_N_A_ties_current_and_commit(self):
        m=self.model(arm='N13_A')
        with self.t.no_grad():m.read_heads[0].bias.fill_(.1)
        y=m(self.features,self.z);self.assertTrue(all(self.t.equal(x,z) for x,z in zip(y['read'],y['commit'])))
    def test_N_AB_parameter_counts_match(self):
        self.assertEqual(sum(p.numel() for p in self.model(arm='N13_A').parameters()),sum(p.numel() for p in self.model(arm='N13_B').parameters()))
    def test_D_alpha_zero_preserves_commit(self):
        m=self.model('D13','D13_B')
        with self.t.no_grad():
            for h in [*m.trans_heads,*m.read_heads]:h.bias.fill_(.1)
        y=m(self.features,self.z,force_alpha=0.)
        self.assertTrue(all(self.t.equal(x,z) for x,z in zip(self.features,y['commit'])))
        self.assertTrue(any(not self.t.equal(x,z) for x,z in zip(self.features,y['read'])))
    def test_D_zero_residual_backward_is_finite(self):
        m=self.model('D13','D13_B');y=m(self.features,self.z);sum(x.square().mean() for x in y['commit']).backward()
        self.assertTrue(all(self.t.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
    def test_D_frame_label_not_self_gated(self):
        from .models import gate_loss
        a=self.t.full((1,1,2,3),.5,requires_grad=True);delta=self.t.tensor(.1,requires_grad=True)
        gate_loss([a],delta).backward();self.assertIsNone(delta.grad);self.assertTrue((a.grad>0).all())
    def test_separate_GT_masks_and_invalid_prediction(self):
        from .runtime import loss
        p=self.t.ones(3);gt=self.t.tensor([1.,0.,2.]);self.assertAlmostEqual(loss(self.t,p,gt).item(),.25)
        with self.assertRaises(RuntimeError):loss(self.t,self.t.tensor([float('nan'),1.,1.]),gt)

if __name__=='__main__':unittest.main(verbosity=2)
