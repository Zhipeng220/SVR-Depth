#!/usr/bin/env python3
"""CPU protocol and real unit/report control-flow tests. CUDA smoke stays mandatory."""
import hashlib, json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import y03_core as core
import y03_runtime as runtime
import y03_logic as logic

class Tensor:
    def __init__(self,x):self.x=np.array(x,dtype=np.float64);self._version=0
    @property
    def shape(self):return self.x.shape
    @property
    def dtype(self):return self.x.dtype
    @property
    def device(self):return 'fake-device'
    def detach(self):return self
    def clone(self):return Tensor(self.x.copy())
    def to(self,device):return self
    def float(self):return self
    def square(self):return Tensor(self.x**2)
    def sum(self):return Tensor(self.x.sum())
    def item(self):return self.x.item()
    def data_ptr(self):return id(self.x)
    def __add__(self,y):return Tensor(self.x+(y.x if isinstance(y,Tensor) else y))
    def __sub__(self,y):return Tensor(self.x-(y.x if isinstance(y,Tensor) else y))
    def __mul__(self,y):return Tensor(self.x*(y.x if isinstance(y,Tensor) else y))
    __rmul__=__mul__

class Torch:
    @staticmethod
    def equal(a,b):return np.array_equal(a.x,b.x)
    @staticmethod
    def allclose(a,b,**kw):return np.allclose(a.x,b.x,**kw)
    @staticmethod
    def isfinite(t):return np.isfinite(t.x)

def state(value=2.):return Tensor([value]),Tensor([value/7.]),[Tensor([value+i*.01]) for i in range(4)],Tensor([1.])
def tensor_sha(d):return hashlib.sha256(b''.join(k.encode()+d[k].x.tobytes() for k in sorted(d))).hexdigest()

class FakeEngine:
    """Scalar deterministic kernels; execute the real package's fork/effect/recovery code."""
    def __init__(self,ctx,seq):self.ctx=ctx;self.seq=seq;self.rt=ctx.rt
    def load(self,index,gt=False):
        matched=bool(gt and index!=7)
        return dict(index=index,sequence=self.seq,rgb_file=f'{index}.png',rgb_timestamp=float(index),
            gt_np=np.full((2,2),2.) if matched else None,
            gt_path=Path('/dataset')/self.seq/'depth'/f'{index}.png' if matched else None)
    def static(self,prev,frame,origin):return {'fixed_evidence':frame['index']}
    def step(self,old,prev,frame,k,origin,repeats=0,beta=1.,static=None):
        i=frame['index'];trace=[]
        if i%k==0:result=state(2.+.003*i);calls=0
        else:
            f=[x*.7+old[0]*.3+.002*i for x in old[2]];first=f
            def capture(j,updated):trace.append(dict(iteration=j,beta=beta,relative_feature_change=runtime.norm_delta(Torch,updated,first)))
            f=core.repeat_features(f,lambda xs:[x*.92 for x in xs],repeats,beta,capture)
            value=sum(x.x.item() for x in f)/4
            result=(Tensor([value]),Tensor([value/7]),f,old[3]);calls=repeats
        pred=np.full((2,2),result[0].item(),dtype=np.float32)
        return result,pred,calls,trace
    metric=runtime.Engine.metric
    def reference_audit(self,*args):return 0.

class SyntheticRuntime:
    calls=0
    @classmethod
    def execution_units(cls,ctx,phase,planned,runner,**kw):
        for c in planned:
            folder=ctx.out/phase/c['case_id']
            if core.load_existing_unit(folder,ctx.signature) is None:
                cls.calls+=1;runner(ctx,FakeEngine(ctx,c['sequence']),c,state(),folder,**kw)
        return runtime.close_and_collect(ctx,phase,planned)
    close_and_collect=staticmethod(runtime.close_and_collect)

class ProtocolTests(unittest.TestCase):
    def test_frozen_plan_independent_splits_and_counts(self):
        prefix,tail=core.cases('prefix'),core.cases('tail')
        self.assertEqual(len(prefix),54);self.assertEqual(len(tail),108)
        self.assertEqual(len({c['case_id'] for c in tail}),108)
        self.assertTrue(all(c['origin']=='A' and c['anchor']+core.H<256 for c in prefix))
        self.assertTrue(all(c['anchor']>=core.TAIL[c['sequence']] and c['future_end']<core.N_RGB[c['sequence']] for c in tail))
        self.assertTrue(all(c['snapshot_index']==c['anchor']-1 and c['anchor']%c['K']==c['age'] for c in tail))

    def test_repeat_and_relaxation_actual_kernel(self):
        calls=[]
        def head(x):calls.append(1);return [t*2 for t in x]
        x=[np.ones(3)]
        self.assertTrue(np.allclose(core.repeat_features(x,head,4,1)[0],16))
        self.assertEqual(len(calls),4)
        self.assertTrue(np.allclose(core.repeat_features(x,head,4,.5)[0],1.5**4))
        self.assertTrue(np.array_equal(core.repeat_features(x,head,4,0)[0],x[0]))
        self.assertEqual(len(calls),12)  # beta0 still consumes/records proposals
        with self.assertRaisesRegex(RuntimeError,'Invalid'):core.repeat_features(x,head,-1,1)

    def test_factorial_components_are_cloned_and_rays_fixed(self):
        a,p=state(2),state(4)
        for f in (0,1):
            for rc in (0,1):
                mixed=runtime.mix_components(a,p,f,rc)
                self.assertEqual(mixed[0].item(),4 if rc else 2)
                self.assertEqual(mixed[2][0].item(),4 if f else 2)
                self.assertEqual(mixed[3].item(),a[3].item())
                mixed[2][0].x[:]=99
                self.assertEqual(a[2][0].item(),2);self.assertEqual(p[2][0].item(),4)

    def test_numerical_failures_not_averaged_away_and_missing_GT_kept(self):
        rows=[]
        for arm in ('base','treatment'):
            for h in range(17):
                rows.append(dict(arm=arm,h=h,matched=h!=2,valid_prediction=not(arm=='treatment' and h==2),
                    AbsRel=.1,RMSE=.2,delta1=.9,after_reset=h>=12,head_calls_cumulative=0))
        result=core.paired_windows(rows,'base')
        bad=[r for r in result if r['arm']=='treatment']
        self.assertTrue(all(r['status']=='NUMERICAL_FAILURE' and r['mean_AbsRel']=='' for r in bad))
        self.assertEqual(bad[0]['n_future_RGB'],4);self.assertEqual(bad[0]['n_GT'],3)

    def test_future_contrasts_exclude_common_current_frame(self):
        rows=[dict(arm=a,h=h,matched=True,valid_prediction=True,AbsRel=(99 if a=='treated' and h==0 else .1),
            RMSE=.2,delta1=.9,after_reset=False,head_calls_cumulative=1) for a in ('base','treated') for h in range(17)]
        self.assertTrue(all(r['gain_vs_control']==0 for r in core.paired_windows(rows,'base')))

    def test_archive_preserves_external_input_even_if_last_key_is_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.mkdir();image=root/'depth.png';image.write_bytes(b'PNG_SENTINEL')
            core.write_json(out/'protocol.json',dict(input_sha256={str(image):core.digest(image)}))
            core.write_json(out/'summary.json',dict(status='fixture'))
            path=core.make_archive(out,logic.PREFIX,[])
            self.assertEqual(path,out/(logic.PREFIX+'_review.zip'));self.assertEqual(image.read_bytes(),b'PNG_SENTINEL')
            with __import__('zipfile').ZipFile(path) as z:self.assertIsNone(z.testzip());self.assertIn('summary.json',z.namelist())

    def test_recovery_rejects_changed_identity_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);core.write_csv(path/'frames.csv',[dict(x=1)])
            core.write_json(path/'complete.json',dict(signature='one',rows_sha256=core.digest(path/'frames.csv'),audit_files={}))
            self.assertIsNotNone(core.load_existing_unit(path,'one'))
            with self.assertRaisesRegex(RuntimeError,'identity'):core.load_existing_unit(path,'two')
            (path/'frames.csv').write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'CSV'):core.load_existing_unit(path,'one')

    def test_experiment_specific_matched_budget_or_interaction(self):
        if logic.EXPERIMENT=='Y03':
            for k,ages in core.AGES.items():
                for age in ages:
                    for q in logic.BUDGETS:
                        fn=logic.schedules('alternate',q,k,age)
                        calls=1+sum(fn(h,age+h)[0] if (age+h)%k else 0 for h in range(1,17))
                        self.assertEqual(calls,q);self.assertEqual(logic.budget_count('alternate',q,k,age,16),q)
        else:
            v=logic.factorial_effects(10,7,8,3)
            self.assertEqual(v['F_main'],4);self.assertEqual(v['RC_main'],3);self.assertEqual(v['interaction'],2)
            self.assertEqual(logic.factorial_effects(10,7,8,5)['interaction'],0)

    def test_real_unit_and_report_pipeline_with_synthetic_kernels(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=SimpleNamespace(out=Path(tmp),signature='fixture',args=SimpleNamespace(stage='all',smoke_only=True,save_depth='audit'),
                rt=SimpleNamespace(np=np,torch=Torch,exp1=SimpleNamespace(state_sha256=tensor_sha)))
            ctx.manifests={s:[dict(gt_match_status='UNMATCHED' if i==7 else 'MATCHED') for i in range(core.N_RGB[s])] for s in core.SEQS}
            ctx.plan=dict(tail_cases=core.cases('tail',True),prefix_cases=core.cases('prefix',True) if logic.EXPERIMENT=='Y03' else [])
            result=logic.execute(ctx,SyntheticRuntime)
            self.assertEqual(result['frame_records'],6*17*(12 if logic.EXPERIMENT=='Y03' else 5))
            self.assertEqual(result['numerical_failure_branches'],0)
            calls=SyntheticRuntime.calls
            again=logic.execute(ctx,SyntheticRuntime)
            self.assertEqual(calls,SyntheticRuntime.calls);self.assertEqual(result,again)
            if logic.EXPERIMENT=='Y03':
                rows,_=runtime.close_and_collect(ctx,'prefix',ctx.plan['prefix_cases'])
                rows[0]['index']=256
                with self.assertRaisesRegex(RuntimeError,'leakage'):logic.select_beta(ctx,rows)
            else:
                rows=core.read_csv(ctx.out/'diagnose'/ctx.plan['tail_cases'][0]['case_id']/'frames.csv')
                first=[r for r in rows if r['h']=='0']
                self.assertEqual(len({r['AbsRel'] for r in first}),1)
                audit=core.read_json(ctx.out/'diagnose'/ctx.plan['tail_cases'][0]['case_id']/'cache_contract_audit.json')
                self.assertTrue(audit['confidence_passive_canary']);self.assertTrue(audit['branch_order_canary'])

class TorchCPUTests(unittest.TestCase):
    def test_real_tensor_commit_and_beta_alias(self):
        try:import torch
        except ImportError:self.skipTest('torch unavailable here; real tensor test and CUDA smoke run on server')
        a=(torch.ones(1),torch.ones(1),[torch.ones(1) for _ in range(4)],torch.ones(1))
        p=tuple([t*2 for t in a[:2]])+([t*2 for t in a[2]],a[3].clone())
        mixed=runtime.mix_components(a,p,1,0)
        self.assertTrue(torch.equal(mixed[0],a[0]) and torch.equal(mixed[2][0],p[2][0]))
        x=[torch.ones(3)]
        out=core.repeat_features(x,lambda ys:[v*2 for v in ys],4,0.)
        self.assertTrue(torch.equal(out[0],x[0]));self.assertEqual(out[0]._version,x[0]._version)

if __name__=='__main__':unittest.main(verbosity=2)
