"""Protocol and production orchestration tests; Torch operator tests run in svrdepth."""
from .common import *
import ast,contextlib,importlib.util,io,tempfile,unittest

class ProtocolTests(unittest.TestCase):
    def setUp(self):self.p=read(HERE/'policy.json')
    def manifests(self):return {s:[dict(gt_match_status='MATCHED') for _ in range(300)] for s in self.p['old_sequences']}
    def test_full_prefix_has_672_current_samples(self):
        cc,cov=coverage_plan(self.manifests(),self.p);self.assertEqual(len(cc),672);self.assertEqual(len(cov),21);self.assertTrue(all(r['matched_GT']==32 and r['passed'] for r in cov))
    def test_minimum_sixteen_is_inclusive(self):
        m=self.manifests()
        for i in range(256):
            if i%8==1 and i>=128:m['crowd2'][i]['gt_match_status']='UNMATCHED'
        _,cov=coverage_plan(m,self.p);r=next(r for r in cov if r['sequence']=='crowd2' and r['phase']==1);self.assertEqual(r['matched_GT'],16);self.assertTrue(r['passed'])
    def test_fifteen_blocks_training(self):
        m=self.manifests()
        for i in range(256):
            if i%8==7 and i>=120:m['balloon2'][i]['gt_match_status']='UNMATCHED'
        _,cov=coverage_plan(m,self.p);self.assertFalse(all(r['passed'] for r in cov))
    def test_current_frame_255_is_allowed(self):
        cc,_=coverage_plan(self.manifests(),self.p);self.assertIn('crowd2_000255',[c['id'] for c in cc])
    def test_next_frame_GT_does_not_control_current_sample(self):
        m=self.manifests();m['crowd2'][2]['gt_match_status']='UNMATCHED';cc,_=coverage_plan(m,self.p)
        self.assertIn('crowd2_000001',[c['id'] for c in cc]);self.assertNotIn('crowd2_000002',[c['id'] for c in cc])
    def test_h0_excluded_and_28_of_32_enabled(self):
        self.assertEqual(sum(micro_due(i) for i in range(32)),28)
        for i in [0,8,16,24,32]:self.assertFalse(micro_due(i))
    def test_schedule_and_local_phase_are_distinct(self):
        self.assertTrue(active(8));self.assertEqual(phase(8),0);self.assertFalse(active(32));self.assertEqual(phase(31),7)
    def test_no_new_recordings_in_training(self):
        cc,_=coverage_plan(self.manifests(),self.p);self.assertTrue(set(c['sequence'] for c in cc).isdisjoint(self.p['development_sequences']))
    def test_group_order_balanced(self):
        cc,_=coverage_plan(self.manifests(),self.p);o=order_for(cc,1000,20261031,self.p['old_sequences']);by={c['id']:(c['sequence'],c['phase']) for c in cc};counts={}
        for x in o:counts[by[x]]=counts.get(by[x],0)+1
        self.assertEqual(len(counts),21);self.assertEqual(max(counts.values())-min(counts.values()),1)
    def test_order_deterministic(self):
        cc,_=coverage_plan(self.manifests(),self.p);self.assertEqual(order_for(cc,1000,7,self.p['old_sequences']),order_for(cc,1000,7,self.p['old_sequences']))
    def test_recovery_rejects_duplicate_step(self):
        with self.assertRaises(RuntimeError):check_recovery([dict(step=1,case_id='x'),dict(step=1,case_id='y')],['x','y'])
    def test_recovery_rejects_order_change(self):
        with self.assertRaises(RuntimeError):check_recovery([dict(step=1,case_id='y')],['x'])
    def test_recovery_accepts_valid_cursor(self):check_recovery([dict(step=1,case_id='x')],['x'])
    def test_gpu_resolved_without_CUDA_initialization(self):
        u,p=gpu_lock_path('1','0, GPU-aaa\n1, GPU-bbb');self.assertEqual(u,'bbb');self.assertIn('svrdepth_diagnostics_',p.name)
    def test_multi_GPU_selection_rejected(self):
        with self.assertRaises(RuntimeError):gpu_lock_path('0,1','0, GPU-a\n1, GPU-b')
    def test_equal_recording_macro(self):self.assertEqual(aggregate_rows([dict(sequence='a',AbsRel=1)]*100+[dict(sequence='b',AbsRel=3)]),2)
    def test_hash_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_text('good');v={str(p):sha(p)};verify_hashes(v);p.write_text('bad')
            with self.assertRaises(RuntimeError):verify_hashes(v)
    def test_every_state_field_must_match(self):
        e={k:'same' for k in ['radius','confidence','F0','F1','F2','F3','rays']}
        for k in e:
            actual=e.copy();actual[k]='changed'
            with self.assertRaises(RuntimeError):compare_state(e,actual)
    def test_missing_state_field_rejected(self):
        with self.assertRaises(RuntimeError):compare_state({'rays':'x'}, {})
    def test_policy_disables_other_mechanisms(self):
        for k in ['future_GT','teacher','state_writeback','function_distillation','harm_gate','convex_rollback','auto_download','final_test']:self.assertFalse(self.p[k])
    def test_micro_source_has_no_persistent_assignment(self):
        tree=ast.parse((HERE/'runtime.py').read_text());carrier=next(n for n in tree.body if isinstance(n,ast.ClassDef));micro=next(n for n in carrier.body if isinstance(n,ast.FunctionDef) and n.name=='micro')
        for n in ast.walk(micro):
            if isinstance(n,(ast.Assign,ast.AugAssign,ast.AnnAssign)):
                targets=n.targets if isinstance(n,ast.Assign) else [n.target]
                self.assertFalse(any(isinstance(t,ast.Attribute) and t.attr in ['state','previous'] for t in targets))
    def test_cached_training_has_no_step_or_rollout(self):
        tree=ast.parse((HERE/'runtime.py').read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='micro_from_cache')
        forbidden={'advance','tensor_step','context','pair_forward','future','fast_step'}
        self.assertTrue(all(n.attr not in forbidden for n in ast.walk(f) if isinstance(n,ast.Attribute)))
    def test_MLP_parameter_formula(self):self.assertEqual((3*8+8)+(8*1024+1024)+(3+1),9252)

class OrchestrationTests(unittest.TestCase):
    def test_real_CLI_coverage_block_precedes_GPU_or_optimizer(self):
        import sys,types
        from . import common as cm,runner
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base=root/'outputs/gt4a2_y05r_fixture/validation';base.mkdir(parents=True);cp=base/'training/C_star/final.pt';cp.parent.mkdir(parents=True);cp.write_bytes(b'CPU coverage fixture, not a real model')
            image=root/'image.png';image.write_bytes(b'CPU fixture');paths={};hashes={}
            policy=read(HERE/'policy.json')
            for s in policy['old_sequences']:
                p=root/(s+'.csv');rr=[dict(frame_index_zero_based=i,rgb_path=str(image),depth_path=str(image),gt_match_status='UNMATCHED' if s=='crowd2' and i%8==7 and i>=56 else 'MATCHED') for i in range(256)]
                csvwrite(p,rr);paths[s+'_pair_csv']=str(p);hashes[s]=sha(p)
            write(base/'protocol.json',dict(paths=paths));sig=sha(base/'protocol.json');write(base/'summary.json',dict(signature=sig,status='Y05R_COMPLETE_NOT_METHOD_PASS'))
            identity=dict(source_y05r_protocol_sha256=sig,Cstar_file_sha256=sha(cp),original_source_sha256={},reviewed_manifest_sha256=hashes)
            out=root/'outputs/gt4a2_n14av2_fixture/validation'
            oldroot,oldcmroot,oldread,oldgpu,oldargv=runner.ROOT,cm.ROOT,runner.read,runner.gpu_lock_path,sys.argv
            runner.ROOT=cm.ROOT=root;runner.read=lambda p:identity if Path(p).name=='identity.json' else oldread(p)
            runner.gpu_lock_path=lambda:(_ for _ in ()).throw(AssertionError('GPU touched before G0'))
            sys.argv=['runner','--y05r-dir',str(base),'--out-dir',str(out)]
            try:
                with contextlib.redirect_stdout(io.StringIO()):rc=runner.main()
                self.assertEqual(rc,2);self.assertEqual(read(out/'summary.json')['status'],'COVERAGE_BLOCKED')
                self.assertTrue((out/'n14av2_coverage_review.zip').is_file());self.assertFalse((out/'training').exists());self.assertFalse((out/'protocol.json').exists())
            finally:runner.ROOT=oldroot;cm.ROOT=oldcmroot;runner.read=oldread;runner.gpu_lock_path=oldgpu;sys.argv=oldargv
    def test_full_quality_timing_state_phase_report_resume_and_tamper_CPU_fixture(self):
        import numpy as np
        import sys,types
        from . import evaluation as ev
        from .state_audit import audit_all,FIELDS
        p=read(HERE/'policy.json');sequences=p['old_sequences']+p['development_sequences']
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);rgb=out/'rgb.png';rgb.write_bytes(b'CPU test bytes; no real image decoding')
            manifests={s:[dict(rgb_path=str(rgb),frame_index_zero_based=i,gt_match_status='UNMATCHED' if i%5==0 else 'MATCHED') for i in range(24)] for s in sequences}
            ctx=types.SimpleNamespace(out=out,signature='CPU-fixture',policy=p|dict(tail_start={s:4 for s in p['old_sequences']}),old_sequences=p['old_sequences'],manifests=manifests,
                rt=types.SimpleNamespace(np=np,torch=types.SimpleNamespace(device=lambda x:x,cuda=types.SimpleNamespace(synchronize=lambda:None)),exp1=types.SimpleNamespace(load_frame=lambda *a,**k:{'gt_np':np.ones((3,4),dtype='float32')})),
                plan=dict(prediction_retention='CPU fixture',coverage=[dict(passed=True)]*21))
            write(out/'smoke_checks.json',dict(passed=True));write(out/'gradient_audit.json',dict(passed=True))
            calls=[];values={'A':1.1,'C_star':1.05,'O_fast':1.08,'O_multi':1.03,'N14_A0':1.048,'N14_A1':1.04}
            class FakePredictor:
                mutate=False
                def __init__(self,c,m,s,diagnostics=False,audit=False):self.method=m;self.shape={};self.audit=audit
                def step(self,i,record):
                    calls.append((self.method,i));v=values[self.method]
                    if self.method in p['arms'] and phase(i)==0:v=values['C_star']
                    meta=dict(keyframe=i%32==0,age=i%32,phase=phase(i),micro_applied=self.method in p['arms'] and micro_due(i),residual_norm=0.)
                    if self.audit:
                        meta['state_hashes']={k:str(i)+k for k in FIELDS}
                        if self.mutate and self.method=='N14_A1':meta['state_hashes']['F0']='mutated'
                    return np.full((3,4),v,dtype='float32'),meta
                def reset(self):pass
                def close(self):pass
            oldpred,oldhere,oldbench=ev.Predictor,ev.HERE,sys.modules.get('gt4a2_y08cal_y09pareto');ev.Predictor=FakePredictor;ev.HERE=out/'no_reference_metrics'
            sys.modules['gt4a2_y08cal_y09pareto']=types.SimpleNamespace(gpu_endpoint=lambda rt:{'clean':True})
            try:
                with contextlib.redirect_stdout(io.StringIO()):state=audit_all(ctx);summary=ev.evaluate_all(ctx)
                self.assertTrue(state['passed']);self.assertEqual(len(state['records']),24);self.assertEqual(summary['quality_units'],48);self.assertEqual(summary['timing_trials'],144)
                self.assertFalse(summary['method_pass']);self.assertFalse(summary['screening_gate_passed']);self.assertFalse(summary['N14B_started'])
                phases=csvread(out/'phase_summary.csv');self.assertEqual(len(phases),6*8*11);count=len(calls)
                self.assertTrue((out/'nonharm_summary.csv').is_file());self.assertEqual(len(csvread(out/'paired_recording_bootstrap.csv')),12)
                with contextlib.redirect_stdout(io.StringIO()):audit_all(ctx);again=ev.evaluate_all(ctx)
                self.assertEqual(len(calls),count);self.assertEqual(summary,again)
                q=out/'quality/A/crowd2.csv';q.write_bytes(q.read_bytes()+b'changed')
                with self.assertRaises(RuntimeError):ev.quality(ctx,'A','crowd2')
                q=out/'timing/A/crowd2_r1.json';timing_record=read(q);timing_record['mean_ms']=999.;write(q,timing_record)
                with self.assertRaises(RuntimeError):ev.timing(ctx,'A','crowd2',1)
                endpoint_calls=[]
                def contaminated(rt):endpoint_calls.append(1);return {'clean':len(endpoint_calls)==1}
                sys.modules['gt4a2_y08cal_y09pareto']=types.SimpleNamespace(gpu_endpoint=contaminated)
                with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(RuntimeError):ev.timing(ctx,'O_multi','crowd2',4)
                self.assertFalse((out/'timing/O_multi/crowd2_r4.json').exists());self.assertTrue((out/'timing/O_multi/crowd2_r4.rejected.json').exists())
                for q in (out/'state_audit/N14_A1').glob('crowd2.*'):q.unlink()
                FakePredictor.mutate=True
                with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(RuntimeError):audit_all(ctx)
            finally:
                ev.Predictor=oldpred;ev.HERE=oldhere
                if oldbench is None:sys.modules.pop('gt4a2_y08cal_y09pareto',None)
                else:sys.modules['gt4a2_y08cal_y09pareto']=oldbench

@unittest.skipUnless(importlib.util.find_spec('torch'),'Torch unavailable locally; run operator/gradient tests in svrdepth')
class TorchTests(unittest.TestCase):
    def setUp(self):
        import torch
        from .models import PhaseMicroReadout
        self.t=torch;self.cls=PhaseMicroReadout;torch.manual_seed(20261031);torch.set_num_threads(2)
        self.features=[torch.randn(1,512,h,w) for h,w in [(8,10),(4,5),(3,3),(2,2)]];self.z=torch.randn(1,3,8,10)
    def test_parameter_count_and_zero_identity(self):
        m=self.cls();self.assertEqual(sum(p.numel() for p in m.parameters()),9252)
        for h in range(1,8):
            ys,reg=m(self.features,self.z,h,h);self.assertEqual(reg.item(),0);self.assertTrue(all(self.t.equal(x,y) for x,y in zip(self.features,ys)))
    def test_h0_bypasses_exactly(self):
        m=self.cls()
        for a in (0,8,16,24):
            ys,reg=m(self.features,None,0,a);self.assertIs(ys,self.features);self.assertEqual(reg.item(),0)
    def test_current_gradient_micro_only(self):
        m=self.cls();f=[x.clone().requires_grad_(True) for x in self.features];ys,reg=m(f,self.z,3,11);sum(x.square().mean() for x in ys).backward()
        self.assertTrue(all(x.grad is None for x in f));self.assertGreater(sum(x.grad.abs().sum().item() for x in m.parameters() if x.grad is not None),0)
    def test_A0_ignores_phase_values(self):
        m=self.cls(False)
        with self.t.no_grad():m.phase_mlp[-1].weight.normal_(0,.01)
        a,_=m(self.features,self.z,1,1);b,_=m(self.features,self.z,7,31);self.assertTrue(all(self.t.equal(x,y) for x,y in zip(a,b)))
    def test_A1_responds_to_phase_after_nonzero_weights(self):
        m=self.cls(True)
        with self.t.no_grad():m.phase_mlp[-1].weight.normal_(0,.01)
        a,_=m(self.features,self.z,1,1);b,_=m(self.features,self.z,7,31);self.assertTrue(any(not self.t.equal(x,y) for x,y in zip(a,b)))
    def test_forward_never_modifies_feature_storage(self):
        m=self.cls();copies=[x.clone() for x in self.features];versions=[x._version for x in self.features];m(self.features,self.z,2,18)
        self.assertEqual(versions,[x._version for x in self.features]);self.assertTrue(all(self.t.equal(x,y) for x,y in zip(copies,self.features)))
    def test_explicit_age_channel_rejected(self):
        with self.assertRaises(ValueError):self.cls()(self.features,self.t.randn(1,4,8,10),2,2)
    def test_wrong_feature_channels_rejected(self):
        with self.assertRaises(ValueError):self.cls()([x[:,:256] for x in self.features],self.z,2,2)
    def test_invalid_prediction_cannot_shrink_GT_mask(self):
        from .runtime import loss
        with self.assertRaises(RuntimeError):loss(self.t,self.t.tensor([float('nan'),1.]),self.t.ones(2))

if __name__=='__main__':unittest.main(verbosity=2)
