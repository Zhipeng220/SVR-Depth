"""Crash recovery, paired statistics, sample exposures and v2 operator regressions."""
from .common import *
import ast,contextlib,importlib.util,io,tempfile,types,unittest
from unittest import mock

class CacheRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'run/validation';self.root.mkdir(parents=True)
        self.policy=read(HERE/'policy.json');self.ctx=types.SimpleNamespace(out=self.root,signature='v2',policy=self.policy,plan={'Cstar_file_sha256':'frozen-source'})
        self.cases=[dict(id=f'crowd2_{i:06d}',sequence='crowd2',index=i,phase=phase(i),age32=i%32) for i in (1,2,3)]
    def pair(self,c):
        p=self.root/'cache'/c['sequence']/f'{c["index"]:06d}.pt';atomic_bytes(p,b'reproducible test Tensor bytes')
        m=dict(signature='v2',case_id=c['id'],phase=c['phase'],age32=c['age32'],path=str(p),Cstar_checkpoint_sha256='frozen-source',file_sha256=sha(p))
        write(p.with_suffix('.json'),m);return p
    def recover(self):
        from .cache_integrity import repair_cache
        with contextlib.redirect_stdout(io.StringIO()):return repair_cache(self.ctx,self.cases)
    def test_good_pairs_retained_and_source_checkpoint_untouched(self):
        pp=[self.pair(c) for c in self.cases];source=self.root/'training/N14_A0/recovery.pt';atomic_bytes(source,b'optimizer checkpoint')
        before=source.read_bytes();self.assertEqual(len(self.recover()),3);self.assertEqual(source.read_bytes(),before);self.assertTrue(all(p.is_file() for p in pp))
    def test_zero_JSON_pair_and_completion_are_quarantined(self):
        good=self.pair(self.cases[0]);bad=self.pair(self.cases[1]);bad.with_suffix('.json').write_bytes(b'')
        complete=bad.parent/'complete.json';write(complete,dict(signature='v2',indices=[1,2,3]))
        self.assertEqual(len(self.recover()),1);self.assertTrue(good.is_file());self.assertFalse(bad.exists());self.assertFalse(complete.exists())
        m=read(self.root/'cache_recovery.json');self.assertEqual(m['quarantined_files'],3);self.assertEqual(len(list(Path(m['backup']).rglob('000002.pt'))),1)
    def test_missing_metadata_and_truncated_tensor_are_rebuilt(self):
        p=self.pair(self.cases[0]);p.with_suffix('.json').unlink();q=self.pair(self.cases[1]);q.write_bytes(b'truncated')
        self.assertEqual(self.recover(),{});self.assertFalse(p.exists());self.assertFalse(q.exists())
    def test_foreign_signature_aborts_before_any_moves(self):
        p=self.pair(self.cases[0]);p.with_suffix('.json').write_bytes(b'');q=self.pair(self.cases[1]);m=read(q.with_suffix('.json'));m['signature']='v1';write(q.with_suffix('.json'),m)
        with self.assertRaisesRegex(RuntimeError,'Foreign cache signature'):self.recover()
        self.assertTrue(p.is_file());self.assertTrue(q.is_file());self.assertFalse(list(self.root.parent.glob('cache_quarantine_*')))
    def test_uncommitted_temporary_files_are_quarantined(self):
        p=self.pair(self.cases[0]);tmp=p.with_name('000002.pt.tmp');tmp.write_bytes(b'partial')
        self.recover();self.assertTrue(p.exists());self.assertFalse(tmp.exists())
    def test_failed_replace_keeps_previous_JSON_readable(self):
        from . import common
        p=self.root/'metadata.json';write(p,{'version':'old'})
        with mock.patch.object(common.os,'replace',side_effect=OSError('simulated interruption')):
            with self.assertRaises(OSError):write(p,{'version':'new'})
        self.assertEqual(read(p),{'version':'old'})
    def test_metadata_flushes_file_and_directory(self):
        from . import common
        with mock.patch.object(common.os,'fsync',wraps=os.fsync) as fsync:write(self.root/'x.json',{'committed':True})
        self.assertEqual(fsync.call_count,2)
    def test_full_force_verification_rejects_same_stat_tampering(self):
        from .training import verify_cache
        p=self.pair(self.cases[0]);verify_cache(self.ctx,self.cases[0]);s=p.stat();data=p.read_bytes();p.write_bytes(b'X'+data[1:]);os.utime(p,ns=(s.st_atime_ns,s.st_mtime_ns))
        with self.assertRaises(RuntimeError):verify_cache(self.ctx,self.cases[0],force=True)

class SamplingStatisticsTests(unittest.TestCase):
    def setUp(self):self.p=read(HERE/'policy.json')
    def test_1000_updates_are_1000_not_7000_GT_exposures(self):
        cases,cov=coverage_plan({s:[dict(gt_match_status='MATCHED') for i in range(256)] for s in self.p['old_sequences']},self.p)
        order=order_for(cases,1000,self.p['seed'],self.p['old_sequences']);strata,frames=sampling_audit(cases,order)
        self.assertEqual(len(strata),21);self.assertEqual(sum(r['GT_frame_exposures'] for r in strata),1000);self.assertEqual(sum(r['exposures'] for r in frames),1000)
        self.assertTrue(all(r['min_repeat']==1 and r['max_repeat']==2 for r in strata));self.assertTrue(all(r['distinct_used']==32 for r in strata))
    def test_frame_types_partition_period_with_separate_h0_sources(self):
        types_=[frame_type(i) for i in range(32)]
        self.assertEqual(types_.count('K32_reset'),1);self.assertEqual(types_.count('S8_pulse'),3);self.assertEqual(types_.count('propagation'),28)
    def test_reference_phase_numbers_reproduce_reviewed_N13(self):
        from .reference_audit import reference_phase_audit
        with tempfile.TemporaryDirectory() as d:
            _,rr=reference_phase_audit(Path(d),self.p);by={(r['group'],r['phase']):r for r in rr}
            for key,want in [(('old_tail',1),.055763),(('old_tail',7),.061763),(('Y11_development',1),.070807),(('Y11_development',7),.072447)]:self.assertAlmostEqual(by[key]['AbsRel'],want,places=6)
    def test_paired_recording_bootstrap_deterministic_and_constant_delta(self):
        import numpy as np
        from .nonharm import paired_bootstrap
        a=paired_bootstrap(np,[.09,.19,.29],[.1,.2,.3]);b=paired_bootstrap(np,[.09,.19,.29],[.1,.2,.3])
        self.assertEqual(a,b);self.assertEqual(a['recordings'],3);self.assertAlmostEqual(a['delta_low95'],-.01);self.assertAlmostEqual(a['delta_high95'],-.01)
    def test_nonharm_macro_equal_recordings_and_Cstar_reference(self):
        from .nonharm import nonharm_report
        with tempfile.TemporaryDirectory() as d:
            ctx=types.SimpleNamespace(out=Path(d),policy=self.p,old_sequences=self.p['old_sequences']);rows=[]
            for seq,count,err in [('crowd2',100,.12),('crowd3',1,.08)]:
                for i in range(count):
                    for method,v in [('C_star',.10),('N14_A0',err),('N14_A1',err)]:rows.append(dict(method=method,sequence=seq,index=i,AbsRel=v,mask_sha256='fixed',valid_pixels=8))
            nonharm_report(ctx,rows);rr=csvread(Path(d)/'nonharm_macro_summary.csv');m=next(r for r in rr if r['method']=='N14_A1' and r['category']=='all')
            self.assertAlmostEqual(float(m['mean_delta_AbsRel']),0);self.assertEqual(float(m['worse_vs_Cstar_rate']),.5)
            rows[-1]['mask_sha256']='changed'
            with self.assertRaises(RuntimeError):nonharm_report(ctx,rows)
    def test_hinge_is_after_scalar_frame_reduction(self):
        tree=ast.parse((HERE/'runtime.py').read_text());f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='current_objective')
        self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='detach' for n in ast.walk(f)))
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='mean' for n in ast.walk(f)))
    def test_installed_namespace_does_not_import_v1(self):
        for p in HERE.glob('*.py'):
            tree=ast.parse(p.read_text())
            for n in ast.walk(tree):
                if isinstance(n,ast.ImportFrom):self.assertNotIn('n14a_suite',n.module or '')

@unittest.skipUnless(importlib.util.find_spec('torch'),'Torch unavailable locally; production svrdepth runs these tests before CUDA work')
class V2TorchTests(unittest.TestCase):
    def setUp(self):
        import torch
        from .models import PhaseMicroReadout
        self.t=torch;self.cls=PhaseMicroReadout;torch.manual_seed(20261031);torch.set_num_threads(2)
        self.f=[torch.ones(1,512,h,w) for h,w in [(4,4),(3,3),(2,2),(1,1)]];self.z=torch.zeros(1,3,4,4)
    def test_envelope_fixed_tanh_ratio_and_no_extra_parameter(self):
        m=self.cls();t=self.t
        with t.no_grad():m.phase_mlp[-1].bias[:512].fill_(.1)
        for h in range(1,8):
            y,_=m(self.f,self.z,h,h);self.assertAlmostEqual(float((y[0]-self.f[0]).mean()),.1*math.tanh(h/8),places=6)
        self.assertEqual(sum(p.numel() for p in m.parameters()),9252);self.assertFalse(any('envelope' in k for k in m.state_dict()))
    def test_identical_local_phase_uses_distinct_K32_age(self):
        m=self.cls()
        with self.t.no_grad():m.phase_mlp[0].weight.fill_(.1);m.phase_mlp[0].bias.zero_();m.phase_mlp[-1].weight.fill_(.01)
        a,_=m(self.f,self.z,1,1);b,_=m(self.f,self.z,1,25);self.assertFalse(self.t.equal(a[0],b[0]))
    def test_inconsistent_age_phase_rejected(self):
        with self.assertRaises(ValueError):self.cls()(self.f,self.z,1,8)
    def test_safe_hinge_gradient_doubles_only_for_worse_frame_and_detaches_reference(self):
        from .runtime import current_objective
        p=read(HERE/'policy.json');t=self.t
        for value,want in [(.2,2.),(.05,1.)]:
            current=t.tensor(value,requires_grad=True);reference=t.tensor(.1,requires_grad=True)
            objective,hinge=current_objective(t,current,reference,t.tensor(0.),p);objective.backward()
            self.assertEqual(float(current.grad),want);self.assertIsNone(reference.grad);self.assertAlmostEqual(float(hinge),max(0,value-.1),places=6)
    def test_mean_hinge_not_pixelwise_hinge(self):
        from .runtime import current_objective,loss
        t=self.t;p=read(HERE/'policy.json');gt=t.ones(2);pred=t.tensor([1.,1.15]);ref=t.tensor([1.1,1.1])
        _,hinge=current_objective(t,loss(t,pred,gt),loss(t,ref,gt),t.tensor(0.),p)
        self.assertEqual(float(hinge),0.);self.assertGreater(float(t.relu((pred-gt).abs()-(ref-gt).abs()).mean()),0.)

if __name__=='__main__':unittest.main(verbosity=2)
