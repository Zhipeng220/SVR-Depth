import unittest
import numpy as np
from types import SimpleNamespace
from y05dirprobe_common import *
from y05dirprobe_edit import *

class PrepareRegression(unittest.TestCase):
    def _prepare_with_original_loader(self, corruption=None):
        # Run actual prepare + the unmodified Y05 module/head_states function.
        # Only old.preflight and torch.load's tensor deserialization are faked.
        import importlib.util,sys,tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        here=Path(__file__).resolve().parent
        reference=next(p for p in (here,here/'local_validation/reference_y05') if (p/'y05_train.py').is_file())
        self.assertEqual(sha(reference/'y05_train.py'),'f5c2e25ea16981114d10a21239503ba2fdd062ef111d5ef77bcfcf5e12df37ed')
        spec=importlib.util.spec_from_file_location('dirprobe_prepare_regression',here/'gt4a2_y05dirprobe.py')
        main=importlib.util.module_from_spec(spec);spec.loader.exec_module(main)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);installed=root/'experiments/p1';installed.mkdir(parents=True)
            source=root/'outputs/old/validation';source.mkdir(parents=True)
            oldplan=dict(paths={},input_sha256={},environment={'gpu_name':'fixture','uuid':'fixture'})
            jwrite(source/'protocol.json',oldplan);source_signature=sha(source/'protocol.json')
            jwrite(source/'summary.json',{})
            cwrite(source/'y05_frame_metrics.csv',[dict(config_id='C_current_K32_S8_g1',sequence='crowd2',index=0)])
            jwrite(source/'training_summary.json',{});jwrite(source/'gradient_audit.json',{})
            loaded=[]
            arms=('C_current','F_future','FH_future_harm','P_closed_loop')
            checkpoints=[]
            for arm in arms:
                cwrite(source/'training'/arm/'curve.csv',[dict(step=1,cuda_sec=1.,step_wall_sec=1.)])
                for selection in (('equal_updates','equal_compute') if arm=='P_closed_loop' else ('final',)):
                    label='P_'+selection if arm=='P_closed_loop' else arm
                    cp=source/'training'/arm/(selection+'.pt')
                    jwrite(cp,dict(signature=source_signature,state=label,head_sha256=label,step=1000))
                    checkpoints.append(cp)
            if corruption:
                cp=checkpoints[0];obj=jread(cp);obj[corruption]='foreign-checkpoint';jwrite(cp,obj)
            before={p:p.read_bytes() for p in checkpoints}
            for name in INSTALLED:(installed/name).write_text('fixture')
            (root/'run_y05dirprobe.sh').write_text('fixture')
            jwrite(installed/'y05dirprobe_identity.json',dict(source_summary_sha256=sha(source/'summary.json'),source_frame_metrics_sha256=sha(source/'y05_frame_metrics.csv'),source_protocol_sha256=source_signature,y05_source_sha256={},head_tensor_sha256={h:h for h in CARRIERS}))
            def load(path,map_location,weights_only):
                self.assertEqual(map_location,'cpu');self.assertIs(weights_only,False)
                loaded.append(Path(path));return jread(path)
            ctx=SimpleNamespace(plan=oldplan,head_state='P_init',rt=SimpleNamespace(torch=SimpleNamespace(load=load),exp1=SimpleNamespace(state_sha256=lambda s:s)))
            self.assertFalse(hasattr(ctx,'out'));self.assertFalse(hasattr(ctx,'signature'))
            with patch.dict(sys.modules):
                modules={}
                for name in ('y05_common','y05_engine','y05_train'):
                    module_spec=importlib.util.spec_from_file_location(name,reference/(name+'.py'))
                    module=importlib.util.module_from_spec(module_spec);sys.modules[name]=module
                    module_spec.loader.exec_module(module);modules[name]=module
                modules['y05_common'].preflight=lambda args:ctx
                with patch.object(main,'HERE',installed),patch.object(main,'ROOT',root),patch.object(main,'source_dir',lambda _:source):
                    if corruption:
                        with self.assertRaisesRegex(RuntimeError,'Selected checkpoint identity differs'):
                            main.prepare(SimpleNamespace(y05_dir='auto',smoke_only=True))
                    else:
                        result=main.prepare(SimpleNamespace(y05_dir='auto',smoke_only=True))
                        self.assertEqual(result.out,source)
                        self.assertEqual(result.signature,source_signature)
                        self.assertEqual(result.dirprobe_plan['paths']['y05_dir'],str(source))
                        self.assertEqual(result.dirprobe_plan['source_y05_signature'],source_signature)
                        self.assertEqual(set(result.states),set(CARRIERS))
                        self.assertEqual(loaded,checkpoints)
                        # The real CLI must switch to its own output/signature
                        # only after the SOURCE loader successfully returns.
                        out=root/'outputs/gt4a2_y05dirprobe_fixture/preflight'
                        argv=['gt4a2_y05dirprobe.py','--preflight-only','--smoke-only','--out-dir',str(out)]
                        with patch.object(sys,'argv',argv):self.assertEqual(main.main(),0)
                        self.assertEqual(ctx.out,out)
                        self.assertEqual(ctx.signature,sha(out/'protocol.json'))
                        self.assertNotEqual(ctx.signature,source_signature)
                        self.assertEqual(jread(out/'protocol.json')['source_y05_signature'],source_signature)
                        self.assertFalse(list(out.rglob('*.pt')))
            self.assertEqual(before,{p:p.read_bytes() for p in checkpoints})

    def test_prepare_and_CLI_use_original_checkpoint_loader(self):
        self._prepare_with_original_loader()

    def test_prepare_rejects_foreign_checkpoint_signature(self):
        self._prepare_with_original_loader('signature')

    def test_prepare_rejects_changed_checkpoint_tensor_hash(self):
        self._prepare_with_original_loader('head_sha256')

class DirectionTests(unittest.TestCase):
    def setUp(self):
        self.ops=NumpyOps(np);rng=np.random.default_rng(42)
        self.f=rng.normal(size=(1,8,3,3)).astype('float32')
        self.u=rng.normal(size=self.f.shape).astype('float32')
        self.d=rng.normal(size=self.f.shape).astype('float32')*.1
    def test_decomposition_and_norm_bound(self):
        p,q,_=components(self.d,self.u,self.f,self.ops)
        np.testing.assert_allclose(p+q,self.d,atol=1e-7)
        np.testing.assert_allclose((q*self.u).sum(axis=1),0,atol=1e-6)
        self.assertLessEqual(float((p*p).sum()),float((self.d*self.d).sum())+1e-6)
    def test_zero_innovation(self):
        p,q,_=components(self.d,np.zeros_like(self.u),self.f,self.ops)
        np.testing.assert_array_equal(p,0);np.testing.assert_array_equal(q,self.d)
    def test_full_exact_identity(self):
        y=self.f+self.d;out,_=edit(self.f,y,self.f-self.u,'full',self.ops)
        self.assertIs(out,y)
    def test_controls_norm_match(self):
        y=self.f+self.d
        for mode in ('parallel','orthogonal'):
            _,a=edit(self.f,y,self.f-self.u,mode,self.ops)
            for control in ('norm_'+mode,'scrambled_'+mode):
                _,b=edit(self.f,y,self.f-self.u,control,self.ops)
                self.assertAlmostEqual(a['realized_energy'],b['realized_energy'],places=5)
    def test_scramble_preserves_norm(self):
        u=self.ops.scramble(self.u)
        np.testing.assert_allclose((u*u).sum(axis=1),(self.u*self.u).sum(axis=1),rtol=1e-6)
    def test_future_resumes_carrier(self):
        s=schedule(8,'P_init','orthogonal')
        self.assertEqual(s[0]['active_head'],'C_current')
        self.assertEqual(s[8]['active_head'],'P_init')
        self.assertEqual(s[16]['active_head'],'P_init')
        s=schedule(24,'C_current','skip')
        self.assertFalse(s[0]['repair_applied']);self.assertFalse(s[8]['repair_applied'])
        self.assertEqual(s[16]['active_head'],'C_current')
    def test_exact_native_warp_hook_and_future_restore(self):
        class Layer:
            def __init__(self):self.hooks=[]
            def register_forward_pre_hook(self,fn):
                self.hooks.append(fn);return SimpleNamespace(remove=lambda:self.hooks.remove(fn))
        class Head:
            def forward(self,fs):return [f+np.array([.1,.2,.3,.4],dtype='float32').reshape(1,4,1,1) for f in fs]
        layers=[Layer() for _ in range(4)];head=Head()
        engine=SimpleNamespace(head=head,model=SimpleNamespace(pixel_decoder=SimpleNamespace(radial_module=SimpleNamespace(fusion=layers))))
        editor=DirectionalEditor(engine,self.ops);fs=[np.ones((1,4,1,1),dtype='float32') for _ in range(4)]
        editor.configure('orthogonal',True)
        for layer in layers:
            warped=np.array([.9,1.,1.,1.],dtype='float32').reshape(1,4,1,1)
            for hook in layer.hooks:hook(layer,(warped,None))
            warped[:]=100
        result=head.forward(fs)
        self.assertAlmostEqual(float(result[0][0,0,0,0]),1.,places=6);self.assertEqual(len(editor.stats),4)
        editor.configure('full',False);result=head.forward(fs)
        self.assertAlmostEqual(float(result[0][0,0,0,0]),1.1,places=6)
        editor.close();self.assertTrue(all(not layer.hooks for layer in layers))
    def test_missing_native_warp_rejected(self):
        class Layer:
            def register_forward_pre_hook(self,fn):return SimpleNamespace(remove=lambda:None)
        head=SimpleNamespace(forward=lambda fs:fs)
        engine=SimpleNamespace(head=head,model=SimpleNamespace(pixel_decoder=SimpleNamespace(radial_module=SimpleNamespace(fusion=[Layer() for _ in range(4)]))))
        editor=DirectionalEditor(engine,self.ops);editor.configure('parallel',True)
        with self.assertRaises(RuntimeError):head.forward([self.f]*4)
        editor.close()

if __name__=='__main__':unittest.main(verbosity=2)

