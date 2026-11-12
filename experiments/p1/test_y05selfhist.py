"""State-history diagnosis contract checks; real CUDA canaries run in smoke."""
import unittest
from y05selfhist_common import *

class ContractTests(unittest.TestCase):
    def _prepare_with_original_loader(self, corruption=None):
        # Run actual prepare + the unmodified Y05 module/head_states function.
        # Only old.preflight and torch.load's tensor deserialization are faked.
        import importlib.util,sys,tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        here=Path(__file__).resolve().parent
        reference=next(p for p in (here,here/'local_validation/reference_y05') if (p/'y05_train.py').is_file())
        self.assertEqual(sha(reference/'y05_train.py'),'f5c2e25ea16981114d10a21239503ba2fdd062ef111d5ef77bcfcf5e12df37ed')
        spec=importlib.util.spec_from_file_location('selfhist_prepare_regression',here/'gt4a2_y05selfhist.py')
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
            (root/'run_y05selfhist.sh').write_text('fixture')
            jwrite(installed/'y05selfhist_identity.json',dict(source_summary_sha256=sha(source/'summary.json'),source_frame_metrics_sha256=sha(source/'y05_frame_metrics.csv'),source_protocol_sha256=source_signature,y05_source_sha256={},head_tensor_sha256={h:h for h in CARRIERS}))
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
                        self.assertEqual(result.selfhist_plan['paths']['y05_dir'],str(source))
                        self.assertEqual(result.selfhist_plan['source_y05_signature'],source_signature)
                        self.assertEqual(set(result.states),set(CARRIERS))
                        self.assertEqual(loaded,checkpoints)
                        # The real CLI must switch to its own output/signature
                        # only after the SOURCE loader successfully returns.
                        out=root/'outputs/gt4a2_y05selfhist_fixture/preflight'
                        argv=['gt4a2_y05selfhist.py','--preflight-only','--smoke-only','--out-dir',str(out)]
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

    def test_follow_carrier_after_anchor_not_candidate(self):
        s=schedule(8,'C_current','F_future')
        self.assertEqual([(r['h'],r['active_head']) for r in s if r['repair_applied']],[(0,'F_future'),(8,'C_current'),(16,'C_current')])

    def test_skip_only_anchor_not_all_future(self):
        s=schedule(8,'F_future','skip')
        self.assertFalse(s[0]['repair_applied'])
        self.assertEqual(s[8]['active_head'],'F_future')
        self.assertEqual(s[16]['active_head'],'F_future')

    def test_native_reset_precedes_repair(self):
        s=schedule(24,'FH_future_harm','C_current')
        self.assertFalse(s[8]['repair_applied'])
        self.assertTrue(s[8]['after_reset'])
        self.assertEqual(s[16]['active_head'],'FH_future_harm')

    def test_deduplicate_parent_origins_without_selecting_cases(self):
        rows=[]
        for seq in SEQS:
            for cycle in (256,288,320):
                for age in (8,16,24):
                    for origin in ('A','S8'):rows.append(dict(sequence=seq,anchor=cycle+age,age=age,K=32,origin=origin))
        self.assertEqual(len(anchors({'probe_plan':rows})),27)
        with self.assertRaises(RuntimeError):anchors({'probe_plan':rows+[rows[0]]})

    def test_smoke_covers_history_and_full_reset(self):
        cases=anchors({},True)
        self.assertEqual(len(cases),6)
        self.assertEqual({c['age'] for c in cases},{8,24})

    def fixture(self):
        by={}
        for action in ACTIONS:
            by[action]={}
            for spec in schedule(24,'C_current',action):
                h=spec['h'];matched=h!=2
                loss=1. if action=='skip' or spec['after_reset'] else .5
                by[action][h]=dict(**spec,matched=matched,valid_pixels=100,AbsRel=loss,RMSE=loss,delta1=.9)
        return by

    def test_current_excluded_GT_missing_still_full_trajectory(self):
        case=dict(case_id='case',sequence='crowd2',anchor=24,age=24)
        by=self.fixture();by['F_future'][0]['AbsRel']=1000.
        r=next(r for r in windows(case,'C_current',by) if r['action']=='F_future' and r['horizon']==4)
        self.assertEqual(r['n_GT'],3)
        self.assertEqual(r['mean_AbsRel'],.5)
        self.assertEqual(r['current_AbsRel'],1000.)

    def test_prediction_cannot_change_GT_mask(self):
        by=self.fixture();by['F_future'][1]['valid_pixels']=99
        with self.assertRaises(RuntimeError):windows(dict(case_id='x',sequence='crowd2',anchor=24,age=24),'C_current',by)

    def test_reset_erases_all_initial_actions(self):
        by=self.fixture();by['F_future'][8]['AbsRel']=.2
        with self.assertRaises(RuntimeError):windows(dict(case_id='x',sequence='crowd2',anchor=24,age=24),'C_current',by)

    def test_macro_equal_videos_not_frame_counts(self):
        rr=[]
        for carrier in CARRIERS:
            for seq,n,value in [('crowd2',1,1.),('crowd3',2,2.),('balloon2',3,3.)]:
                for i in range(n):
                    for action in ACTIONS:
                        for h in HORIZONS:rr.append(dict(carrier=carrier,sequence=seq,case_id=seq+str(i),action=action,horizon=h,mean_AbsRel=value,mean_gain=0.,harmed_fraction=0.))
        _,macro,_=aggregate(rr)
        self.assertTrue(all(r['macro_AbsRel']==2. for r in macro))

if __name__=='__main__':unittest.main(verbosity=2)
