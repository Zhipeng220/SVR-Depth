"""CPU contracts and real orchestration regression; real model checks run in server smoke."""
from y11_common import *
from y11_data import associate,timestamp_rows,prior_exposure_audit
import y11_data
from types import SimpleNamespace,ModuleType
from contextlib import nullcontext,redirect_stdout
from tempfile import TemporaryDirectory
from unittest.mock import patch
import ast,copy,importlib.util,io,sys,unittest,zipfile
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('y11_actual_entry',HERE/'gt4a2_y11_fallback.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)

def old_function(name):
    reference=next(p for p in (HERE,HERE/'local_validation/reference_y09') if (p/'gt4a2_y08cal_y09pareto.py').is_file())
    path=reference/'gt4a2_y08cal_y09pareto.py'
    identity=jread(HERE/'y11_identity.json')
    assert sha(path)==identity['source_sha256']['experiments/p1/gt4a2_y08cal_y09pareto.py']
    tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    return compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec')

class Parameter:
    def __init__(self):self.requires_grad=True
    def requires_grad_(self,value):self.requires_grad=value

class Head:
    def __init__(self,**kwargs):self.kwargs=kwargs;self.p=Parameter();self.calls=0;self.active=True
    def cuda(self):return self
    def parameters(self):return [self.p]
    def eval(self):return self
    def load_state_dict(self,value,strict=True):self.loaded=value;assert strict

class Gain(Head):
    def __init__(self,head,gain):super().__init__();self.head=head;self.gain=gain

class Tensor:
    def __init__(self,value):self.value=np.array(value,dtype=np.float32)
    def detach(self):return self
    def float(self):return self
    def cpu(self):return self
    def numpy(self):return self.value

class DataTests(unittest.TestCase):
    def indexed_fixture(self,root,absent_source_depth=False,absent_source_RGB=False):
        seq='fixture';folder=root/'rgbd_bonn_fixture';folder.mkdir()
        rgb=b'1.000 rgb/1.png\n1.030 rgb/2.png\n';depth=b'1.001 depth/1.png\n1.031 depth/2.png\n'
        files={'rgb.txt':rgb,'depth.txt':depth,'rgb/1.png':b'RGB1','rgb/2.png':b'RGB2','depth/2.png':b'DEPTH2'}
        if not absent_source_depth:files['depth/1.png']=b'DEPTH1'
        if absent_source_RGB:files.pop('rgb/1.png')
        archive=root/'y11_downloads/rgbd_bonn_fixture.zip';archive.parent.mkdir()
        with zipfile.ZipFile(archive,'w') as z:
            for name,value in files.items():z.writestr('rgbd_bonn_fixture/'+name,value)
        for name,value in files.items():
            path=folder/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
        return folder,archive,jread(HERE/'y11_policy.json')

    def test_local_missing_depth_is_restored_from_matching_CRC_verified_ZIP(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);folder,archive,policy=self.indexed_fixture(root)
            (folder/'depth/1.png').unlink();before=(folder/'depth.txt').read_bytes()
            with redirect_stdout(io.StringIO()):audit=y11_data.audit_indexed_files(folder,'fixture',root,policy)
            self.assertEqual((folder/'depth/1.png').read_bytes(),b'DEPTH1')
            self.assertEqual(audit['source_proven_missing_depth'],[]);self.assertEqual(len(audit['restored_from_ZIP']),1)
            self.assertTrue(audit['source_zip_CRC_verified']);self.assertEqual((folder/'depth.txt').read_bytes(),before)

    def test_source_absent_depth_is_explicitly_unavailable_without_RGB_drop(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);folder,archive,policy=self.indexed_fixture(root,absent_source_depth=True)
            with redirect_stdout(io.StringIO()):audit=y11_data.audit_indexed_files(folder,'fixture',root,policy)
            self.assertEqual(audit['source_proven_missing_depth'],[str(folder/'depth/1.png')])
            rgb=timestamp_rows(folder/'rgb.txt',folder);depth=timestamp_rows(folder/'depth.txt',folder,allow_missing=True)
            rows=associate(rgb,depth,unavailable_depth_paths=audit['source_proven_missing_depth'])
            self.assertEqual(len(rows),2);self.assertEqual(rows[0]['gt_match_status'],'MISSING_DEPTH_FILE')
            self.assertEqual(rows[0]['depth_path'],str(folder/'depth/1.png'));self.assertEqual(rows[1]['gt_match_status'],'MATCHED')

    def test_missing_GT_keeps_full_index_pairing_not_nearest_available_reassignment(self):
        rgb=[(1.,Path('r0')),(1.020,Path('r1'))]
        depth=[(1.001,Path('missing')),(1.004,Path('available-but-not-replacement')),(1.021,Path('d1'))]
        rows=associate(rgb,depth,unavailable_depth_paths=['missing'])
        self.assertEqual(rows[0]['gt_match_status'],'MISSING_DEPTH_FILE')
        self.assertEqual(rows[0]['depth_path'],'missing');self.assertEqual(rows[1]['depth_path'],'d1')

    def test_missing_GT_without_source_proof_is_not_silently_omitted(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);folder,archive,policy=self.indexed_fixture(root,absent_source_depth=True);archive.unlink()
            with self.assertRaisesRegex(RuntimeError,'original ZIP is unavailable'):
                y11_data.audit_indexed_files(folder,'fixture',root,policy,download=False)

    def test_source_index_mismatch_prevents_GT_missing_interpretation(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);folder,archive,policy=self.indexed_fixture(root,absent_source_depth=True)
            p=folder/'depth.txt';p.write_text(p.read_text().replace('1.001','1.002'));before=p.read_bytes()
            with redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'do not uniquely match'):
                y11_data.audit_indexed_files(folder,'fixture',root,policy)
            self.assertEqual(p.read_bytes(),before)

    def test_source_missing_RGB_still_stops_complete_trajectory(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);folder,archive,policy=self.indexed_fixture(root,absent_source_RGB=True)
            with redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'lacks indexed RGB'):
                y11_data.audit_indexed_files(folder,'fixture',root,policy)

    def test_association_retains_missing_GT_and_never_reuses_depth(self):
        rgb=[(0.,Path('r0')),(.01,Path('r1')),(.03,Path('r2')),(1.,Path('r3'))]
        depth=[(.009,Path('d0')),(.031,Path('d1'))]
        rows=associate(rgb,depth)
        self.assertEqual(len(rows),4)
        self.assertEqual([r['gt_match_status'] for r in rows],['UNMATCHED','MATCHED','MATCHED','UNMATCHED'])
        self.assertEqual(len({r['depth_path'] for r in rows if r['gt_match_status']=='MATCHED'}),2)

    def test_association_uses_global_distance_order_and_strict_float_limit(self):
        rows=associate([(0.,Path('a')),(.018,Path('b'))],[(.019,Path('d'))])
        self.assertEqual(rows[1]['gt_match_status'],'MATCHED')
        self.assertTrue(all(float(r['abs_delta_ms'])<=20 for r in rows if r['gt_match_status']=='MATCHED'))

    def test_timestamp_duplicates_and_path_escape_rejected(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'r').write_bytes(b'a');f=p/'rgb.txt'
            f.write_text('1 r\n1 r\n')
            with self.assertRaisesRegex(RuntimeError,'Timestamp'):timestamp_rows(f,p)
            f.write_text('1 ../outside\n')
            with self.assertRaisesRegex(RuntimeError,'unsafe'):timestamp_rows(f,p)

    def test_exposure_audit_detects_sequence_lists_and_CSV_records(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'outputs';out.mkdir()
            jwrite(out/'old/protocol.json',{'sequences':['new_clip']})
            cwrite(out/'old/results.csv',[dict(sequence='new_clip',AbsRel=.1)])
            a=prior_exposure_audit(root,['new_clip'],out/'y11_confirmation_data')
            self.assertEqual(len(a['conflicts']),2)

    def test_frozen_policy_excludes_all_five_old_recordings_and_sealed_data(self):
        p=jread(HERE/'y11_policy.json')
        self.assertEqual(p['known_development_sequences'],['crowd2','crowd3','balloon2','person_tracking','synchronous'])
        self.assertFalse(set(p['confirmation_sequences'])&set(p['known_development_sequences']))
        self.assertFalse(set(p['confirmation_sequences'])&set(p['sealed_future_sequences']))
        self.assertEqual(len(p['confirmation_sequences']),5)

    def test_actual_data_prepare_freezes_and_rejects_later_changes(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'datasets';out=root/'outputs/y11_confirmation_data';old=root/'outputs/old/validation'
            jwrite(old/'protocol.json',{'input_sha256':{'/old/rgb/one.png':'not-a-new-image'}})
            policy=jread(HERE/'y11_policy.json')
            for seq in policy['confirmation_sequences']:
                folder=data/f'rgbd_bonn_{seq}';folder.mkdir(parents=True);rgb=[];depth=[]
                for i in range(33):
                    name=f'r{i}.png';(folder/name).write_bytes(f'RGB {seq} {i}'.encode());rgb.append(f'{i/30:.8f} {name}')
                    if i!=5:
                        name=f'd{i}.png';(folder/name).write_bytes(f'GT {seq} {i}'.encode());depth.append(f'{i/30:.8f} {name}')
                (folder/'rgb.txt').write_text('\n'.join(rgb)+'\n');(folder/'depth.txt').write_text('\n'.join(depth)+'\n')
            with patch.object(y11_data,'source_y09',return_value=old),redirect_stdout(io.StringIO()):
                path=y11_data.prepare(root,data,out);before=path.read_bytes()
                self.assertEqual(y11_data.prepare(root,data,out).read_bytes(),before)
                obj=jread(path);self.assertEqual(sum(r['RGB'] for r in obj['manifests'].values()),165)
                self.assertEqual(sum(r['matched_GT'] for r in obj['manifests'].values()),160)
                first=next(iter(obj['manifests'].values()));victim=Path(cread(first['path'])[0]['rgb_path']);victim.write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError,'Frozen data changed'):y11_data.prepare(root,data,out)
            self.assertEqual(before,path.read_bytes())

    def test_download_resume_keeps_existing_prefix_and_validates_ZIP(self):
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w') as z:z.writestr('fixture.txt','payload')
        payload=buffer.getvalue()
        class Response(io.BytesIO):
            status=206
            headers={'Content-Range':f'bytes 10-{len(payload)-1}/{len(payload)}'}
        with TemporaryDirectory() as tmp:
            target=Path(tmp)/'data.zip';target.with_suffix('.zip.part').write_bytes(payload[:10])
            with patch.object(y11_data.urllib.request,'urlopen',return_value=Response(payload[10:])):
                y11_data.download_archive('https://example.invalid/data.zip',target)
            self.assertEqual(target.read_bytes(),payload)

class MetricTests(unittest.TestCase):
    def test_invalid_prediction_cannot_shrink_GT_mask(self):
        gt=np.array([[1.,2.,0.]],dtype=np.float32);p=np.array([[1.,np.nan,0.]])
        with self.assertRaisesRegex(RuntimeError,'fixed GT mask'):metric(np,p,gt)
        m=metric(np,np.array([[1.,2.,np.nan]]),gt)
        self.assertEqual(m['valid_pixels'],2);self.assertEqual(m['AbsRel'],0.)

    def test_empty_GT_mask_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'empty'):metric(np,np.ones((2,2)),np.zeros((2,2)))

class SourceInterfaceTests(unittest.TestCase):
    def source_fixture(self,corrupt=None):
        with TemporaryDirectory() as tmp:
            y09=Path(tmp);jwrite(y09/'protocol.json',{'old':'identity'});signature=sha(y09/'protocol.json')
            obj={'signature':signature,'state':{'weight':'known'}}
            if corrupt=='signature':obj['signature']='new-diagnostic-signature'
            if corrupt=='state':obj['state']={'weight':'foreign'}
            calls=[]
            def load(path,map_location,weights_only):
                calls.append((path,map_location,weights_only));return obj
            rt=SimpleNamespace(torch=SimpleNamespace(load=load,isfinite=lambda x:SimpleNamespace(all=lambda:SimpleNamespace(item=lambda:True))),exp1=SimpleNamespace(state_sha256=lambda x:x['weight']))
            identity=dict(y09_protocol_sha256=signature,O_fast_tensor_sha256='known')
            result=entry.source_weights(rt,y09,{'weight':'P'},{'weight':'O'},identity)
            self.assertEqual(result[0]['O_fast'],{'weight':'known'});self.assertEqual(calls[0][1:],('cpu',False))

    def test_old_O_fast_signature_is_used_before_new_output_signature(self):self.source_fixture()
    def test_foreign_source_signature_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'source checkpoint signature'):self.source_fixture('signature')
    def test_foreign_source_tensor_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'tensor identity'):self.source_fixture('state')

    def test_actual_original_model_factory_all_four_interfaces(self):
        models=ModuleType('y08y09_models');models.GainHead=Gain;models.ChannelAffine=Head;models.FastDepthAdapter=Head
        repairs=ModuleType('velodepth.models.persistent_feature_repair');repairs.PersistentFeatureRepairHead=Head
        def load_model(snapshot,mode,device):
            model=Head();model.mode=mode;model.pixel_decoder=SimpleNamespace(persistent_feature_repair=Head());return model
        rt=SimpleNamespace(gt2=SimpleNamespace(load_model=load_model),torch=SimpleNamespace(device=lambda x:x))
        env={'Path':Path};exec(old_function('make_model'),env)
        with patch.dict(sys.modules,{'y08y09_models':models,'velodepth.models.persistent_feature_repair':repairs}):
            for method in METHODS:
                m,a=env['make_model'](SimpleNamespace(snapshot='snapshot'),rt,config(method),{'P':1},{'O':1},{'O_fast':{'fast':1}})
                self.assertEqual(m.mode,'pre_readout' if method=='P' else 'off')
                self.assertTrue(all(not p.requires_grad for p in m.parameters()))
                if method=='P':self.assertEqual(m.pixel_decoder.persistent_feature_repair.head.loaded,{'P':1})
                if method=='O_multi':self.assertEqual(a.kwargs,dict(channels=512,bottleneck=32,levels=4));self.assertEqual(a.loaded,{'O':1})
                if method=='O_fast':self.assertEqual(a.loaded,{'fast':1})
                if a:self.assertTrue(all(not p.requires_grad for p in a.parameters()))

    def test_GPU_lock_is_resolved_without_initializing_Torch(self):
        result=SimpleNamespace(stdout='0, GPU-abc\n1, GPU-def\n')
        with patch.object(entry.shutil,'which',return_value='/nvidia-smi'),patch.object(entry.subprocess,'run',return_value=result),patch.dict(entry.os.environ,{'CUDA_VISIBLE_DEVICES':'1'}):
            uuid,path=entry.gpu_lock_path()
        self.assertEqual(uuid,'def');self.assertIn(hashlib.sha256(b'def').hexdigest()[:16],str(path))

    def test_original_Stream_step_schedule_and_output_only_contract(self):
        reference=next(p for p in (HERE,HERE/'local_validation/reference_y09') if (p/'y08y09_models.py').is_file())
        path=reference/'y08y09_models.py';identity=jread(HERE/'y11_identity.json')
        self.assertEqual(sha(path),identity['source_sha256']['experiments/p1/y08y09_models.py'])
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Stream')
        fake_torch=SimpleNamespace(autocast=lambda *a,**k:nullcontext(),device=lambda x:x,float16='half')
        env=dict(torch=fake_torch,require=require,repair_due=lambda age,freq:(age-1)%freq==0,
            transition=lambda *a:{},output_depth=lambda rt,m,state,shape:state[0],rgb_for_output=lambda *a:None,
            versions=lambda state:(state[0].value.tobytes(),tuple(state[2])),readout_radius=lambda m,f:Tensor([[sum(f)]]),F=None,GainHead=Gain)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),env)
        Stream=env['Stream']
        def fast(self,frame,static,apply):
            if apply:self.model.pixel_decoder.persistent_feature_repair.calls+=1
            self.counts['native_fast_calls']+=1
            self.state=(Tensor([[2.+frame['index']*.001]]),0,[1.,2.,3.,4.],0)
            return None,None
        Stream.fast=fast
        rt=SimpleNamespace(cv2=SimpleNamespace(DISOpticalFlow_create=lambda x:None,DISOPTICAL_FLOW_PRESET_FAST=0),
            exp1=SimpleNamespace(keyframe_state=lambda m,f,s:(0,Tensor([[2.]]),0,[1.,2.,3.,4.]),load_frame=lambda rec,*a,**k:rec),
            gt2=SimpleNamespace(make_z_depth=lambda r,*a:r))
        shape=dict(padded_shape=(1,1),paddings=(0,0,0,0))
        for method in METHODS:
            m=SimpleNamespace(pixel_decoder=SimpleNamespace(persistent_feature_repair=SimpleNamespace(calls=0)),interpolation_mode='bilinear')
            adapter=(lambda pred,rgb,gain:Tensor(pred.value*.9)) if method=='O_fast' else (lambda features:[x+.1 for x in features])
            stream=Stream(rt,m,config(method),shape,adapter,debug=True)
            for i in range(33):
                pred,meta=stream.step(i,{'index':i})
                self.assertEqual(meta['keyframe'],i%16==0);self.assertEqual(meta['age'],i%16)
                if method.startswith('O_') and i%16:
                    self.assertEqual(stream.state[2],[1.,2.,3.,4.])
                    self.assertFalse(np.array_equal(pred,stream.native_prediction.numpy()))
            self.assertEqual(stream.counts,expected_counts(method,33))

    def test_actual_preflight_CLI_creates_new_signature_after_source_loading(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'gt4a2_y11_fallback_fixture/preflight'
            ctx=SimpleNamespace(plan=dict(experiment='Y11-Fallback',environment={'uuid':'abc'},input_sha256={}))
            with patch.object(entry,'prepare',return_value=ctx),patch.object(entry,'gpu_lock_path',return_value=('abc',root/'gpu.lock')),redirect_stdout(io.StringIO()):
                self.assertEqual(entry.main(['--snapshot','fixture','--preflight-only','--out-dir',str(out)]),0)
            self.assertEqual(ctx.out,out);self.assertEqual(ctx.signature,sha(out/'protocol.json'))

class FakeStream:
    def __init__(self,method):self.method=method;self.reset()
    def reset(self):self.counts={k:0 for k in expected_counts(self.method,0)}
    def step(self,index,record):
        key=index%16==0
        if key:self.counts['keyframe_calls']+=1
        else:
            self.counts['flow_calls']+=1;self.counts['native_fast_calls']+=1
            if self.method=='P':self.counts['repair_head_calls']+=1
            if self.method.startswith('O_'):self.counts['output_adapter_calls']+=1
            if self.method=='O_multi':self.counts['additional_readout_passes']+=1
        native=np.full((2,2),2.+index*.001,dtype=np.float32)
        self.native_prediction=Tensor(native)
        pred=native if self.method=='A' or key else native*{'O_fast':.9,'O_multi':.8,'P':.85}[self.method]
        return pred,dict(keyframe=key,age=index%16,repair_requested=not key and self.method=='P',output_adapter_applied=not key and self.method.startswith('O_'))

class OrchestrationTests(unittest.TestCase):
    def test_real_quality_timing_report_archive_resume_and_tamper_checks(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.mkdir();policy=jread(HERE/'y11_policy.json');policy['bootstrap_repeats']=20
            manifests={}
            for seq in policy['confirmation_sequences']:
                rows=[]
                for i in range(33):
                    rgb=root/seq/f'{i}.rgb';rgb.parent.mkdir(parents=True,exist_ok=True);rgb.write_bytes(f'{seq} {i}'.encode())
                    rows.append(dict(frame_index_zero_based=i,rgb_timestamp=i/30,rgb_path=str(rgb),depth_path=str(root/'gt.png'),gt_match_status='MISSING_DEPTH_FILE' if i==5 else 'MATCHED'))
                manifests[seq]=rows
            cuda=SimpleNamespace(empty_cache=lambda:None,synchronize=lambda:None,reset_peak_memory_stats=lambda:None,max_memory_allocated=lambda:100)
            rt=SimpleNamespace(np=np,torch=SimpleNamespace(no_grad=nullcontext,cuda=cuda),cv2=SimpleNamespace(IMREAD_UNCHANGED=-1,imread=lambda *a:np.full((2,2),5000,dtype=np.uint16)))
            ctx=SimpleNamespace(rt=rt,manifests=manifests,out=out,policy=policy,signature='fixture',old=SimpleNamespace(gpu_endpoint=lambda rt:{'clean':True}),identity={'source_sha256':{}},plan={'paths':{}})
            def factory(ctx,method,records,debug):return FakeStream(method),None,None,{'original_shape':(2,2)}
            (root/'run_y11.sh').write_text('fixture')
            with patch.object(entry,'new_stream',factory),patch.object(entry,'ROOT',root),redirect_stdout(io.StringIO()):
                for method in METHODS:
                    for seq in policy['confirmation_sequences']:entry.quality_unit(ctx,method,seq)
                first=entry.quality_path(out,'P',policy['confirmation_sequences'][0]);stamp=first.stat().st_mtime_ns
                entry.quality_unit(ctx,'P',policy['confirmation_sequences'][0]);self.assertEqual(stamp,first.stat().st_mtime_ns)
                for method in METHODS:
                    for seq in policy['confirmation_sequences']:
                        for repeat in range(3):entry.timing_unit(ctx,method,seq,repeat,0)
                entry.report(ctx);archive=entry.archive(ctx)
            summary=jread(out/'summary.json');self.assertEqual(summary['quality_RGB_records'],660);self.assertEqual(summary['timing_trials'],60)
            self.assertFalse(summary['Y11_pass']);self.assertEqual(summary['inherited_training_seeds'],1)
            rr=cread(out/'y11_frame_metrics.csv');self.assertEqual(sum(not truth(r['matched']) for r in rr),20)
            self.assertEqual(sum(r['gt_match_status']=='MISSING_DEPTH_FILE' for r in rr),20)
            self.assertEqual(sum(1 for _ in (out/'predictions').rglob('*.npz')),660)
            with zipfile.ZipFile(archive) as z:self.assertIsNone(z.testzip());self.assertFalse(any(n.endswith('.npz') for n in z.namelist()))
            first.write_bytes(first.read_bytes()+b'corruption')
            with self.assertRaisesRegex(RuntimeError,'Quality unit changed'):entry.load_quality(ctx,'P',policy['confirmation_sequences'][0])

    def test_unverified_timing_does_not_enter_Pareto_or_decision(self):
        p=jread(HERE/'y11_policy.json');p['confirmation_sequences']=['v'];p['bootstrap_repeats']=2
        frames=[dict(method=m,sequence='v',index=0,matched=True,valid_pixels=1,mask_sha256='x',AbsRel=.1,RMSE=.1,delta1=.9) for m in METHODS]
        trials=[dict(method=m,sequence='v',endpoint_clean=False,quality_checked=True,whole_mean_ms=1.,whole_p95_ms=1.,peak_allocated_gib=1.) for m in METHODS for _ in range(3)]
        with self.assertRaisesRegex(RuntimeError,'Unverified timing'):aggregate(frames,trials,p)

    def test_macro_equal_recordings_not_frame_count_and_GT_selection_guard(self):
        p=jread(HERE/'y11_policy.json');p['confirmation_sequences']=['short','long'];p['bootstrap_repeats']=10
        frames=[];trials=[]
        for method in METHODS:
            for seq,n in [('short',1),('long',9)]:
                value=(.1 if seq=='short' else .9) if method=='P' else 1.
                for i in range(n):frames.append(dict(method=method,sequence=seq,index=i,matched=True,valid_pixels=1,mask_sha256='x',AbsRel=value,RMSE=value,delta1=.9))
                for _ in range(3):trials.append(dict(method=method,sequence=seq,endpoint_clean=True,quality_checked=True,whole_mean_ms=1.,whole_p95_ms=1.,peak_allocated_gib=1.))
        sequence,macro,*_=aggregate(frames,trials,p)
        self.assertAlmostEqual(next(r['AbsRel'] for r in macro if r['method']=='P'),.5)
        frames[-1]['matched']=False
        with self.assertRaisesRegex(RuntimeError,'GT selection differs'):aggregate(frames,trials,p)

if __name__=='__main__':unittest.main(verbosity=2)
