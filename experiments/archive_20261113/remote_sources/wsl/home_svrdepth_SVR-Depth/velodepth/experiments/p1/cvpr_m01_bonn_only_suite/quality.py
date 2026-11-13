#!/usr/bin/env python3
"""M01-only Bonn continuous quality pass; no timing or blind-test claims."""
import argparse,csv,gc,hashlib,json,os,shutil,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from contract import digest,read_json,require,validate_config,validate_manifest,verify_counts
from m04_branch import branch_step

ROOT=Path.cwd().resolve()

def write_json(p,obj):
    tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def verify(plan):
    for i,(p,h) in enumerate(plan['input_sha256'].items(),1):
        require(Path(p).is_file() and digest(p)==h,'Frozen input changed: '+p)
        if i%500==0:print('[VERIFY]',i,'/',len(plan['input_sha256']),flush=True)
    for p in plan['expected_absent_depth']:require(not Path(p).exists(),'Previously missing GT is now present: '+p)

def environment(rt):
    t=rt.torch;backends=getattr(t,'backends',None);cudnn=getattr(backends,'cudnn',None);cuda=getattr(backends,'cuda',None);matmul=getattr(cuda,'matmul',None)
    props=t.cuda.get_device_properties(0)
    return dict(python=sys.version.split()[0],torch=str(getattr(t,'__version__','unknown')),cuda=str(getattr(getattr(t,'version',None),'cuda','unknown')),
        numpy=str(getattr(rt.np,'__version__','unknown')),opencv=str(getattr(rt.cv2,'__version__','unknown')),
        gpu_name=str(getattr(props,'name','unknown')),gpu_uuid=str(getattr(props,'uuid','unknown')),
        cudnn_benchmark=getattr(cudnn,'benchmark',None),cudnn_allow_tf32=getattr(cudnn,'allow_tf32',None),matmul_allow_tf32=getattr(matmul,'allow_tf32',None))

def state_for(torch,cp,family):
    # Deserialization allowed only for the explicitly chosen, hash-frozen local checkpoint.
    obj=torch.load(cp['path'],map_location='cpu',weights_only=False)
    key={'P':'repair_head_state_dict','O_multi':'adapter_state_dict','O_fast':'state'}[family]
    require(isinstance(obj,dict) and isinstance(obj.get(key),dict) and bool(obj[key]),'Unsupported '+family+' checkpoint structure')
    tensors=list(obj[key].values());require(all(torch.is_tensor(x) and bool(torch.isfinite(x).all().item()) for x in tensors),'Invalid checkpoint tensor')
    require(sum(x.numel() for x in tensors)==(10465 if family=='O_fast' else 170496),'Unexpected head parameter count')
    return obj[key]

def metric(np,pred,record,cv2,depth_min,depth_max):
    status=record['gt_match_status'].strip().upper()
    out=dict(gt_status=status,valid_pixels=0,mask_sha256='',prediction_status='ok',AbsRel='',RMSE='',delta1='')
    if status!='MATCHED':
        if not np.isfinite(pred).all() or not (pred>0).all():out['prediction_status']='invalid_unscored'
        return out
    raw=cv2.imread(str(record['depth_path']),cv2.IMREAD_UNCHANGED)
    require(raw is not None and raw.dtype==np.uint16 and raw.ndim==2 and tuple(raw.shape)==tuple(pred.shape),'Bonn GT dtype/registration/geometry mismatch')
    gt=raw.astype(np.float32)/5000.;mask=np.isfinite(gt)&(gt>=depth_min)&(gt<=depth_max);out['valid_pixels']=int(mask.sum());out['mask_sha256']=hashlib.sha256(mask.tobytes()).hexdigest()
    if not mask.any():out['prediction_status']='empty_GT_mask';return out
    x=pred[mask];y=gt[mask]
    if not (np.isfinite(x)&(x>0)).all():out['prediction_status']='invalid_prediction';return out
    x=x.astype('float64');y=y.astype('float64')
    out.update(AbsRel=float(np.mean(abs(x-y)/y)),RMSE=float(np.sqrt(np.mean((x-y)**2))),delta1=float(np.mean(np.maximum(x/y,y/x)<1.25)))
    return out

def temporal_metric(np,pred,record,cv2,depth_min,depth_max,previous):
    status=record['gt_match_status'].strip().upper()
    blank=dict(temporal_status='no_current_GT',temporal_valid_pixels=0,temporal_depth_change_mae_m='')
    if status!='MATCHED':return blank,None
    raw=cv2.imread(str(record['depth_path']),cv2.IMREAD_UNCHANGED)
    require(raw is not None and raw.dtype==np.uint16 and raw.ndim==2 and tuple(raw.shape)==tuple(pred.shape),'Bonn GT dtype/registration/geometry mismatch')
    gt=raw.astype(np.float32)/5000.;mask=np.isfinite(gt)&(gt>=depth_min)&(gt<=depth_max)
    valid=bool((np.isfinite(pred[mask])&(pred[mask]>0)).all()) and bool(mask.any())
    current=(gt,mask,pred.copy(),valid)
    if previous is None:return dict(blank,temporal_status='no_previous_GT'),current
    prev_gt,prev_mask,prev_pred,prev_valid=previous;common=mask&prev_mask;count=int(common.sum())
    if not count:return dict(blank,temporal_status='empty_intersection'),current
    if not valid or not prev_valid:return dict(blank,temporal_status='invalid_prediction_or_GT',temporal_valid_pixels=count),current
    delta=(pred[common]-prev_pred[common])-(gt[common]-prev_gt[common])
    return dict(temporal_status='ok',temporal_valid_pixels=count,temporal_depth_change_mae_m=float(np.mean(np.abs(delta)))),current

class Predictor:
    def __init__(self,rt,plan,unit,records):
        from y08y09_models import Stream,GainHead,FastDepthAdapter
        from velodepth.models.persistent_feature_repair import PersistentFeatureRepairHead
        self.rt=rt;self.unit=unit;self.engine=None;self.stream=None;self.model=None;self.adapter=None;self.state=None;self.prev=None;self.fast_steps=0
        t=rt.torch;family=unit['family'];seed=unit['seed'];cp=None
        if family!='A':cp=next(x for x in plan['checkpoints'][family] if str(x['seed'])==seed)
        state=None if cp is None else state_for(t,cp,family)
        if unit['experiment']=='M04':
            from y10_runtime import Engine
            ctx=SimpleNamespace(rt=rt,args=SimpleNamespace(snapshot=plan['snapshot']),head_state=state,manifests={'run':records})
            self.engine=Engine(ctx,'run');self.shape=self.engine.shape
        else:
            self.model=rt.gt2.load_model(Path(plan['snapshot']),'pre_readout' if family=='P' else 'off',t.device('cuda'))
            if family=='P':
                h=self.model.pixel_decoder.persistent_feature_repair;h.load_state_dict(state,strict=True);self.model.pixel_decoder.persistent_feature_repair=GainHead(h,unit['gain'])
            elif family=='O_multi':
                self.adapter=PersistentFeatureRepairHead(channels=512,bottleneck=32,levels=4).cuda();self.adapter.load_state_dict(state,strict=True)
            elif family=='O_fast':
                self.adapter=FastDepthAdapter().cuda();self.adapter.load_state_dict(state,strict=True)
            self.model.eval()
            for p in self.model.parameters():p.requires_grad_(False)
            if self.adapter is not None:
                self.adapter.eval()
                for p in self.adapter.parameters():p.requires_grad_(False)
            self.shape=rt.exp1.shape_info_for(self.model,Path(records[0]['rgb_path']),t.device('cuda'))
            cfg=dict(family=family,schedule='K'+str(unit['k']),gain=unit['gain'],frequency=unit['frequency'])
            self.stream=Stream(rt,self.model,cfg,self.shape,adapter=self.adapter)
    def step(self,i,record):
        if self.stream is not None:
            with self.rt.torch.no_grad():return self.stream.step(i,record)
        from y10_runtime import clone_state
        e=self.engine;frame=e.load(i,False);static=e.static(self.prev,frame,'P') if i%16 else None
        def do(repeats):
            state,depth,_,_=e.step(self.state,self.prev,frame,16,'P',repeats=repeats,static=static)
            return state,depth
        self.state,pred,n=branch_step(self.unit['arm'],i,16,do,clone_state);self.fast_steps+=n;self.prev=frame
        return pred,dict(keyframe=i%16==0,age=i%16)
    def counts(self):
        return dict(self.stream.counts) if self.stream is not None else dict(physical_fast_steps=self.fast_steps,repair_head_calls=self.engine.control.calls,diagnostic_note='OUTPUT_ONLY/F_ONLY repeat entire fast path')
    def finite_state(self):
        from y10_runtime import finite_state
        state=self.stream.state if self.stream is not None else self.state
        return state is not None and finite_state(self.rt.torch,state)
    def reset(self):
        if self.stream is not None:self.stream.reset()
        if self.engine is not None:self.engine.control.calls=0
        self.state=None;self.prev=None;self.fast_steps=0
    def close(self):
        if self.engine is not None:self.engine.close();self.engine=None
        self.stream=None;self.model=None;self.adapter=None;gc.collect();self.rt.torch.cuda.empty_cache()

def run(plan_path,output):
    plan_path=Path(plan_path).resolve();plan=read_json(plan_path);require(plan.get('status')=='FROZEN_QUALITY_ONLY','Expected frozen Bonn quality plan')
    if plan.get('scope')=='M01_ONLY':
        require(plan['units']==validate_config(plan),'Frozen M01 units differ from configuration')
        expected_blind='VERIFIED_UNSEEN' if plan['role']=='heldout' else 'HISTORY_UNVERIFIED_NOT_BLIND_VERIFIED'
        require(plan.get('blindness')==expected_blind,'Blindness label differs from role')
    else:
        require(plan.get('scope')=='M01_DEV_SMOKE' and plan.get('role')=='development_smoke','Unknown plan scope')
        require(len(plan['units'])==4 and {u['family'] for u in plan['units']}=={'A','O_fast','O_multi','P'} and all(u['k']==16 and u['experiment']=='M01_M02' for u in plan['units']),'Invalid development smoke units')
    output=Path(output).resolve();require(not output.exists(),'Output directory must be new');output.mkdir(parents=True)
    write_json(output/'invocation.json',dict(plan=str(plan_path),plan_sha256=digest(plan_path),role=plan['role'],status='RUNNING_QUALITY_ONLY',started_at_utc=datetime.now(timezone.utc).isoformat()))
    failures=[];done=[];completion_hashes={};replay=[];access=None
    try:
        print('[PREFLIGHT] hashing frozen inputs before any GPU work',flush=True);verify(plan)
        sys.path[:0]=[str(ROOT),str(ROOT/'experiments/p1')]
        from gt4a2_y11_fallback import gpu_lock_path
        from gt4a2_y04a_native_pre_readout_eval import load_runtime
        import fcntl
        uuid,lock_path=gpu_lock_path()
        with lock_path.open('a+') as lock:
            print('[GPU QUEUE] waiting for shared GPU lock',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
            rt=load_runtime();t=rt.torch;np=rt.np
            require(t.cuda.is_available() and str(getattr(t.cuda.get_device_properties(0),'uuid','')).removeprefix('GPU-')==uuid,'CUDA device differs from GPU lock')
            t.set_num_threads(4);runtime_environment=environment(rt)
            for s in plan['sequences']:
                seq=s['sequence'];records=rt.exp1.read_manifest(Path(s['manifest']),training=False)
                require(len(records)==s['RGB'],'Manifest RGB count changed')
                common=None
                for unit in plan['units']:
                    uid=unit['id'];folder=output/'quality'/seq/uid;folder.mkdir(parents=True,exist_ok=False);stage=seq+'/'+uid
                    try:
                        predictor=Predictor(rt,plan,unit,records);rows=[];files={};previous=None
                        try:
                            with (folder/'frames.csv').open('w',encoding='utf-8',newline='') as f:
                                fields=['index','gt_status','keyframe','age','valid_pixels','mask_sha256','prediction_status','AbsRel','RMSE','delta1','temporal_status','temporal_valid_pixels','temporal_depth_change_mae_m','prediction_sha256']
                                w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
                                for i,r in enumerate(records):
                                    if plan['role'] in ('heldout','heldout_history_unverified') and access is None:
                                        access=dict(first_inference_utc=datetime.now(timezone.utc).isoformat(),sequence=seq,unit=uid,index=i,first_matched_GT_read_utc=None)
                                        write_json(output/'test_access.json',access)
                                    pred,meta=predictor.step(i,r);require(tuple(pred.shape)==tuple(predictor.shape['original_shape']),'Prediction geometry differs')
                                    require(predictor.finite_state(),f'Nonfinite recurrent state at RGB index {i}; unit stopped')
                                    require(meta['keyframe']==(i%unit['k']==0) and int(meta['age'])==i%unit['k'],'Schedule mismatch')
                                    if access is not None and access['first_matched_GT_read_utc'] is None and r['gt_match_status'].strip().upper()=='MATCHED':
                                        access['first_matched_GT_read_utc']=datetime.now(timezone.utc).isoformat();write_json(output/'test_access.json',access)
                                    row=dict(index=i,keyframe=meta['keyframe'],age=meta['age'],prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),**metric(np,pred,r,rt.cv2,plan['depth_min_m'],plan['depth_max_m']))
                                    temporal,previous=temporal_metric(np,pred,r,rt.cv2,plan['depth_min_m'],plan['depth_max_m'],previous);row.update(temporal)
                                    if shutil.disk_usage(folder).free<pred.nbytes+1024**3:raise RuntimeError('Insufficient output disk space')
                                    dp=folder/'depth';dp.mkdir(exist_ok=True);path=dp/f'{i:06d}.npz';tmp=path.with_name(path.name+'.tmp')
                                    with tmp.open('wb') as f1:np.savez_compressed(f1,depth=pred);f1.flush();os.fsync(f1.fileno())
                                    os.replace(tmp,path);files[path.relative_to(output).as_posix()]=digest(path)
                                    rows.append(row);w.writerow(row)
                                    if i%100==0 or i+1==len(records):f.flush();print('[QUALITY]',stage,i+1,'/',len(records),flush=True)
                            verify_counts(unit,len(records),predictor.counts())
                            masks=[(r['index'],r['gt_status'],r['mask_sha256'],r['valid_pixels']) for r in rows]
                            if common is None:common=masks
                            else:require(common==masks,'GT frames or GT-only masks differ between methods')
                            complete=dict(unit=unit,sequence=seq,RGB=len(rows),valid=sum(r['AbsRel']!='' for r in rows),
                                failed=sum(r['prediction_status']!='ok' for r in rows),counts=predictor.counts(),frames_sha256=digest(folder/'frames.csv'),prediction_sha256=files)
                            write_json(folder/'complete.json',complete);done.append(stage);completion_hashes[stage]=digest(folder/'complete.json')
                            if complete['failed']:
                                failures.append(dict(unit=unit,sequence=seq,error=f"{complete['failed']} frames have invalid predictions; see frames.csv"))
                        finally:predictor.close()
                    except Exception as e:
                        gc.collect();t.cuda.empty_cache()
                        fail=dict(unit=unit,sequence=seq,error=str(e),traceback=traceback.format_exc());write_json(folder/'failure.json',fail);failures.append(fail);print('[UNIT FAILED]',stage,str(e),flush=True)
            # Full-video P_NATIVE is checked against the actual M01 P trajectory for the same checkpoint.
            for s in plan['sequences']:
                seq=s['sequence']
                for cp in plan['checkpoints']['P']:
                    seed=str(cp['seed']);p=next(u for u in plan['units'] if u['experiment']=='M01_M02' and u['family']=='P' and u['seed']==seed and u['k']==16 and u['gain']==1 and u['frequency']==1)
                    m=f'M04__P_NATIVE__s{seed}'
                    if seq+'/'+p['id'] not in done or seq+'/'+m not in done:continue
                    mismatched=[];max_error=0.
                    for i in range(s['RGB']):
                        with np.load(output/'quality'/seq/p['id']/'depth'/f'{i:06d}.npz',allow_pickle=False) as x:a=x['depth']
                        with np.load(output/'quality'/seq/m/'depth'/f'{i:06d}.npz',allow_pickle=False) as x:b=x['depth']
                        valid=bool(np.isfinite(a).all() and np.isfinite(b).all())
                        if valid:max_error=max(max_error,float(np.max(np.abs(a-b))))
                        if not valid or not np.allclose(a,b,rtol=plan['native_replay_rtol'],atol=plan['native_replay_atol'],equal_nan=False):mismatched.append(i)
                    item=dict(sequence=seq,seed=seed,P_unit=p['id'],M04_unit=m,frames=s['RGB'],mismatched_frames=mismatched,max_finite_abs_difference=max_error,rtol=plan['native_replay_rtol'],atol=plan['native_replay_atol'])
                    replay.append(item)
                    if mismatched:
                        failures.append(dict(unit=dict(id=m),sequence=seq,error='P_NATIVE full-trajectory replay mismatch; see m04_native_replay.json'))
            write_json(output/'m04_native_replay.json',replay)
            print('[FINAL VERIFY] checking all frozen inputs',flush=True);verify(plan)
        result=dict(status='QUALITY_ONLY_COMPLETE' if not failures else 'QUALITY_INCOMPLETE',role=plan['role'],
            plan_sha256=digest(plan_path),completed_units=done,completion_hashes=completion_hashes,m04_native_replay_sha256=digest(output/'m04_native_replay.json'),failed_units=[dict(sequence=x['sequence'],unit=x['unit']['id'],error=x['error']) for x in failures],
            formal_v21_complete=False,timing_performed=False,statistical_intervals_computed=False,environment=runtime_environment,
            warning='M01 quality pass only; history-unverified means not a verified blind test; preserve all failed units.')
        write_json(output/'summary.json',result);print('[QUALITY END]',result['status'],output,flush=True);return 0 if not failures else 2
    except BaseException as e:
        write_json(output/'failure.json',dict(error=str(e),traceback=traceback.format_exc()));raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--out-dir',required=True);a=ap.parse_args();return run(a.plan,a.out_dir)
if __name__=='__main__':raise SystemExit(main())
